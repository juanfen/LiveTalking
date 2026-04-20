from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from logger import logger


@dataclass
class VoicePipelineConfig:
    asr_model_dir: str = "models/SenseVoiceSmall"
    asr_output_dir: str = "tmp"
    vad_model_dir: str = "models/snakers4_silero-vad"
    vad_threshold: float = 0.5
    vad_threshold_low: float = 0.3
    min_silence_duration_ms: int = 1000
    sample_rate: int = 16000
    vad_frame_samples: int = 512
    vad_window_frames: int = 5
    vad_window_threshold: int = 3
    pre_roll_ms: int = 320
    min_speech_ms: int = 300

    @classmethod
    def from_opt(cls, opt) -> "VoicePipelineConfig":
        return cls(
            asr_model_dir=getattr(opt, "voice_asr_model_dir", cls.asr_model_dir),
            asr_output_dir=getattr(opt, "voice_asr_output_dir", cls.asr_output_dir),
            vad_model_dir=getattr(opt, "voice_vad_model_dir", cls.vad_model_dir),
            vad_threshold=float(getattr(opt, "voice_vad_threshold", cls.vad_threshold)),
            vad_threshold_low=float(
                getattr(opt, "voice_vad_threshold_low", cls.vad_threshold_low)
            ),
            min_silence_duration_ms=int(
                getattr(
                    opt,
                    "voice_min_silence_duration_ms",
                    cls.min_silence_duration_ms,
                )
            ),
            pre_roll_ms=int(getattr(opt, "voice_pre_roll_ms", cls.pre_roll_ms)),
            min_speech_ms=int(getattr(opt, "voice_min_speech_ms", cls.min_speech_ms)),
        )


class SenseVoiceASR:
    def __init__(self, config: VoicePipelineConfig):
        self.config = config
        os.makedirs(config.asr_output_dir, exist_ok=True)

        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as exc:
            raise RuntimeError(
                "FunASR is not installed. Install funasr and modelscope in the "
                "LiveTalking environment before using voice chat."
            ) from exc

        self._postprocess = rich_transcription_postprocess
        self._lock = threading.Lock()
        logger.info("loading SenseVoice ASR model from %s", config.asr_model_dir)
        self.model = AutoModel(
            model=config.asr_model_dir,
            vad_kwargs={"max_single_segment_time": 30000},
            disable_update=True,
            hub="hf",
        )
        logger.info("SenseVoice ASR model loaded")

    def transcribe_pcm(self, pcm_bytes: bytes, session_id: str) -> str:
        if not pcm_bytes:
            return ""

        with self._lock:
            result = self.model.generate(
                input=pcm_bytes,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )

        if not result:
            return ""

        text = result[0].get("text", "")
        text = self._postprocess(text).strip()
        logger.info("ASR session=%s text=%s", session_id, text)
        return text


class SileroVAD:
    def __init__(self, config: VoicePipelineConfig):
        self.config = config
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed. Install torch in the LiveTalking "
                "environment before using voice chat."
            ) from exc

        self.torch = torch
        self._lock = threading.Lock()
        logger.info("loading Silero VAD model from %s", config.vad_model_dir)
        self.model, _ = torch.hub.load(
            repo_or_dir=config.vad_model_dir,
            source="local",
            model="silero_vad",
            force_reload=False,
        )
        self.model.eval()
        logger.info("Silero VAD model loaded")

    def speech_probability(self, frame_bytes: bytes) -> float:
        audio_int16 = np.frombuffer(frame_bytes, dtype="<i2")
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = self.torch.from_numpy(audio_float32)
        with self._lock, self.torch.no_grad():
            return float(self.model(audio_tensor, self.config.sample_rate).item())


class VoiceSession:
    def __init__(self, vad: SileroVAD, config: VoicePipelineConfig):
        self.vad = vad
        self.config = config
        self.frame_bytes = config.vad_frame_samples * 2
        self.frame_ms = config.vad_frame_samples * 1000 / config.sample_rate
        self.pre_roll_frames = max(1, int(config.pre_roll_ms / self.frame_ms))
        self.min_speech_bytes = int(config.sample_rate * config.min_speech_ms / 1000) * 2
        self.reset()

    def reset(self) -> None:
        self.pending = bytearray()
        self.utterance = bytearray()
        self.pre_roll = deque(maxlen=self.pre_roll_frames)
        self.voice_window = deque(maxlen=self.config.vad_window_frames)
        self.in_speech = False
        self.last_is_voice = False
        self.silence_ms = 0.0

    def accept_pcm(self, pcm_bytes: bytes) -> Tuple[bool, Optional[bytes]]:
        self.pending.extend(pcm_bytes)
        speech_started = False
        complete_utterance = None

        while len(self.pending) >= self.frame_bytes:
            frame = bytes(self.pending[: self.frame_bytes])
            del self.pending[: self.frame_bytes]

            prob = self.vad.speech_probability(frame)
            if prob >= self.config.vad_threshold:
                is_voice = True
            elif prob <= self.config.vad_threshold_low:
                is_voice = False
            else:
                is_voice = self.last_is_voice
            self.last_is_voice = is_voice

            self.voice_window.append(is_voice)
            have_voice = (
                self.voice_window.count(True) >= self.config.vad_window_threshold
            )

            if have_voice:
                if not self.in_speech:
                    self.in_speech = True
                    speech_started = True
                    self.utterance.extend(b"".join(self.pre_roll))
                    self.pre_roll.clear()
                self.utterance.extend(frame)
                self.silence_ms = 0.0
            elif self.in_speech:
                self.utterance.extend(frame)
                self.silence_ms += self.frame_ms
                if self.silence_ms >= self.config.min_silence_duration_ms:
                    complete_utterance = self._finish_utterance()
                    break
            else:
                self.pre_roll.append(frame)

        return speech_started, complete_utterance

    def flush(self) -> Optional[bytes]:
        if self.in_speech and self.utterance:
            return self._finish_utterance()
        self.reset()
        return None

    def _finish_utterance(self) -> Optional[bytes]:
        pcm = bytes(self.utterance)
        self.reset()
        if len(pcm) < self.min_speech_bytes:
            return None
        return pcm


class VoicePipeline:
    def __init__(self, opt):
        self.config = VoicePipelineConfig.from_opt(opt)
        self.vad = SileroVAD(self.config)
        self.asr = SenseVoiceASR(self.config)

    def create_session(self) -> VoiceSession:
        return VoiceSession(self.vad, self.config)

    def transcribe(self, pcm_bytes: bytes, session_id: str) -> str:
        return self.asr.transcribe_pcm(pcm_bytes, session_id)
