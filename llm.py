import os
import time
from typing import Callable, Optional

from basereal import BaseReal
from logger import logger


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus-character"
DEFAULT_API_KEY = "sk-4fbefd0a3bbf4d9f9cc74d48429f698b"
DEFAULT_SYSTEM_PROMPT = "你是一个自然、口语化的数字人助手，回答要简短、清楚。"
SENTENCE_END = set(",.!?;:，。！？；：\n")


def _get_opt_value(opt, name: str, default):
    value = getattr(opt, name, None)
    if value in (None, ""):
        return default
    return value


def _get_llm_client(nerfreal: BaseReal):
    from openai import OpenAI

    opt = nerfreal.opt
    api_key = _get_opt_value(
        opt,
        "llm_api_key",
        os.getenv("ALI_LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or DEFAULT_API_KEY,
    )
    if not api_key:
        raise RuntimeError(
            "LLM api key is empty. Set ALI_LLM_API_KEY/DASHSCOPE_API_KEY or pass --llm_api_key."
        )

    base_url = _get_opt_value(opt, "llm_base_url", DEFAULT_BASE_URL)
    cache_key = (api_key, base_url)
    if getattr(nerfreal, "_llm_client_key", None) != cache_key:
        nerfreal._llm_client = OpenAI(api_key=api_key, base_url=base_url)
        nerfreal._llm_client_key = cache_key
    return nerfreal._llm_client


def _get_dialogue(nerfreal: BaseReal):
    if not hasattr(nerfreal, "_llm_dialogue"):
        system_prompt = _get_opt_value(
            nerfreal.opt, "llm_system_prompt", DEFAULT_SYSTEM_PROMPT
        )
        nerfreal._llm_dialogue = [{"role": "system", "content": system_prompt}]
    return nerfreal._llm_dialogue


def _trim_dialogue(nerfreal: BaseReal) -> None:
    dialogue = _get_dialogue(nerfreal)
    max_rounds = int(_get_opt_value(nerfreal.opt, "llm_history_rounds", 8))
    system = dialogue[:1]
    rest = dialogue[1:]
    max_messages = max_rounds * 2
    if len(rest) > max_messages:
        nerfreal._llm_dialogue = system + rest[-max_messages:]


def llm_response(
    message: str,
    nerfreal: BaseReal,
    on_text: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
):
    start = time.perf_counter()
    opt = nerfreal.opt
    model_name = _get_opt_value(opt, "llm_model_name", DEFAULT_MODEL)

    try:
        client = _get_llm_client(nerfreal)
        dialogue = _get_dialogue(nerfreal)
        dialogue.append({"role": "user", "content": message})
        _trim_dialogue(nerfreal)
        messages = list(_get_dialogue(nerfreal))

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=int(_get_opt_value(opt, "llm_max_tokens", 300)),
            temperature=float(_get_opt_value(opt, "llm_temperature", 0.6)),
            top_p=float(_get_opt_value(opt, "llm_top_p", 1.0)),
            frequency_penalty=float(_get_opt_value(opt, "llm_frequency_penalty", 0.0)),
        )

        logger.info("llm init time: %.3fs", time.perf_counter() - start)
        pending_text = ""
        full_text = ""
        first = True

        def emit(text: str) -> bool:
            text = text.strip()
            if not text:
                return True
            if should_stop and should_stop():
                return False
            logger.info("llm segment: %s", text)
            if on_text:
                on_text(text)
            nerfreal.put_msg_txt(text)
            return True

        for chunk in completion:
            if should_stop and should_stop():
                logger.info("llm response stopped by newer voice turn")
                return

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            msg = getattr(delta, "content", "") or ""
            if not msg:
                continue

            if first:
                logger.info("llm first chunk time: %.3fs", time.perf_counter() - start)
                first = False

            for char in msg:
                pending_text += char
                full_text += char
                if char in SENTENCE_END and len(pending_text.strip()) >= 10:
                    if not emit(pending_text):
                        return
                    pending_text = ""

        if pending_text.strip():
            if not emit(pending_text):
                return

        if full_text.strip():
            _get_dialogue(nerfreal).append(
                {"role": "assistant", "content": full_text.strip()}
            )
            _trim_dialogue(nerfreal)

        logger.info("llm total time: %.3fs", time.perf_counter() - start)
    except Exception as exc:
        logger.exception("llm response failed")
        fallback = "抱歉，我现在回答失败了，请稍后再试。"
        if on_text:
            on_text(fallback)
        nerfreal.put_msg_txt(fallback)
        logger.error("llm error: %s", exc)
