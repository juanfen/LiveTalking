###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
import os
import ssl
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response
from voice_pipeline import VoicePipeline

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger
import gc


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None
voice_pipeline = None
voice_turns:Dict[int, int] = {}
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def get_voice_pipeline() -> VoicePipeline:
    global voice_pipeline
    if voice_pipeline is None:
        voice_pipeline = VoicePipeline(opt)
    return voice_pipeline

async def ws_send_json(ws, payload:dict):
    if not ws.closed:
        await ws.send_str(json.dumps(payload, ensure_ascii=False))

def schedule_ws_json(loop, ws, payload:dict):
    if ws.closed:
        return
    asyncio.run_coroutine_threadsafe(ws_send_json(ws, payload), loop)

def process_voice_utterance(pipeline:VoicePipeline, sessionid:int, pcm_bytes:bytes, loop, ws, turn_id:int):
    def notify(payload:dict):
        schedule_ws_json(loop, ws, payload)

    try:
        nerfreal = nerfreals.get(sessionid)
        if nerfreal is None:
            notify({"type": "session_lost", "sessionid": sessionid})
            return

        notify({"type": "asr_start"})
        text = pipeline.transcribe(pcm_bytes, str(sessionid))
        if voice_turns.get(sessionid) != turn_id:
            return
        if not text:
            notify({"type": "asr_empty"})
            return

        notify({"type": "asr", "text": text})

        def on_text(segment:str):
            notify({"type": "answer_chunk", "text": segment})

        def should_stop() -> bool:
            return voice_turns.get(sessionid) != turn_id

        notify({"type": "llm_start"})
        llm_response(text, nerfreal, on_text=on_text, should_stop=should_stop)
        if not should_stop():
            notify({"type": "done"})
    except Exception as e:
        logger.exception("voice utterance failed")
        notify({"type": "error", "message": str(e)})

async def voice_ws(request):
    ws = web.WebSocketResponse(max_msg_size=1024**2 * 16)
    await ws.prepare(request)

    try:
        sessionid = int(request.query.get("sessionid", "0"))
    except ValueError:
        await ws_send_json(ws, {"type": "error", "message": "invalid sessionid"})
        await ws.close()
        return ws

    try:
        pipeline = get_voice_pipeline()
    except Exception as e:
        logger.exception("voice pipeline init failed")
        await ws_send_json(ws, {"type": "error", "message": str(e)})
        await ws.close()
        return ws

    voice_session = pipeline.create_session()
    loop = asyncio.get_event_loop()
    logger.info("voice websocket connected sessionid=%s", sessionid)
    await ws_send_json(ws, {"type": "ready", "sessionid": sessionid})

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            if sessionid not in nerfreals:
                await ws_send_json(ws, {"type": "session_lost", "sessionid": sessionid})
                await ws.close()
                break
            speech_started, utterance = voice_session.accept_pcm(msg.data)
            if speech_started:
                voice_turns[sessionid] = voice_turns.get(sessionid, 0) + 1
                if not getattr(opt, "no_voice_interrupt", False):
                    nerfreal = nerfreals.get(sessionid)
                    if nerfreal is not None:
                        nerfreal.flush_talk()
                await ws_send_json(ws, {"type": "speech_start"})
            if utterance:
                turn_id = voice_turns.get(sessionid, 0)
                await ws_send_json(ws, {"type": "speech_end"})
                loop.run_in_executor(
                    None,
                    process_voice_utterance,
                    pipeline,
                    sessionid,
                    utterance,
                    loop,
                    ws,
                    turn_id,
                )
        elif msg.type == aiohttp.WSMsgType.TEXT:
            try:
                params = json.loads(msg.data)
            except json.JSONDecodeError:
                params = {}
            msg_type = params.get("type")
            if msg_type == "reset":
                voice_session.reset()
                await ws_send_json(ws, {"type": "reset"})
            elif msg_type == "stop":
                utterance = voice_session.flush()
                if utterance:
                    turn_id = voice_turns.get(sessionid, 0)
                    await ws_send_json(ws, {"type": "speech_end"})
                    loop.run_in_executor(
                        None,
                        process_voice_utterance,
                        pipeline,
                        sessionid,
                        utterance,
                        loop,
                        ws,
                        turn_id,
                    )
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error("voice websocket error: %s", ws.exception())

    logger.info("voice websocket closed sessionid=%s", sessionid)
    return ws

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    # if len(nerfreals) >= opt.max_session:
    #     logger.info('reach max session')
    #     return web.Response(
    #         content_type="application/json",
    #         text=json.dumps(
    #             {"code": -1, "msg": "reach max session"}
    #         ),
    #     )
    sessionid = randN(6) #len(nerfreals)
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    #ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
    # 让两个在不同内网的设备能够直接建立连接
    # 创建 RTCPeerConnection
    ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            nerfreals.pop(sessionid, None)
        if pc.connectionState == "closed":
            pcs.discard(pc)
            nerfreals.pop(sessionid, None)
            # gc.collect()
#  获取用户媒体
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()

        if params['type']=='echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type']=='chat':
            asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])                         
            #nerfreals[sessionid].put_msg_txt(res)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def interrupt_talk(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        nerfreals[sessionid].flush_talk()
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)    
        nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def record(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreals[sessionid].start_recording()
        elif params['type']=='end_record':
            nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def is_speaking(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")
    parser.add_argument('--tts', type=str, default='index-stream', help="tts service type") #xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 index-stream azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    parser.add_argument('--INDEX_TTS_URL', type=str, default='http://192.168.99.103:6007/clone_voice')
    parser.add_argument('--INDEX_TTS_VOICE', type=str, default='晓伊')
    parser.add_argument('--INDEX_TTS_SAMPLE_RATE', type=int, default=24000)
    parser.add_argument('--INDEX_TTS_TIMEOUT', type=float, default=60)
    parser.add_argument('--llm_base_url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--llm_model_name', type=str, default='qwen-plus-character')
    parser.add_argument('--llm_api_key', type=str, default=os.getenv('ALI_LLM_API_KEY') or os.getenv('DASHSCOPE_API_KEY') or 'sk-4fbefd0a3bbf4d9f9cc74d48429f698b')
    parser.add_argument('--llm_temperature', type=float, default=0.6)
    parser.add_argument('--llm_max_tokens', type=int, default=300)
    parser.add_argument('--llm_top_p', type=float, default=1.0)
    parser.add_argument('--llm_frequency_penalty', type=float, default=0.0)
    parser.add_argument('--llm_history_rounds', type=int, default=8)
    parser.add_argument('--llm_system_prompt', type=str, default='你是一个自然、口语化的数字人助手，回答要简短、清楚。')
    parser.add_argument('--voice_asr_model_dir', type=str, default='models/SenseVoiceSmall')
    parser.add_argument('--voice_asr_output_dir', type=str, default='tmp')
    parser.add_argument('--voice_vad_model_dir', type=str, default='models/snakers4_silero-vad')
    parser.add_argument('--voice_vad_threshold', type=float, default=0.5)
    parser.add_argument('--voice_vad_threshold_low', type=float, default=0.3)
    parser.add_argument('--voice_min_silence_duration_ms', type=int, default=1000)
    parser.add_argument('--voice_pre_roll_ms', type=int, default=320)
    parser.add_argument('--voice_min_speech_ms', type=int, default=300)
    parser.add_argument('--no_voice_interrupt', action='store_true')
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")
    parser.add_argument('--ssl_certfile', type=str, default='', help="HTTPS certificate file")
    parser.add_argument('--ssl_keyfile', type=str, default='', help="HTTPS private key file")

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt)
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_get("/ws/audio", voice_ws)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    scheme = 'https' if opt.ssl_certfile and opt.ssl_keyfile else 'http'
    logger.info('start %s server; %s://<serverip>:%s/%s', scheme, scheme, opt.listenport, pagename)
    logger.info('voice chat page: %s://<serverip>:%s/voicechat.html', scheme, opt.listenport)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        ssl_context = None
        if opt.ssl_certfile and opt.ssl_keyfile:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(opt.ssl_certfile, opt.ssl_keyfile)
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport, ssl_context=ssl_context)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
