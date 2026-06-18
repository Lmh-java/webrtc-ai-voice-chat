import copy
import logging
from asyncio import Task
from typing import Optional
import uuid

import librosa
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection
from av import AudioFrame

from playback_stream_track import PlaybackStreamTrack


class State:
    logger = logging.getLogger("pc")

    def __init__(self):
        self.pc: RTCPeerConnection = RTCPeerConnection()
        self.id: str = str(uuid.uuid4())
        self.filename: str = f"{self.id}.wav"
        self.track: Optional[MediaStreamTrack] = None
        self.buffer: list[np.ndarray] = []
        self.recording: bool = False
        self.task: Optional[Task] = None
        self.sample_rate: int = 16000
        self.counter: int = 0
        self.response_player: PlaybackStreamTrack = PlaybackStreamTrack()

    def log_info(self, msg, *args):
        self.logger.info(self.id + " " + msg, *args)

    def append_frame(self, frame: AudioFrame):
        buffer = frame.to_ndarray().flatten().astype(np.int16)
        if self.sample_rate != frame.sample_rate * 2:
            self.sample_rate = frame.sample_rate * 2
        self.buffer.append(buffer)

    def flush_audio(self):
        buffer = np.array(self.buffer).flatten()
        self.log_info("Buffer Size: %s", len(buffer))
        data = copy.deepcopy(buffer)
        data = librosa.util.buf_to_float(data)
        self.buffer = []
        if self.sample_rate != 16000:
            data = librosa.resample(data, orig_sr=self.sample_rate, target_sr=16000)
        return data
