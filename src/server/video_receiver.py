# src/server/video_receiver.py
# UDP 영상 수신 - Queue 기반

import queue
import numpy as np
from src.common.config import VIDEO_WIDTH, VIDEO_HEIGHT
from src.common.packet_format import VIDEO_PACKET_COUNT, VIDEO_PACKET_SIZE


def receive_video(video_queue: queue.Queue, frame_callback=None):
    s = [b'\xff' * VIDEO_PACKET_SIZE for _ in range(VIDEO_PACKET_COUNT)]

    print("영상 수신 대기 중...")
    while True:
        try:
            data = video_queue.get(timeout=1.0)
            idx = data[1]
            if idx == 0:
                s = [b'\xff' * VIDEO_PACKET_SIZE for _ in range(VIDEO_PACKET_COUNT)]
            s[idx] = data[2:VIDEO_PACKET_SIZE + 2]
            if idx == VIDEO_PACKET_COUNT - 1:
                picture = b''.join(s)
                frame = np.frombuffer(picture, dtype=np.uint8)
                frame = frame.reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 3)
                if frame_callback:
                    frame_callback(frame.copy())
        except queue.Empty:
            continue
        except OSError:
            break
