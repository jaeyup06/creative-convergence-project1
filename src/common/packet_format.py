# src/common/packet_format.py
# 영상/음성 패킷 구조 정의

import struct

# 영상 패킷: [ 4byte 크기 | N byte JPEG ]
def pack_video(frame_bytes):
    return struct.pack('>I', len(frame_bytes)) + frame_bytes

def unpack_video(data):
    size = struct.unpack('>I', data[:4])[0]
    return data[4:4+size]

# 음성 패킷: [ N byte PCM ]
def pack_audio(audio_bytes):
    return audio_bytes

def unpack_audio(data):
    return data