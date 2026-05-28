# src/common/packet_format.py
import struct

PKT_PATIENT_VIDEO = 0x01
PKT_PATIENT_AUDIO = 0x02
PKT_DOCTOR_VIDEO  = 0x03
PKT_DOCTOR_AUDIO  = 0x04

VIDEO_PACKET_COUNT = 20
VIDEO_PACKET_SIZE  = 640 * 480 * 3 // VIDEO_PACKET_COUNT

def pack_video(frame_bytes): return struct.pack('>I', len(frame_bytes)) + frame_bytes
def unpack_video(data): size = struct.unpack('>I', data[:4])[0]; return data[4:4+size]
def pack_audio(audio_bytes): return audio_bytes
def unpack_audio(data): return data
