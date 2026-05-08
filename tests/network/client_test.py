# tests/network/client_test.py
# TCP/UDP 연결 확인용 더미 클라이언트

import socket
import threading
import time
from src.common.config import SERVER_IP, TCP_PORT, UDP_AUDIO_PORT, UDP_VIDEO_PORT

def test_tcp():
    # TCP 연결 테스트
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_IP, TCP_PORT))
        print(f"TCP 연결 성공 - {SERVER_IP}:{TCP_PORT}")
        
        # 테스트 문장 수신
        while True:
            data = sock.recv(1024)
            if not data:
                break
            print(f"수신 문장: {data.decode()}")
    except Exception as e:
        print(f"TCP 연결 실패: {e}")
    finally:
        sock.close()

def test_udp_video():
    # UDP 영상 송신 테스트 (더미 데이터)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP 영상 송신 테스트 - {SERVER_IP}:{UDP_VIDEO_PORT}")
    
    for i in range(20):
        dummy = bytes([i]) + b'\x00' * 46080
        sock.sendto(dummy, (SERVER_IP, UDP_VIDEO_PORT))
    
    print("UDP 영상 더미 패킷 전송 완료")
    sock.close()

def test_udp_audio():
    # UDP 음성 송신 테스트 (더미 데이터)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP 음성 송신 테스트 - {SERVER_IP}:{UDP_AUDIO_PORT}")
    
    dummy = b'\x00' * 2048
    sock.sendto(dummy, (SERVER_IP, UDP_AUDIO_PORT))
    
    print("UDP 음성 더미 패킷 전송 완료")
    sock.close()

if __name__ == "__main__":
    threading.Thread(target=test_tcp,       daemon=True).start()
    threading.Thread(target=test_udp_video, daemon=True).start()
    threading.Thread(target=test_udp_audio, daemon=True).start()

    time.sleep(5)
    print("테스트 종료")