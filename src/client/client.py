# src/client/client.py
# 환자 클라이언트 메인 - TCP/UDP 통합 소켓 클라이언트

import socket
import threading
from src.common.config import SERVER_IP, TCP_PORT, UDP_AUDIO_PORT, UDP_VIDEO_PORT
from src.client.video_sender import send_video
from src.client.audio_sender import send_audio

# 소켓 생성
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
udp_video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def handle_tcp():
    # 재활 문장 수신 / 분석 결과 송신
    tcp_sock.connect((SERVER_IP, TCP_PORT))
    print(f"서버 접속 완료 - {SERVER_IP}:{TCP_PORT}")

    while True:
        try:
            data = tcp_sock.recv(1024)
            if not data:
                break
            print(f"수신 문장: {data.decode()}")
            # TODO: gui_client.py 연동 (문장 화면 출력)
        except OSError:
            break

    tcp_sock.close()
    print("TCP 연결 종료")

def send_result(result: str):
    # 분석 결과 서버로 송신
    tcp_sock.sendall(result.encode())

if __name__ == "__main__":
    threading.Thread(target=handle_tcp, daemon=True).start()
    threading.Thread(target=send_video, daemon=True).start()
    threading.Thread(target=send_audio, daemon=True).start()

    print("클라이언트 실행 중 - Ctrl+C 로 종료")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("클라이언트 종료")
        tcp_sock.close()
        udp_video_sock.close()
        udp_audio_sock.close()