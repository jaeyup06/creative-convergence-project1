# src/server/server.py
# 의료진 서버 메인 - TCP/UDP 통합 소켓 서버

import socket
import threading
from src.common.config import SERVER_IP, TCP_PORT, UDP_AUDIO_PORT, UDP_VIDEO_PORT

# 소켓 생성
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
udp_video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 바인딩
tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp_sock.bind((SERVER_IP, TCP_PORT))
udp_video_sock.bind((SERVER_IP, UDP_VIDEO_PORT))
udp_audio_sock.bind((SERVER_IP, UDP_AUDIO_PORT))

# 클라이언트 소켓 (TCP 연결 후 저장)
client_conn = None

def handle_tcp():
    # 재활 문장 전송 / 분석 결과 수신
    global client_conn
    tcp_sock.listen(1)
    print(f"TCP 대기 중 - 포트 {TCP_PORT}")
    conn, addr = tcp_sock.accept()
    client_conn = conn
    print(f"클라이언트 접속: {addr}")

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            print(f"수신 데이터: {data.decode()}")
        except OSError:
            break

    conn.close()
    print("TCP 연결 종료")

def handle_video():
    # UDP 영상 수신
    print(f"UDP 영상 대기 중 - 포트 {UDP_VIDEO_PORT}")
    while True:
        try:
            data, addr = udp_video_sock.recvfrom(65535)
            # TODO: 영상 디코딩 및 출력 (video_receiver.py 연동)
        except OSError:
            break

def handle_audio():
    # UDP 음성 수신
    print(f"UDP 음성 대기 중 - 포트 {UDP_AUDIO_PORT}")
    while True:
        try:
            data, addr = udp_audio_sock.recvfrom(4096)
            # TODO: 음성 재생 및 분석 (audio_analyzer.py 연동)
        except OSError:
            break

def send_message(msg: str):
    # 재활 문장 전송
    global client_conn
    if client_conn:
        client_conn.sendall(msg.encode())

if __name__ == "__main__":
    threading.Thread(target=handle_tcp,   daemon=True).start()
    threading.Thread(target=handle_video, daemon=True).start()
    threading.Thread(target=handle_audio, daemon=True).start()

    print("서버 실행 중 - Ctrl+C 로 종료")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("서버 종료")
        tcp_sock.close()
        udp_video_sock.close()
        udp_audio_sock.close()