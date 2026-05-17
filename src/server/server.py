# src/server/server.py
# 의료진 서버 메인 - TCP/UDP 통합 소켓 서버

import socket
import threading
from src.common.config import SERVER_IP, TCP_PORT
from src.server.video_receiver import receive_video
from src.server.audio_analyzer import receive_audio

# 소켓 생성
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 바인딩
tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp_sock.bind((SERVER_IP, TCP_PORT))

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

def send_message(msg: str):
    # 재활 문장 전송
    global client_conn
    if client_conn:
        client_conn.sendall(msg.encode())

if __name__ == "__main__":
    threading.Thread(target=handle_tcp,    daemon=True).start()
    threading.Thread(target=receive_audio, daemon=True).start()

    print("서버 실행 중 - Ctrl+C 로 종료")
    # cv2.imshow는 메인 스레드에서 실행해야 함
    receive_video()