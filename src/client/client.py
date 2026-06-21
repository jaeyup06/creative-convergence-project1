# src/client/client.py
# 환자 클라이언트 메인 - TCP/UDP 통합 소켓 클라이언트

import socket
import threading
import numpy as np
import sounddevice as sd
from src.common.config import SERVER_IP, TCP_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT
from src.common.packet_format import PKT_DOCTOR_VIDEO, PKT_DOCTOR_AUDIO, VIDEO_PACKET_COUNT, VIDEO_PACKET_SIZE
from src.client.video_sender import (
    send_video, sock as udp_sock,
    request_set_shoulder, request_save_baseline,
    request_start_session, request_stop_session,
)
from src.client.audio_sender import send_audio

# TCP 소켓
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 콜백
doctor_frame_callback = None
on_message_callback = None
doctor_volume_callback = None  # 의료진 음량 표시용 콜백 추가


def handle_tcp():
    tcp_sock.connect((SERVER_IP, TCP_PORT))
    udp_port = udp_sock.getsockname()[1]
    tcp_sock.sendall(f"UDP_PORT:{udp_port}\n".encode())
    print(f"서버 접속 완료 - {SERVER_IP}:{TCP_PORT} / UDP:{udp_port}")

    while True:
        try:
            data = tcp_sock.recv(1024)
            if not data:
                break
            msg = data.decode().strip()

            handled = _handle_command(msg)

            if not handled and on_message_callback:
                on_message_callback(msg)
        except OSError:
            break

    tcp_sock.close()
    print("TCP 연결 종료")


def _handle_command(msg: str) -> bool:
    handled = False
    for line in msg.split("\n"):
        line = line.strip()
        if line.startswith("CMD:SET_SHOULDER"):
            parts = line.split(":")
            if len(parts) >= 3 and "," in parts[2]:
                try:
                    x1, y1, x2, y2 = map(int, parts[2].split(","))
                    request_set_shoulder((x1, y1), (x2, y2))
                    print(f"[Client] 어깨 기준점 좌표 수신 - 왼쪽:({x1},{y1}) 오른쪽:({x2},{y2})")
                except ValueError:
                    print(f"[Client] 어깨 좌표 파싱 실패: {line}")
                    request_set_shoulder()
            else:
                request_set_shoulder()
                print("[Client] 어깨 기준점 설정 요청 수신 (좌표 없음, 자동 추정)")
            handled = True
        elif line == "CMD:SAVE_BASELINE":
            request_save_baseline()
            print("[Client] 베이스라인 저장 요청 수신")
            handled = True
        elif line == "CMD:START_SESSION":
            request_start_session()
            print("[Client] 재활 세션 시작 요청 수신 (자세 가이드 모드 종료)")
            handled = True
        elif line == "CMD:STOP_SESSION":
            request_stop_session()
            print("[Client] 재활 세션 종료 요청 수신 (자세 가이드 모드로 복귀)")
            handled = True
        elif line == "CMD:DOCTOR_MIC_OFF":
            # 의료진 마이크 종료 시 즉시 환자 화면 게이지 0으로 초기화
            if doctor_volume_callback:
                doctor_volume_callback(0)
            handled = True
    return handled


def send_result(result: str):
    tcp_sock.sendall((result + "\n").encode())


def receive_doctor_stream():
    s = [b'\xff' * VIDEO_PACKET_SIZE for _ in range(VIDEO_PACKET_COUNT)]

    stream = sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16)
    stream.start()
    audio_buffer = b''

    print("의료진 스트림 수신 대기 중...")
    while True:
        try:
            data, _ = udp_sock.recvfrom(VIDEO_PACKET_SIZE + 2)
            if not data:
                continue
            pkt_type = data[0]

            if pkt_type == PKT_DOCTOR_VIDEO:
                idx = data[1]
                if idx == 0:
                    s = [b'\xff' * VIDEO_PACKET_SIZE for _ in range(VIDEO_PACKET_COUNT)]
                s[idx] = data[2:VIDEO_PACKET_SIZE + 2]
                if idx == VIDEO_PACKET_COUNT - 1:
                    picture = b''.join(s)
                    frame = np.frombuffer(picture, dtype=np.uint8)
                    frame = frame.reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 3)
                    if doctor_frame_callback:
                        doctor_frame_callback(frame.copy())

            elif pkt_type == PKT_DOCTOR_AUDIO:
                payload = data[1:]
                
                # 즉각적인 UI 반영 (버퍼링 대기 없이 매 패킷마다)
                if doctor_volume_callback:
                    temp_audio = np.frombuffer(payload, dtype=np.int16)
                    rms = int(np.sqrt(np.mean(temp_audio.astype(np.float32)**2)))
                    volume = min(int(rms / 300 * 100), 100)
                    doctor_volume_callback(volume)

                audio_buffer += payload
                if len(audio_buffer) >= AUDIO_CHUNK_SIZE * 8:
                    audio = np.frombuffer(audio_buffer, dtype=np.int16)
                    try:
                        # [핵심] 재생 에러(Underflow 등)가 나도 수신 스레드가 죽지 않도록 방어
                        stream.write(audio)
                    except Exception:
                        pass
                    audio_buffer = b''

        except OSError:
            break

    stream.stop()


if __name__ == "__main__":
    threading.Thread(target=handle_tcp, daemon=True).start()
    threading.Thread(target=receive_doctor_stream, daemon=True).start()
    threading.Thread(target=send_video, daemon=True).start()
    threading.Thread(target=send_audio, daemon=True).start()

    print("클라이언트 실행 중 - Ctrl+C 로 종료")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("클라이언트 종료")
        tcp_sock.close()