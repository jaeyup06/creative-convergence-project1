# src/server/server.py
# 의료진 서버 메인 - TCP/UDP 통합 소켓 서버

import socket
import threading
import queue
import time
import json
import os
import tkinter as tk
import numpy as np
import cv2
import sounddevice as sd
from src.common.config import SERVER_IP, TCP_PORT, UDP_PORT, VIDEO_WIDTH, VIDEO_HEIGHT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE
from src.common.packet_format import PKT_PATIENT_VIDEO, PKT_PATIENT_AUDIO, PKT_DOCTOR_VIDEO, PKT_DOCTOR_AUDIO, VIDEO_PACKET_COUNT, VIDEO_PACKET_SIZE
from src.server.video_receiver import receive_video
from src.server.audio_analyzer import receive_audio, start_session, snapshot_session
from src.server.gui_server import ServerGUI
from src.server.session_recorder import save_excel, save_audio
from src.recognition.voice_analyzer import analyze_voice
from src.recognition.pronunciation import score_pronunciation

PATIENTS_FILE = "data/sessions/patients.json"

# TCP 소켓
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp_sock.bind((SERVER_IP, TCP_PORT))

# UDP 소켓
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
udp_sock.bind((SERVER_IP, UDP_PORT))

# 패킷 분류 Queue
video_queue = queue.Queue()
audio_queue = queue.Queue()

# 전역 변수
client_conn = None
client_udp_addr = None
gui: ServerGUI = None

# 마지막으로 수신한 비대칭 지수 (세션 종료 시 엑셀 저장용)
latest_asymmetry = None
latest_asym_diff = None
# 마지막으로 환자에게 보낸 발화 문장 (발음 정확도 DTW 비교 대상)
latest_sentence = None

# 제어 이벤트
camera_event = threading.Event()
doctor_audio_event = threading.Event()
patient_mute_event = threading.Event()


def _load_patients() -> list:
    if not os.path.exists(PATIENTS_FILE):
        return []
    with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("patients", [])

def _find_patient_by_ip(ip: str) -> dict:
    for p in _load_patients():
        if p.get("ip") == ip:
            return p
    return None

def udp_dispatcher():
    global client_udp_addr
    print(f"UDP 수신 대기 중 - 포트 {UDP_PORT}")
    while True:
        try:
            data, addr = udp_sock.recvfrom(VIDEO_PACKET_SIZE + 2)
            if not data:
                continue
            client_udp_addr = addr
            pkt_type = data[0]
            if pkt_type == PKT_PATIENT_VIDEO:
                video_queue.put(data)
            elif pkt_type == PKT_PATIENT_AUDIO:
                audio_queue.put(data)
        except OSError:
            break

def _handle_result_message(line: str) -> bool:
    global latest_asymmetry, latest_asym_diff

    if not line.startswith("RESULT:ASYMMETRY:"):
        return False

    parts = line.split(":")
    try:
        asymmetry = float(parts[2])
    except (IndexError, ValueError):
        return True 

    asym_diff = None
    if len(parts) > 3 and parts[3] != "":
        try:
            asym_diff = float(parts[3])
        except ValueError:
            asym_diff = None

    latest_asymmetry = asymmetry
    latest_asym_diff = asym_diff

    if gui:
        gui.root.after(0, lambda a=asymmetry: gui.update_metrics({"asymmetry": a}))

    return True

def handle_tcp():
    global client_conn, client_udp_addr, gui
    tcp_sock.listen(1)
    print(f"TCP 대기 중 - 포트 {TCP_PORT}")
    conn, addr = tcp_sock.accept()
    client_conn = conn
    client_ip = addr[0]
    print(f"클라이언트 접속: {addr}")

    try:
        first = conn.recv(1024).decode().strip()
        if first.startswith("UDP_PORT:"):
            udp_port = int(first.split(":")[1])
            client_udp_addr = (client_ip, udp_port)
            print(f"클라이언트 UDP 주소 설정: {client_udp_addr}")
    except (OSError, ValueError):
        pass

    patient = _find_patient_by_ip(client_ip)
    if patient:
        if gui:
            gui.root.after(0, lambda: gui._set_patient(patient))
    else:
        if gui:
            gui.root.after(0, lambda: gui.show_new_patient_popup(client_ip))

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            raw = data.decode().strip()

            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if _handle_result_message(line):
                    continue

                if line == "CMD:PATIENT_CAM_OFF":
                    if gui:
                        gui.root.after(0, gui._clear_patient_frame)
                elif line == "CMD:PATIENT_CAM_ON":
                    if gui:
                        gui.root.after(0, gui._on_patient_camera_on)
                else:
                    print(f"수신 데이터: {line}")
        except OSError:
            break

    conn.close()
    print("TCP 연결 종료")

def send_message(msg: str):
    global client_conn, latest_sentence
    if msg.startswith("SENTENCE:"):
        parts = msg.split(":", 2)
        if len(parts) == 3 and parts[2].strip():
            latest_sentence = parts[2].strip()
    if client_conn:
        try:
            client_conn.sendall(msg.encode())
        except OSError:
            pass

def capture_doctor_video():
    cap = None
    while True:
        if camera_event.is_set():
            if cap is None:
                cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                continue
            if gui:
                gui.root.after(0, lambda f=frame: gui.update_doctor_frame(f))
            if client_udp_addr:
                resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                d = resized.flatten().tobytes()
                for i in range(VIDEO_PACKET_COUNT):
                    packet = bytes([PKT_DOCTOR_VIDEO, i]) + d[i * VIDEO_PACKET_SIZE:(i + 1) * VIDEO_PACKET_SIZE]
                    udp_sock.sendto(packet, client_udp_addr)
        else:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.1)

def capture_doctor_audio():
    global client_udp_addr
    
    def callback(indata, frames, time_info, status):
        # 마이크가 꺼져있거나 클라이언트 연결이 없으면 게이지를 0으로 만들고 패킷 전송 안 함
        if not doctor_audio_event.is_set() or not client_udp_addr:
            if gui:
                gui.root.after(0, lambda: gui.update_doctor_volume(0))
            return

        # 의료진 본인 화면(서버)의 음량 게이지 업데이트
        if gui:
            rms = int(np.sqrt(np.mean(indata.astype(np.float32)**2)))
            volume = min(int(rms / 300 * 100), 100)
            gui.root.after(0, lambda v=volume: gui.update_doctor_volume(v))

        packet = bytes([PKT_DOCTOR_AUDIO]) + indata.tobytes()
        try:
            udp_sock.sendto(packet, client_udp_addr)
        except OSError:
            pass

    try:
        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16, 
                            blocksize=AUDIO_CHUNK_SIZE, callback=callback):
            while True:
                sd.sleep(100)
    except Exception as e:
        print(f"[Server] 의료진 마이크 오류: {e}")

def on_session_start():
    global latest_asymmetry, latest_asym_diff, latest_sentence
    print("세션 시작")
    if gui and gui.current_patient:
        gui.session_active = True
        start_session()
        latest_asymmetry = None
        latest_asym_diff = None
        latest_sentence = None
        send_message("CMD:START_SESSION")

def _finalize_session(name: str, audio_bytes: bytes, target_text: str, asymmetry):
    audio_np = (np.frombuffer(audio_bytes, dtype=np.int16)
                if audio_bytes else np.array([], dtype=np.int16))

    voice = analyze_voice(audio_np, AUDIO_SAMPLE_RATE)
    accuracy = score_pronunciation(audio_np, target_text, AUDIO_SAMPLE_RATE) if target_text else None

    # 분석이 끝나면 환자 클라이언트에게도 결과를 전송한다
    send_message(f"RESULT:AUDIO:{accuracy}:{voice.get('speech_rate')}:{voice.get('silence_sec')}")

    if gui:
        gui.root.after(0, lambda: gui.update_metrics({
            "asymmetry": asymmetry,
            "pronunciation": accuracy,
            "speech_rate": voice.get("speech_rate"),
            "silence": voice.get("silence_sec"),
        }))

    session_data = {
        "비대칭 지수": asymmetry if asymmetry is not None else "-",
        "발음 정확도": accuracy if accuracy is not None else "-",
        "발화 속도": voice.get("speech_rate") if voice.get("speech_rate") is not None else "-",
        "음량": voice.get("volume") if voice.get("volume") is not None else "-",
        "묵음 구간": voice.get("silence_sec") if voice.get("silence_sec") is not None else "-",
    }
    save_excel(name, session_data)
    if audio_bytes:
        save_audio(name, audio_bytes, AUDIO_SAMPLE_RATE)
    print(f"[Server] 세션 분석 완료: {name} / 발음정확도={accuracy} / {voice}")

def on_session_stop():
    print("세션 종료 및 저장")
    if not (gui and gui.current_patient):
        return
    name = gui.current_patient["name"]
    send_message("CMD:STOP_SESSION")

    audio_bytes = snapshot_session()
    threading.Thread(
        target=_finalize_session,
        args=(name, audio_bytes, latest_sentence, latest_asymmetry),
        daemon=True,
    ).start()

def on_camera_toggle(active: bool):
    if active:
        camera_event.set()
        send_message("CMD:CAMERA_ON")
    else:
        camera_event.clear()
        send_message("CMD:CAMERA_OFF")

def on_doctor_audio_toggle(active: bool):
    if active:
        doctor_audio_event.set()
    else:
        doctor_audio_event.clear()

def on_mute_patient(muted: bool):
    if muted:
        patient_mute_event.set()
    else:
        patient_mute_event.clear()

def on_set_shoulder(left: tuple, right: tuple):
    send_message(f"CMD:SET_SHOULDER:{left[0]},{left[1]},{right[0]},{right[1]}")
    print(f"[Server] 어깨 기준점 좌표 전송 - 왼쪽:{left} 오른쪽:{right}")

def on_save_baseline():
    send_message("CMD:SAVE_BASELINE")
    print("[Server] 베이스라인 저장 요청 전송")


if __name__ == "__main__":
    root = tk.Tk()
    gui = ServerGUI(root)

    gui.on_send_message = send_message
    gui.on_session_start = on_session_start
    gui.on_session_stop = on_session_stop
    gui.on_camera_toggle = on_camera_toggle
    gui.on_doctor_audio_toggle = on_doctor_audio_toggle
    gui.on_mute_patient = on_mute_patient
    gui.on_set_shoulder = on_set_shoulder
    gui.on_save_baseline = on_save_baseline

    def video_callback(frame):
        if gui:
            gui.root.after(0, lambda f=frame: gui.update_patient_frame(f))

    threading.Thread(target=handle_tcp, daemon=True).start()
    threading.Thread(target=udp_dispatcher, daemon=True).start()
    threading.Thread(target=capture_doctor_video, daemon=True).start()
    threading.Thread(target=capture_doctor_audio, daemon=True).start()
    threading.Thread(target=lambda: receive_audio(audio_queue, gui=gui, mute_event=patient_mute_event), daemon=True).start()
    threading.Thread(target=lambda: receive_video(video_queue, frame_callback=video_callback), daemon=True).start()

    print("서버 실행 중")
    root.mainloop()