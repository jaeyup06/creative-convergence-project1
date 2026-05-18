# src/server/server.py
# 의료진 서버 메인 - TCP/UDP 통합 소켓 서버

import socket
import threading
import time
import json
import os
import tkinter as tk
import numpy as np
import cv2
from src.common.config import SERVER_IP, TCP_PORT
from src.server.video_receiver import receive_video
from src.server.audio_analyzer import receive_audio
from src.server.gui_server import ServerGUI
from src.server.session_recorder import save_excel

PATIENTS_FILE = "data/sessions/patients.json"

# 소켓 생성
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcp_sock.bind((SERVER_IP, TCP_PORT))

# 전역 변수
client_conn = None
gui: ServerGUI = None
camera_event = threading.Event()

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

def handle_tcp():
    global client_conn, gui
    tcp_sock.listen(1)
    print(f"TCP 대기 중 - 포트 {TCP_PORT}")
    conn, addr = tcp_sock.accept()
    client_conn = conn
    client_ip = addr[0]
    print(f"클라이언트 접속: {addr}")

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
            print(f"수신 데이터: {data.decode()}")
        except OSError:
            break

    conn.close()
    print("TCP 연결 종료")

def send_message(msg: str):
    global client_conn
    if client_conn:
        client_conn.sendall(msg.encode())

def capture_doctor_video():
    cap = cv2.VideoCapture(0)
    while True:
        if camera_event.is_set():
            ret, frame = cap.read()
            if ret and gui:
                gui.root.after(0, lambda f=frame: gui.update_doctor_frame(f))
        else:
            time.sleep(0.1)
    cap.release()

def on_session_start():
    print("세션 시작")
    if gui and gui.current_patient:
        gui.session_active = True

def on_session_stop():
    print("세션 종료 및 저장")
    if gui and gui.current_patient:
        name = gui.current_patient["name"]
        save_excel(name, {})  # 분석 수치 연결 후 data dict 채울 것

def on_mute_patient(muted: bool):
    print(f"환자 음소거: {muted}")
    # TODO: audio 제어로 교체

def on_mute_doctor(muted: bool):
    print(f"의료진 음소거: {muted}")
    # TODO: audio 제어로 교체

def on_camera_toggle(active: bool):
    if active:
        camera_event.set()
    else:
        camera_event.clear()

if __name__ == "__main__":
    root = tk.Tk()
    gui = ServerGUI(root)

    gui.on_send_message = send_message
    gui.on_session_start = on_session_start
    gui.on_session_stop = on_session_stop
    gui.on_mute_patient = on_mute_patient
    gui.on_mute_doctor = on_mute_doctor
    gui.on_camera_toggle = on_camera_toggle

    def video_callback(frame):
        if gui:
            gui.root.after(0, lambda: gui.update_patient_frame(frame))

    threading.Thread(target=handle_tcp, daemon=True).start()
    threading.Thread(target=capture_doctor_video, daemon=True).start()
    threading.Thread(target=lambda: receive_audio(gui=gui), daemon=True).start()
    threading.Thread(target=lambda: receive_video(frame_callback=video_callback), daemon=True).start()

    print("서버 실행 중")
    root.mainloop()