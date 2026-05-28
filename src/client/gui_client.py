# src/client/gui_client.py
# 환자용 GUI

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading


class PatientGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("안면 및 구음 재활 모니터링 — 환자")
        self.root.configure(bg="#F5F5F0")
        self.root.resizable(False, False)

        self.camera_active = False
        self._camera_stop = threading.Event()
        self._video_thread = None

        self.audio_active = False
        self._audio_stop = threading.Event()
        self._audio_thread = None

        self._last_frame = None
        self.doctor_camera_active = False

        self.on_patient_camera_on = None
        self.on_patient_camera_off = None

        self._build_ui()

    def _build_ui(self):
        # 헤더
        header = tk.Frame(self.root, bg="#FFFFFF", pady=8)
        header.pack(fill="x")
        tk.Label(header, text="안면 및 구음 재활 모니터링 — 환자",
                 font=("맑은 고딕", 11, "bold"), bg="#FFFFFF", fg="#222222").pack(side="left", padx=12)
        self.status_label = tk.Label(header, text="● 서버 연결됨",
                                     font=("맑은 고딕", 9), bg="#E6F7EE", fg="#22AA66",
                                     padx=10, pady=3)
        self.status_label.pack(side="right", padx=12)

        content = tk.Frame(self.root, bg="#F5F5F0", padx=10, pady=8)
        content.pack(fill="both", expand=True)

        # 재활 문장
        sent_frame = self._section(content, "재활 문장")
        self.sentence_var = tk.StringVar(value="세션 대기 중...")
        tk.Label(sent_frame, textvariable=self.sentence_var,
                 font=("맑은 고딕", 20, "bold"), bg="#F8F8FF",
                 fg="#222222", wraplength=820, justify="center", pady=18
                 ).pack(fill="x", padx=2, pady=2)

        # ── 카메라 행 ──────────────────────────────────────
        cam_row = tk.Frame(content, bg="#F5F5F0")
        cam_row.pack(fill="x", pady=(0, 4))

        # 내 화면
        my_sec = self._section(cam_row, "내 화면")
        my_sec.pack(side="left", fill="both", expand=True, padx=(0, 5))

        patient_holder = tk.Frame(my_sec, width=400, height=300, bg="#E0E0E0")
        patient_holder.pack(pady=(2, 4))
        patient_holder.pack_propagate(False)
        self.patient_canvas = tk.Label(patient_holder, bg="#E0E0E0",
                                       text="카메라 꺼짐", fg="#AAAAAA",
                                       font=("맑은 고딕", 9))
        self.patient_canvas.pack(fill="both", expand=True)

        # 버튼
        btn_frame = tk.Frame(my_sec, bg="#FFFFFF")
        btn_frame.pack(fill="x", padx=2, pady=(0, 2))

        self.cam_btn = tk.Button(btn_frame, text="카메라 켜기",
                                 font=("맑은 고딕", 9), bg="#E0E0E0",
                                 relief="flat", padx=6, pady=4,
                                 command=self._toggle_camera)
        self.cam_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.mic_btn = tk.Button(btn_frame, text="마이크 켜기",
                                 font=("맑은 고딕", 9), bg="#E0E0E0",
                                 relief="flat", padx=6, pady=4,
                                 command=self._toggle_audio)
        self.mic_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.baseline_btn = tk.Button(btn_frame, text="Baseline 설정",
                                      font=("맑은 고딕", 9), bg="#E8F0FF",
                                      relief="flat", padx=6, pady=4,
                                      command=self._set_baseline)
        self.baseline_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.session_btn = tk.Button(btn_frame, text="세션 시작",
                                     font=("맑은 고딕", 9), bg="#E6F7EE",
                                     relief="flat", padx=6, pady=4,
                                     command=self._start_session)
        self.session_btn.pack(side="left", fill="x", expand=True)

        # 자세 가이드 - 버튼 아래
        self.face_var = tk.StringVar(value="—")
        self.face_desc_var = tk.StringVar(value="")
        self.shoulder_var = tk.StringVar(value="—")
        self.shoulder_desc_var = tk.StringVar(value="")
        self.asym_var = tk.StringVar(value="—")
        self.asym_desc_var = tk.StringVar(value="")

        guide_row = tk.Frame(my_sec, bg="#FFFFFF")
        guide_row.pack(fill="x", padx=2, pady=(2, 4))

        self._guide_item(guide_row, "얼굴 중앙 정렬", self.face_var, self.face_desc_var)
        self._guide_item(guide_row, "어깨 수평", self.shoulder_var, self.shoulder_desc_var)
        self._guide_item(guide_row, "비대칭 지수", self.asym_var, self.asym_desc_var)

        # 의료진 화면 + 지시 사항
        doc_sec = self._section(cam_row, "의료진 화면")
        doc_sec.pack(side="left", fill="both", expand=True)

        doctor_holder = tk.Frame(doc_sec, width=400, height=300, bg="#E0E0E0")
        doctor_holder.pack(pady=(2, 4))
        doctor_holder.pack_propagate(False)
        self.doctor_canvas = tk.Label(doctor_holder, bg="#E0E0E0",
                                      text="의료진 카메라 대기 중...", fg="#AAAAAA",
                                      font=("맑은 고딕", 9))
        self.doctor_canvas.pack(fill="both", expand=True)

        tk.Label(doc_sec, text="지시 사항", font=("맑은 고딕", 8),
                 bg="#FFFFFF", fg="#888888").pack(anchor="w")
        chat_wrap = tk.Frame(doc_sec, bg="#CCCCCC", bd=1)
        chat_wrap.pack(fill="x", pady=(2, 4))

        chat_scroll = tk.Scrollbar(chat_wrap)
        chat_scroll.pack(side="right", fill="y")

        self.chat_text = tk.Text(chat_wrap, height=6,
                                  font=("맑은 고딕", 9),
                                  bg="#FFFFFF", fg="#000000",
                                  state="disabled", wrap="word",
                                  yscrollcommand=chat_scroll.set,
                                  relief="flat", bd=6)
        self.chat_text.pack(side="left", fill="x", expand=True)
        chat_scroll.config(command=self.chat_text.yview)

        # 마이크 음량
        vol_frame = self._section(content, "내 마이크 음량")
        self.volume_bar = ttk.Progressbar(vol_frame, orient="horizontal",
                                          length=820, mode="determinate", maximum=100)
        self.volume_bar.pack(fill="x", padx=2, pady=4)

        # 분석 수치
        metrics_frame = self._section(content, "분석 수치")
        row = tk.Frame(metrics_frame, bg="#FFFFFF")
        row.pack(fill="x")

        self.accuracy_var = tk.StringVar(value="—")
        self.speed_var = tk.StringVar(value="—")
        self.silence_var = tk.StringVar(value="—")

        self._metric_cell(row, "발음 정확도", self.accuracy_var)
        self._metric_cell(row, "발화 속도", self.speed_var)
        self._metric_cell(row, "묵음 구간", self.silence_var)

    def _section(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg="#FFFFFF", bd=1, relief="solid", pady=6, padx=8)
        outer.pack(fill="x", pady=(0, 6))
        tk.Label(outer, text=title, font=("맑은 고딕", 8), bg="#FFFFFF",
                 fg="#888888").pack(anchor="w")
        return outer

    def _guide_item(self, parent, label: str, value_var: tk.StringVar, desc_var: tk.StringVar):
        frame = tk.Frame(parent, bg="#F5F5F0", pady=6, padx=12)
        frame.pack(side="left", fill="both", expand=True, padx=3, pady=2)
        tk.Label(frame, text=label, font=("맑은 고딕", 8), bg="#F5F5F0",
                 fg="#888888").pack(anchor="w")
        tk.Label(frame, textvariable=value_var, font=("맑은 고딕", 14, "bold"),
                 bg="#F5F5F0", fg="#222222").pack(anchor="w")
        tk.Label(frame, textvariable=desc_var, font=("맑은 고딕", 8),
                 bg="#F5F5F0", fg="#888888").pack(anchor="w")

    def _metric_cell(self, parent, label: str, value_var: tk.StringVar):
        frame = tk.Frame(parent, bg="#F5F5F0", pady=8, padx=10)
        frame.pack(side="left", fill="both", expand=True, padx=3)
        tk.Label(frame, text=label, font=("맑은 고딕", 8), bg="#F5F5F0",
                 fg="#888888").pack()
        tk.Label(frame, textvariable=value_var, font=("맑은 고딕", 18, "bold"),
                 bg="#F5F5F0", fg="#222222").pack()

    def _toggle_camera(self):
        from src.client.video_sender import send_video
        if not self.camera_active:
            self.camera_active = True
            self._camera_stop.clear()
            self._video_thread = threading.Thread(
                target=send_video,
                kwargs={"frame_callback": self.update_patient_frame,
                        "stop_event": self._camera_stop},
                daemon=True
            )
            self._video_thread.start()
            self.cam_btn.config(text="카메라 끄기", bg="#FFE0E0")
            if self.on_patient_camera_on:
                self.on_patient_camera_on()
        else:
            self.camera_active = False
            self._camera_stop.set()
            self.cam_btn.config(text="카메라 켜기", bg="#E0E0E0")
            self.patient_canvas.config(image="", text="카메라 꺼짐")
            self.patient_canvas.image = None
            if self.on_patient_camera_off:
                self.on_patient_camera_off()

    def _toggle_audio(self):
        from src.client.audio_sender import send_audio
        if not self.audio_active:
            self.audio_active = True
            self._audio_stop.clear()
            self._audio_thread = threading.Thread(
                target=send_audio,
                kwargs={"stop_event": self._audio_stop},
                daemon=True
            )
            self._audio_thread.start()
            self.mic_btn.config(text="마이크 끄기", bg="#FFE0E0")
        else:
            self.audio_active = False
            self._audio_stop.set()
            self.mic_btn.config(text="마이크 켜기", bg="#E0E0E0")

    def _set_baseline(self):
        from src.client.video_sender import analyzer
        if self._last_frame is not None:
            analyzer.calibrate(self._last_frame)

    def _start_session(self):
        import src.client.video_sender as vs
        vs.pose_mode = False
        self.session_btn.config(text="세션 중", bg="#CCEECC", state="disabled")

    def update_sentence(self, text: str):
        self.sentence_var.set(text)

    def set_status(self, connected: bool):
        if connected:
            self.status_label.config(text="● 서버 연결됨", bg="#E6F7EE", fg="#22AA66")
        else:
            self.status_label.config(text="● 연결 끊김", bg="#FDECEA", fg="#CC3333")

    def add_chat_message(self, text: str):
        self.chat_text.config(state="normal")
        self.chat_text.insert("end", text + "\n")
        self.chat_text.see("end")
        self.chat_text.config(state="disabled")

    def handle_server_message(self, msg: str):
        if msg == "CMD:CAMERA_OFF":
            self.root.after(0, self._clear_doctor_frame)
        elif msg == "CMD:CAMERA_ON":
            self.doctor_camera_active = True
        elif msg.startswith("SENTENCE:"):
            parts = msg.split(":", 2)
            if len(parts) == 3:
                n, text = parts[1], parts[2]
                self.root.after(0, lambda t=text: self.update_sentence(t))
                self.root.after(0, lambda n=n, t=text: self.add_chat_message(f"발화 문장 {n} : {t}"))
        elif msg.startswith("INSTRUCT:"):
            parts = msg.split(":", 2)
            if len(parts) == 3:
                n, text = parts[1], parts[2]
                self.root.after(0, lambda n=n, t=text: self.add_chat_message(f"발화 문장 {n} : {t}"))

    def _clear_doctor_frame(self):
        self.doctor_camera_active = False
        self.doctor_canvas.config(image="", text="의료진 카메라 꺼짐")
        self.doctor_canvas.image = None

    def update_patient_frame(self, frame: np.ndarray):
        if not self.camera_active:
            return
        self._last_frame = frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 300))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.root.after(0, lambda p=photo: self._set_patient_photo(p))

    def _set_patient_photo(self, photo):
        if not self.camera_active:
            return
        self.patient_canvas.config(image=photo, text="")
        self.patient_canvas.image = photo

    def update_doctor_frame(self, frame: np.ndarray):
        if not self.doctor_camera_active:
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 300))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.root.after(0, lambda p=photo: self._set_doctor_photo(p))

    def _set_doctor_photo(self, photo):
        if not self.doctor_camera_active:
            return
        self.doctor_canvas.config(image=photo, text="")
        self.doctor_canvas.image = photo

    def update_patient_volume(self, volume: int):
        self.root.after(0, lambda v=volume: self.volume_bar.configure(value=v))

    def update_pose_guide(self, face_ok: bool, face_offset: float,
                          shoulder_ok: bool, shoulder_tilt: float,
                          asymmetry: float, asym_diff: float):
        self.face_var.set("양호 ✓" if face_ok else "기울음 !")
        self.face_desc_var.set(f"편차 {face_offset:.1f}%")
        self.shoulder_var.set("양호 ✓" if shoulder_ok else "기울음 !")
        self.shoulder_desc_var.set(f"기울기 {shoulder_tilt:.1f}%")
        self.asym_var.set(f"{asymmetry:.2f}")
        self.asym_desc_var.set(f"기준 대비 {'+' if asym_diff >= 0 else ''}{asym_diff:.2f}")

    def update_metrics(self, accuracy=None, speed=None, silence=None):
        if accuracy is not None:
            self.accuracy_var.set(f"{accuracy:.1f}%")
        if speed is not None:
            self.speed_var.set(f"{speed:.1f}")
        if silence is not None:
            self.silence_var.set(f"{silence:.1f}s")


if __name__ == "__main__":
    import src.client.client as client_mod

    root = tk.Tk()
    gui = PatientGUI(root)

    client_mod.doctor_frame_callback = gui.update_doctor_frame
    client_mod.on_message_callback = gui.handle_server_message

    gui.on_patient_camera_on = lambda: client_mod.send_result("CMD:PATIENT_CAM_ON")
    gui.on_patient_camera_off = lambda: client_mod.send_result("CMD:PATIENT_CAM_OFF")

    threading.Thread(target=client_mod.handle_tcp, daemon=True).start()
    threading.Thread(target=client_mod.receive_doctor_stream, daemon=True).start()

    root.mainloop()
