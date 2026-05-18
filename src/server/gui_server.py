# src/server/gui_server.py
# 의료진용 GUI - tkinter 기반

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
import os
from PIL import Image, ImageTk

PATIENTS_FILE = "data/sessions/patients.json"

class ServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("안면 및 구음 재활 모니터링 — 의료진")
        self.root.geometry("1200x900")
        self.root.configure(bg="#F5F5F0")
        self.root.resizable(False, False)

        self.current_patient = None
        self.session_active = False
        self.patient_muted = False
        self.doctor_muted = False

        self.on_send_message = None
        self.on_session_start = None
        self.on_session_stop = None
        self.on_mute_patient = None
        self.on_mute_doctor = None

        self._build_ui()

    def _build_ui(self):
        FONT = ("맑은 고딕", 10)
        FONT_BOLD = ("맑은 고딕", 11, "bold")
        FONT_SMALL = ("맑은 고딕", 9)

        # ── 헤더 ──────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#FFFFFF", pady=10, padx=16)
        header.pack(fill=tk.X, padx=12, pady=(12, 6))

        tk.Label(header, text="안면 및 구음 재활 모니터링 — 의료진",
                 font=FONT_BOLD, bg="#FFFFFF").pack(side=tk.LEFT)

        self.status_label = tk.Label(header, text="환자 미연결",
                                     font=FONT, bg="#F0F0F0",
                                     fg="#888888", padx=10, pady=4)
        self.status_label.pack(side=tk.RIGHT, padx=(0, 8))

        self.patient_btn = tk.Button(header, text="환자 선택",
                                     font=FONT, bg="#F0F0F0",
                                     relief=tk.FLAT, padx=10, pady=4,
                                     command=self._open_patient_popup)
        self.patient_btn.pack(side=tk.RIGHT, padx=(0, 4))

        # ── 영상 영역 ──────────────────────────────────────────
        video_frame = tk.Frame(self.root, bg="#F5F5F0")
        video_frame.pack(fill=tk.X, padx=12, pady=6)

        # 환자 영상
        patient_video_frame = tk.LabelFrame(video_frame, text="환자 영상",
                                             font=FONT, bg="#FFFFFF",
                                             padx=8, pady=8)
        patient_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.patient_canvas = tk.Canvas(patient_video_frame, width=640, height=480,
                                         bg="#E8E8E8")
        self.patient_canvas.pack()
        self.patient_canvas.create_text(320, 240, text="환자 접속 대기 중",
                                         fill="#AAAAAA", font=FONT)

        # 환자 음량 바
        vol_frame1 = tk.Frame(patient_video_frame, bg="#FFFFFF")
        vol_frame1.pack(fill=tk.X, pady=(6, 0))

        tk.Label(vol_frame1, text="환자 음량", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W)
        self.patient_vol_bar = ttk.Progressbar(vol_frame1, length=600,
                                                mode="determinate", maximum=100)
        self.patient_vol_bar.pack(fill=tk.X, expand=True, pady=2)

        # 어깨/베이스라인 버튼
        btn_frame1 = tk.Frame(patient_video_frame, bg="#FFFFFF")
        btn_frame1.pack(fill=tk.X, pady=(4, 0))
        self.shoulder_btn = tk.Button(btn_frame1, text="어깨 기준점 설정",
                                       font=FONT_SMALL, relief=tk.FLAT,
                                       bg="#F0F0F0", padx=8, pady=4, state=tk.DISABLED)
        self.shoulder_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.baseline_btn = tk.Button(btn_frame1, text="베이스라인 저장",
                                       font=FONT_SMALL, relief=tk.FLAT,
                                       bg="#F0F0F0", padx=8, pady=4, state=tk.DISABLED)
        self.baseline_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 내 화면
        doctor_video_frame = tk.LabelFrame(video_frame, text="내 화면",
                                            font=FONT, bg="#FFFFFF",
                                            padx=8, pady=8)
        doctor_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(6, 0))

        self.doctor_canvas = tk.Canvas(doctor_video_frame, width=426, height=320,
                                        bg="#E8E8E8")
        self.doctor_canvas.pack()
        self.doctor_canvas.create_text(213, 160, text="카메라 꺼짐",
                                        fill="#AAAAAA", font=FONT)

        # 의료진 음량 바 (grid 고정)
        vol_frame2 = tk.Frame(doctor_video_frame, bg="#FFFFFF")
        vol_frame2.pack(fill=tk.X, pady=(6, 0))
        vol_frame2.columnconfigure(1, weight=1)

        tk.Label(vol_frame2, text="내 음량", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888", width=8, anchor=tk.W).grid(row=0, column=0)
        self.doctor_mute_btn = tk.Button(vol_frame2, text="음소거",
                                          font=FONT_SMALL, relief=tk.FLAT,
                                          bg="#F0F0F0", width=8,
                                          command=self._toggle_doctor_mute)
        self.doctor_mute_btn.grid(row=0, column=2)

        self.doctor_vol_bar = ttk.Progressbar(vol_frame2, length=220,
                                               mode="determinate", maximum=100)
        self.doctor_vol_bar.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=2)

        self.camera_btn = tk.Button(doctor_video_frame, text="카메라 켜기 / 끄기",
                                     font=FONT_SMALL, relief=tk.FLAT,
                                     bg="#F0F0F0", padx=8, pady=4)
        self.camera_btn.pack(fill=tk.X, pady=(4, 0))

        # 회차별 추이 그래프 (내 화면 박스 안)
        tk.Label(doctor_video_frame, text="회차별 재활 추이", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, pady=(8, 2))
        self.graph_canvas = tk.Canvas(doctor_video_frame, height=100, bg="#FFFFFF",
                                       highlightthickness=0)
        self.graph_canvas.pack(fill=tk.X)
        self.graph_canvas.create_text(200, 50, text="환자 연결 후 표시됩니다",
                                       fill="#AAAAAA", font=FONT_SMALL)

        # ── 문장 전송 + 분석 수치 ──────────────────────────────
        middle_frame = tk.Frame(self.root, bg="#F5F5F0")
        middle_frame.pack(fill=tk.X, padx=12, pady=6)

        msg_frame = tk.LabelFrame(middle_frame, text="재활 문장 전송",
                                   font=FONT, bg="#FFFFFF", padx=8, pady=8)
        msg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.msg_entry = tk.Text(msg_frame, height=2, font=FONT,
                                  state=tk.DISABLED, wrap=tk.WORD)
        self.msg_entry.pack(fill=tk.X)
        self.send_btn = tk.Button(msg_frame, text="전송", font=FONT,
                                   relief=tk.FLAT, bg="#F0F0F0", pady=5,
                                   state=tk.DISABLED, command=self._send_message)
        self.send_btn.pack(fill=tk.X, pady=(6, 0))

        analysis_frame = tk.LabelFrame(middle_frame, text="분석 수치",
                                        font=FONT, bg="#FFFFFF", padx=8, pady=8)
        analysis_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(6, 0))

        self.metrics = {}
        labels = [("비대칭 지수", "asymmetry"), ("발음 정확도", "pronunciation"),
                  ("발화 속도", "speech_rate"), ("묵음 구간", "silence")]

        for i, (label, key) in enumerate(labels):
            cell = tk.Frame(analysis_frame, bg="#F5F5F0", padx=12, pady=8, width=90)
            cell.grid(row=i//2, column=i%2, padx=4, pady=4)
            cell.pack_propagate(False)
            tk.Label(cell, text=label, font=FONT_SMALL,
                     bg="#F5F5F0", fg="#888888").pack()
            val = tk.Label(cell, text="—", font=("맑은 고딕", 16, "bold"),
                           bg="#F5F5F0")
            val.pack()
            self.metrics[key] = val



        # ── 세션 버튼 ──────────────────────────────────────────
        session_frame = tk.Frame(self.root, bg="#F5F5F0")
        session_frame.pack(fill=tk.X, padx=12, pady=(6, 12))

        self.start_btn = tk.Button(session_frame, text="재활 세션 시작",
                                    font=FONT, relief=tk.FLAT,
                                    bg="#F0F0F0", pady=8, state=tk.DISABLED,
                                    command=self._start_session)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self.stop_btn = tk.Button(session_frame, text="세션 종료 및 저장",
                                   font=FONT, relief=tk.FLAT,
                                   bg="#F0F0F0", pady=8, state=tk.DISABLED,
                                   command=self._stop_session)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

    # ── 환자 팝업 ──────────────────────────────────────────────
    def _open_patient_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("환자 선택")
        popup.geometry("320x400")
        popup.configure(bg="#FFFFFF")
        popup.resizable(False, False)

        FONT = ("맑은 고딕", 10)
        FONT_SMALL = ("맑은 고딕", 9)

        tk.Label(popup, text="등록된 환자", font=FONT,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, padx=16, pady=(16, 4))

        patients = self._load_patients()
        listbox = tk.Listbox(popup, font=FONT, height=5)
        listbox.pack(fill=tk.X, padx=16)
        for p in patients:
            listbox.insert(tk.END, f"{p['name']}  ({p['id']})")

        ttk.Separator(popup).pack(fill=tk.X, padx=16, pady=12)

        tk.Label(popup, text="새 환자 등록", font=FONT,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, padx=16)

        name_var = tk.StringVar()
        id_var = tk.StringVar()
        save_var = tk.BooleanVar(value=True)

        name_entry = tk.Entry(popup, textvariable=name_var, font=FONT, fg="#AAAAAA")
        name_entry.insert(0, "이름")
        name_entry.bind("<FocusIn>", lambda e: (name_entry.delete(0, tk.END), name_entry.config(fg="#000000")) if name_var.get() == "이름" else None)
        name_entry.pack(fill=tk.X, padx=16, pady=4)
        id_entry = tk.Entry(popup, textvariable=id_var, font=FONT, fg="#AAAAAA")
        id_entry.insert(0, "ID (예: P004)")
        id_entry.bind("<FocusIn>", lambda e: (id_entry.delete(0, tk.END), id_entry.config(fg="#000000")) if id_var.get() == "ID (예: P004)" else None)
        id_entry.pack(fill=tk.X, padx=16, pady=4)
        tk.Checkbutton(popup, text="환자 정보 저장", variable=save_var,
                       bg="#FFFFFF", font=FONT_SMALL).pack(anchor=tk.W, padx=16)

        def confirm():
            name = name_var.get().strip()
            pid = id_var.get().strip()
            if not name or not pid:
                messagebox.showwarning("입력 오류", "이름과 ID를 입력해주세요.", parent=popup)
                return
            patient = {"name": name, "id": pid}
            if save_var.get():
                self._save_patient(patient)
            self._set_patient(patient)
            popup.destroy()

        btn_frame = tk.Frame(popup, bg="#FFFFFF")
        btn_frame.pack(fill=tk.X, padx=16, pady=12)
        tk.Button(btn_frame, text="취소", font=FONT, relief=tk.FLAT,
                  bg="#F0F0F0", command=popup.destroy).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Button(btn_frame, text="확인", font=FONT, relief=tk.FLAT,
                  bg="#F0F0F0", command=confirm).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

    def show_new_patient_popup(self, ip: str):
        popup = tk.Toplevel(self.root)
        popup.title("신규 환자 접속 감지")
        popup.geometry("320x320")
        popup.configure(bg="#FFFFFF")
        popup.resizable(False, False)

        FONT = ("맑은 고딕", 10)
        FONT_SMALL = ("맑은 고딕", 9)

        tk.Label(popup, text="신규 환자 접속 감지", font=("맑은 고딕", 12, "bold"),
                 bg="#FFFFFF").pack(pady=(16, 4))
        tk.Label(popup, text=f"접속 IP: {ip}", font=FONT,
                 bg="#F5F5F0", fg="#444444", padx=12, pady=6).pack(fill=tk.X, padx=16)
        tk.Label(popup, text="등록되지 않은 환자입니다. 정보를 입력해주세요.",
                 font=FONT_SMALL, bg="#FFFFFF", fg="#888888",
                 wraplength=280).pack(pady=8)

        name_var = tk.StringVar()
        id_var = tk.StringVar()
        save_var = tk.BooleanVar(value=True)

        tk.Entry(popup, textvariable=name_var, font=FONT).pack(fill=tk.X, padx=16, pady=4)
        tk.Entry(popup, textvariable=id_var, font=FONT).pack(fill=tk.X, padx=16, pady=4)
        tk.Checkbutton(popup, text="환자 정보 저장 (다음 접속 시 자동 인식)",
                       variable=save_var, bg="#FFFFFF", font=FONT_SMALL).pack(anchor=tk.W, padx=16)

        def confirm():
            name = name_var.get().strip()
            pid = id_var.get().strip()
            if not name or not pid:
                messagebox.showwarning("입력 오류", "이름과 ID를 입력해주세요.", parent=popup)
                return
            patient = {"name": name, "id": pid, "ip": ip}
            if save_var.get():
                self._save_patient(patient)
            self._set_patient(patient)
            popup.destroy()

        btn_frame = tk.Frame(popup, bg="#FFFFFF")
        btn_frame.pack(fill=tk.X, padx=16, pady=12)
        tk.Button(btn_frame, text="거절", font=FONT, relief=tk.FLAT,
                  bg="#F0F0F0", command=popup.destroy).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Button(btn_frame, text="등록 및 연결", font=FONT, relief=tk.FLAT,
                  bg="#F0F0F0", command=confirm).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

    def _set_patient(self, patient: dict):
        self.current_patient = patient
        self.patient_btn.config(text=f"환자: {patient['name']} ▾")
        self.status_label.config(text="연결됨", fg="#22AA66", bg="#E6F7EE")
        self.start_btn.config(state=tk.NORMAL)
        self.msg_entry.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.shoulder_btn.config(state=tk.NORMAL)
        self.baseline_btn.config(state=tk.NORMAL)

    def _toggle_patient_mute(self):
        self.patient_muted = not self.patient_muted
        self.patient_mute_btn.config(text="음소거 해제" if self.patient_muted else "음소거")
        if self.on_mute_patient:
            self.on_mute_patient(self.patient_muted)

    def _toggle_doctor_mute(self):
        self.doctor_muted = not self.doctor_muted
        self.doctor_mute_btn.config(text="음소거 해제" if self.doctor_muted else "음소거")
        if self.on_mute_doctor:
            self.on_mute_doctor(self.doctor_muted)

    def _send_message(self):
        msg = self.msg_entry.get("1.0", tk.END).strip()
        if msg and self.on_send_message:
            self.on_send_message(msg)
            self.msg_entry.delete("1.0", tk.END)

    def _start_session(self):
        self.session_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        if self.on_session_start:
            self.on_session_start()

    def _stop_session(self):
        self.session_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.on_session_stop:
            self.on_session_stop()

    def update_patient_frame(self, frame: np.ndarray):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.patient_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.patient_canvas.image = photo

    def update_doctor_frame(self, frame: np.ndarray):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (426, 320))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.doctor_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.doctor_canvas.image = photo

    def update_patient_volume(self, volume: int):
        self.patient_vol_bar["value"] = min(volume, 100)

    def update_doctor_volume(self, volume: int):
        self.doctor_vol_bar["value"] = min(volume, 100)

    def update_metrics(self, data: dict):
        mapping = {
            "asymmetry": lambda v: f"{v:.2f}",
            "pronunciation": lambda v: f"{v}%",
            "speech_rate": lambda v: f"{v:.1f}",
            "silence": lambda v: f"{v:.1f}s",
        }
        for key, fmt in mapping.items():
            if key in data and data[key] is not None:
                self.metrics[key].config(text=fmt(data[key]))

    def _load_patients(self) -> list:
        if not os.path.exists(PATIENTS_FILE):
            return []
        with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("patients", [])

    def _save_patient(self, patient: dict):
        os.makedirs(os.path.dirname(PATIENTS_FILE), exist_ok=True)
        patients = self._load_patients()
        if not any(p["id"] == patient["id"] for p in patients):
            patients.append(patient)
        with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
            json.dump({"patients": patients}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    root = tk.Tk()
    app = ServerGUI(root)
    root.mainloop()