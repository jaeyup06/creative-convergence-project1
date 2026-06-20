# src/server/gui_server.py
# 의료진용 GUI - tkinter 기반

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
import os
import threading
from PIL import Image, ImageTk

from src.server.weather import get_naver_weather, get_naver_weekly_weather, get_outing_guide

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
        self.camera_active = False
        self.doctor_audio_active = False
        self.sentence_counter = 0
        self.patient_camera_active = True

        self.on_send_message = None
        self.on_session_start = None
        self.on_session_stop = None
        self.on_camera_toggle = None
        self.on_doctor_audio_toggle = None
        # OpenCV 관련 버튼 콜백 (어깨 기준점 / 베이스라인)
        self.on_set_shoulder = None
        self.on_save_baseline = None

        self._build_ui()
        self._load_weather()

    def _section(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg="#FFFFFF", bd=1, relief="solid", pady=6, padx=8)
        tk.Label(outer, text=title, font=("맑은 고딕", 8), bg="#FFFFFF",
                 fg="#888888").pack(anchor="w")
        return outer

    def _build_ui(self):
        FONT = ("맑은 고딕", 10)
        FONT_BOLD = ("맑은 고딕", 11, "bold")
        FONT_SMALL = ("맑은 고딕", 9)

        # ── 헤더 (패딩 없음) ──
        header = tk.Frame(self.root, bg="#FFFFFF", pady=8)
        header.pack(fill=tk.X)

        tk.Label(header, text="안면 및 구음 재활 모니터링 — 의료진",
                 font=FONT_BOLD, bg="#FFFFFF", fg="#222222").pack(side=tk.LEFT, padx=12)

        self.status_label = tk.Label(header, text="환자 미연결",
                                     font=FONT, bg="#F0F0F0",
                                     fg="#888888", padx=10, pady=4)
        self.status_label.pack(side=tk.RIGHT, padx=12)

        self.patient_btn = tk.Button(header, text="환자 선택",
                                     font=FONT, bg="#F0F0F0",
                                     relief=tk.FLAT, padx=10, pady=4,
                                     command=self._open_patient_popup)
        self.patient_btn.pack(side=tk.RIGHT, padx=(0, 4))

        # ── 영상 영역 ──
        video_frame = tk.Frame(self.root, bg="#F5F5F0")
        video_frame.pack(fill=tk.X, padx=12, pady=6)

        # 환자 영상 섹션
        patient_video_frame = self._section(video_frame, "환자 영상")
        patient_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.patient_canvas = tk.Canvas(patient_video_frame, width=640, height=480,
                                        bg="#E8E8E8")
        self.patient_canvas.pack()
        self.patient_canvas.create_text(320, 240, text="환자 접속 대기 중",
                                        fill="#AAAAAA", font=FONT)

        vol_frame1 = tk.Frame(patient_video_frame, bg="#FFFFFF")
        vol_frame1.pack(fill=tk.X, pady=(6, 0))
        tk.Label(vol_frame1, text="환자 음량", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W)
        self.patient_vol_bar = ttk.Progressbar(vol_frame1, length=600,
                                               mode="determinate", maximum=100)
        self.patient_vol_bar.pack(fill=tk.X, expand=True, pady=2)

        btn_frame1 = tk.Frame(patient_video_frame, bg="#FFFFFF")
        btn_frame1.pack(fill=tk.X, pady=(4, 0))

        self.shoulder_btn = tk.Button(btn_frame1, text="어깨 기준점 설정",
                                      font=FONT_SMALL, relief=tk.FLAT,
                                      bg="#F0F0F0", padx=8, pady=4, state=tk.DISABLED,
                                      command=self._set_shoulder)
        self.shoulder_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.baseline_btn = tk.Button(btn_frame1, text="베이스라인 저장",
                                      font=FONT_SMALL, relief=tk.FLAT,
                                      bg="#F0F0F0", padx=8, pady=4, state=tk.DISABLED,
                                      command=self._save_baseline)
        self.baseline_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 내 화면 섹션
        doctor_video_frame = self._section(video_frame, "내 화면")
        doctor_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(6, 0))

        self.doctor_canvas = tk.Canvas(doctor_video_frame, width=426, height=320,
                                       bg="#E8E8E8")
        self.doctor_canvas.pack()
        self.doctor_canvas.create_text(213, 160, text="카메라 꺼짐",
                                       fill="#AAAAAA", font=FONT)

        vol_frame2 = tk.Frame(doctor_video_frame, bg="#FFFFFF")
        vol_frame2.pack(fill=tk.X, pady=(6, 0))
        tk.Label(vol_frame2, text="내 음량", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W)
        self.doctor_vol_bar = ttk.Progressbar(vol_frame2, length=300,
                                              mode="determinate", maximum=100)
        self.doctor_vol_bar.pack(fill=tk.X, pady=2)

        cam_mic_frame = tk.Frame(doctor_video_frame, bg="#FFFFFF")
        cam_mic_frame.pack(fill=tk.X, pady=(4, 0))

        self.camera_btn = tk.Button(cam_mic_frame, text="카메라 켜기",
                                    font=FONT_SMALL, relief=tk.FLAT,
                                    bg="#F0F0F0", padx=8, pady=4,
                                    command=self._toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.mic_btn = tk.Button(cam_mic_frame, text="마이크 켜기",
                                 font=FONT_SMALL, relief=tk.FLAT,
                                 bg="#F0F0F0", padx=8, pady=4,
                                 command=self._toggle_doctor_mic)
        self.mic_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # ── 오늘의 날씨 (네이버 날씨 스크래핑) ──
        weather_frame = tk.Frame(doctor_video_frame, bg="#FFFFFF")
        weather_frame.pack(fill=tk.X, pady=(8, 0))

        weather_header = tk.Frame(weather_frame, bg="#FFFFFF")
        weather_header.pack(fill=tk.X)
        tk.Label(weather_header, text="오늘의 날씨", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(side=tk.LEFT)
        tk.Button(weather_header, text="주간 날씨 ≫", font=("맑은 고딕", 8),
                  relief=tk.FLAT, bg="#F0F0F0", padx=6, pady=2,
                  command=self._open_weekly_weather).pack(side=tk.RIGHT)

        self.weather_temp_var = tk.StringVar(value="—")
        self.weather_desc_var = tk.StringVar(value="날씨 정보를 불러오는 중...")
        self.weather_guide_var = tk.StringVar(value="")

        weather_box = tk.Frame(weather_frame, bg="#F5F5F0", padx=10, pady=8)
        weather_box.pack(fill=tk.X, pady=(2, 0))

        tk.Label(weather_box, textvariable=self.weather_temp_var,
                 font=("맑은 고딕", 16, "bold"), bg="#F5F5F0", fg="#222222").pack(anchor=tk.W)
        tk.Label(weather_box, textvariable=self.weather_desc_var,
                 font=FONT_SMALL, bg="#F5F5F0", fg="#888888").pack(anchor=tk.W)
        tk.Label(weather_box, textvariable=self.weather_guide_var,
                 font=FONT_SMALL, bg="#F5F5F0", fg="#CC3333",
                 wraplength=380, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        tk.Button(weather_frame, text="날씨 새로고침", font=FONT_SMALL, relief=tk.FLAT,
                  bg="#F0F0F0", padx=8, pady=4,
                  command=self._load_weather).pack(fill=tk.X, pady=(4, 0))

        # ── 문장 전송 + 분석 수치 ──
        middle_frame = tk.Frame(self.root, bg="#F5F5F0")
        middle_frame.pack(fill=tk.X, padx=12, pady=6)

        msg_frame = self._section(middle_frame, "재활 문장 전송")
        msg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(msg_frame, text="/s 문장  →  발화 문장 전송    일반 텍스트  →  지시 사항 전송",
                 font=FONT_SMALL, bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, pady=(0, 4))

        self.msg_entry = tk.Text(msg_frame, height=2, font=FONT,
                                 state=tk.DISABLED, wrap=tk.WORD)
        self.msg_entry.pack(fill=tk.X)
        self.msg_entry.bind("<Return>", self._on_enter_key)

        self.send_btn = tk.Button(msg_frame, text="전송  (Enter)",
                                  font=FONT, relief=tk.FLAT,
                                  bg="#F0F0F0", pady=5,
                                  state=tk.DISABLED, command=self._send_message)
        self.send_btn.pack(fill=tk.X, pady=(6, 0))

        analysis_frame = self._section(middle_frame, "분석 수치")
        analysis_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(6, 0))

        self.metrics = {}
        labels = [("비대칭 지수", "asymmetry"), ("발음 정확도", "pronunciation"),
                  ("발화 속도", "speech_rate"), ("묵음 구간", "silence")]

        metrics_row = tk.Frame(analysis_frame, bg="#FFFFFF")
        metrics_row.pack(fill=tk.X)

        for label, key in labels:
            cell = tk.Frame(metrics_row, bg="#F5F5F0", padx=12, pady=8)
            cell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            tk.Label(cell, text=label, font=FONT_SMALL,
                     bg="#F5F5F0", fg="#888888").pack()
            val = tk.Label(cell, text="—", font=("맑은 고딕", 16, "bold"),
                           bg="#F5F5F0")
            val.pack()
            self.metrics[key] = val

        # ── 세션 버튼 ──
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

    # ── 날씨 ──
    def _load_weather(self):
        threading.Thread(target=self._fetch_weather, daemon=True).start()

    def _fetch_weather(self):
        result = get_naver_weather()
        self.root.after(0, lambda: self._apply_weather(result))

    def _apply_weather(self, result: dict):
        temp = result.get("temperature", "-")
        desc = result.get("description", "-")
        self.weather_temp_var.set(f"{temp}°C" if temp != "-" else "—")
        self.weather_desc_var.set(desc)
        self.weather_guide_var.set(get_outing_guide(temp))

    def _open_weekly_weather(self):
        popup = tk.Toplevel(self.root)
        popup.title("주간 날씨")
        popup.geometry("300x420")
        popup.configure(bg="#FFFFFF")
        popup.resizable(False, False)

        tk.Label(popup, text="주간 날씨", font=("맑은 고딕", 11, "bold"),
                 bg="#FFFFFF").pack(pady=(12, 8))

        list_frame = tk.Frame(popup, bg="#FFFFFF")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12)

        loading_label = tk.Label(list_frame, text="불러오는 중...", font=("맑은 고딕", 9),
                                 bg="#FFFFFF", fg="#AAAAAA")
        loading_label.pack(pady=20)

        def fetch():
            data = get_naver_weekly_weather()
            self.root.after(0, lambda: render(data))

        def render(data):
            loading_label.destroy()
            if not data:
                tk.Label(list_frame, text="주간 날씨 정보를 불러올 수 없습니다",
                         font=("맑은 고딕", 9), bg="#FFFFFF", fg="#AAAAAA",
                         wraplength=240).pack(pady=20)
                return
            for day in data:
                row = tk.Frame(list_frame, bg="#FFFFFF")
                row.pack(fill=tk.X, pady=4)

                top_row = tk.Frame(row, bg="#FFFFFF")
                top_row.pack(fill=tk.X)
                tk.Label(top_row, text=day["date"], font=("맑은 고딕", 9),
                         bg="#FFFFFF", fg="#222222", width=10, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(top_row, text=f"{day['temp_min']}° / {day['temp_max']}°",
                         font=("맑은 고딕", 9, "bold"), bg="#FFFFFF", fg="#444444").pack(side=tk.LEFT)

                tk.Label(row, text=day.get("condition", "-"), font=("맑은 고딕", 8),
                         bg="#FFFFFF", fg="#888888", anchor=tk.W,
                         wraplength=240, justify=tk.LEFT).pack(fill=tk.X, padx=(10, 0))

        threading.Thread(target=fetch, daemon=True).start()

    # ── OpenCV 버튼 핸들러 ──
    def _set_shoulder(self):
        if self.on_set_shoulder:
            self.on_set_shoulder()

    def _save_baseline(self):
        if self.on_save_baseline:
            self.on_save_baseline()

    # ── 환자 팝업 ──
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

        context_menu = tk.Menu(popup, tearoff=0)

        def show_context_menu(event):
            index = listbox.nearest(event.y)
            if index < 0 or index >= listbox.size():
                return
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            context_menu.delete(0, tk.END)
            selected_patient = self._load_patients()[index]
            context_menu.add_command(
                label="정보 수정",
                command=lambda: self._edit_patient_popup(selected_patient, listbox, index, popup)
            )
            context_menu.add_separator()
            context_menu.add_command(
                label="삭제",
                command=lambda: self._delete_patient(selected_patient, listbox, index)
            )
            context_menu.post(event.x_root, event.y_root)

        listbox.bind("<Button-3>", show_context_menu)
        ttk.Separator(popup).pack(fill=tk.X, padx=16, pady=12)

        tk.Label(popup, text="새 환자 등록", font=FONT,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, padx=16)

        name_var = tk.StringVar()
        id_var = tk.StringVar()
        save_var = tk.BooleanVar(value=True)

        name_entry = tk.Entry(popup, textvariable=name_var, font=FONT, fg="#AAAAAA")
        name_entry.insert(0, "환자 이름 입력")
        name_entry.bind("<FocusIn>", lambda e: (name_entry.delete(0, tk.END), name_entry.config(fg="#000000")) if name_var.get() == "환자 이름 입력" else None)
        name_entry.bind("<FocusOut>", lambda e: (name_entry.insert(0, "환자 이름 입력"), name_entry.config(fg="#AAAAAA")) if name_var.get() == "" else None)
        name_entry.pack(fill=tk.X, padx=16, pady=4)

        id_entry = tk.Entry(popup, textvariable=id_var, font=FONT, fg="#AAAAAA")
        id_entry.insert(0, "환자 ID 입력 (예: P004)")
        id_entry.bind("<FocusIn>", lambda e: (id_entry.delete(0, tk.END), id_entry.config(fg="#000000")) if id_var.get() == "환자 ID 입력 (예: P004)" else None)
        id_entry.bind("<FocusOut>", lambda e: (id_entry.insert(0, "환자 ID 입력 (예: P004)"), id_entry.config(fg="#AAAAAA")) if id_var.get() == "" else None)
        id_entry.pack(fill=tk.X, padx=16, pady=4)

        tk.Checkbutton(popup, text="환자 정보 저장", variable=save_var,
                       bg="#FFFFFF", font=FONT_SMALL).pack(anchor=tk.W, padx=16)

        def confirm():
            name = name_var.get().strip()
            pid = id_var.get().strip()
            if not name or not pid or name == "환자 이름 입력" or pid == "환자 ID 입력 (예: P004)":
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

    def _delete_patient(self, patient: dict, listbox: tk.Listbox, index: int):
        if not messagebox.askyesno("삭제 확인", f"{patient['name']} ({patient['id']}) 을 삭제하시겠습니까?"):
            return
        patients = self._load_patients()
        patients = [p for p in patients if p["id"] != patient["id"]]
        with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
            json.dump({"patients": patients}, f, ensure_ascii=False, indent=2)
        listbox.delete(index)

    def _edit_patient_popup(self, patient: dict, listbox: tk.Listbox, index: int, parent):
        edit_popup = tk.Toplevel(parent)
        edit_popup.title("환자 정보 수정")
        edit_popup.geometry("300x220")
        edit_popup.configure(bg="#FFFFFF")
        edit_popup.resizable(False, False)

        FONT = ("맑은 고딕", 10)
        FONT_SMALL = ("맑은 고딕", 9)

        tk.Label(edit_popup, text="환자 정보 수정", font=("맑은 고딕", 11, "bold"),
                 bg="#FFFFFF").pack(pady=(16, 12))

        name_var = tk.StringVar(value=patient["name"])
        id_var = tk.StringVar(value=patient["id"])

        tk.Label(edit_popup, text="이름", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, padx=16)
        tk.Entry(edit_popup, textvariable=name_var, font=FONT).pack(
            fill=tk.X, padx=16, pady=(2, 8))

        tk.Label(edit_popup, text="ID", font=FONT_SMALL,
                 bg="#FFFFFF", fg="#888888").pack(anchor=tk.W, padx=16)
        tk.Entry(edit_popup, textvariable=id_var, font=FONT).pack(
            fill=tk.X, padx=16, pady=(2, 8))

        def save_edit():
            new_name = name_var.get().strip()
            new_id = id_var.get().strip()
            if not new_name or not new_id:
                messagebox.showwarning("입력 오류", "이름과 ID를 입력해주세요.", parent=edit_popup)
                return
            patients = self._load_patients()
            for p in patients:
                if p["id"] == patient["id"]:
                    p["name"] = new_name
                    p["id"] = new_id
                    if "ip" in patient:
                        p["ip"] = patient["ip"]
                    break
            with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
                json.dump({"patients": patients}, f, ensure_ascii=False, indent=2)
            listbox.delete(index)
            listbox.insert(index, f"{new_name}  ({new_id})")
            edit_popup.destroy()

        btn_frame = tk.Frame(edit_popup, bg="#FFFFFF")
        btn_frame.pack(fill=tk.X, padx=16, pady=8)
        tk.Button(btn_frame, text="취소", font=FONT, relief=tk.FLAT,
                  bg="#F0F0F0", command=edit_popup.destroy).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Button(btn_frame, text="저장", font=FONT, relief=tk.FLAT,
                  bg="#F0F0F0", command=save_edit).pack(
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

        name_entry = tk.Entry(popup, textvariable=name_var, font=FONT, fg="#AAAAAA")
        name_entry.insert(0, "환자 이름 입력")
        name_entry.bind("<FocusIn>", lambda e: (name_entry.delete(0, tk.END), name_entry.config(fg="#000000")) if name_var.get() == "환자 이름 입력" else None)
        name_entry.bind("<FocusOut>", lambda e: (name_entry.insert(0, "환자 이름 입력"), name_entry.config(fg="#AAAAAA")) if name_var.get() == "" else None)
        name_entry.pack(fill=tk.X, padx=16, pady=4)

        id_entry = tk.Entry(popup, textvariable=id_var, font=FONT, fg="#AAAAAA")
        id_entry.insert(0, "환자 ID 입력 (예: P004)")
        id_entry.bind("<FocusIn>", lambda e: (id_entry.delete(0, tk.END), id_entry.config(fg="#000000")) if id_var.get() == "환자 ID 입력 (예: P004)" else None)
        id_entry.bind("<FocusOut>", lambda e: (id_entry.insert(0, "환자 ID 입력 (예: P004)"), id_entry.config(fg="#AAAAAA")) if id_var.get() == "" else None)
        id_entry.pack(fill=tk.X, padx=16, pady=4)

        tk.Checkbutton(popup, text="환자 정보 저장 (다음 접속 시 자동 인식)",
                       variable=save_var, bg="#FFFFFF", font=FONT_SMALL).pack(anchor=tk.W, padx=16)

        def confirm():
            name = name_var.get().strip()
            pid = id_var.get().strip()
            if not name or not pid or name == "환자 이름 입력" or pid == "환자 ID 입력 (예: P004)":
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

    def _toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            self.camera_btn.config(text="카메라 끄기", bg="#FFE0E0")
        else:
            self.camera_btn.config(text="카메라 켜기", bg="#F0F0F0")
            self.doctor_canvas.delete("all")
            self.doctor_canvas.create_text(213, 160, text="카메라 꺼짐",
                                           fill="#AAAAAA", font=("맑은 고딕", 10))
        if self.on_camera_toggle:
            self.on_camera_toggle(self.camera_active)

    def _toggle_doctor_mic(self):
        self.doctor_audio_active = not self.doctor_audio_active
        if self.doctor_audio_active:
            self.mic_btn.config(text="마이크 끄기", bg="#FFE0E0")
        else:
            self.mic_btn.config(text="마이크 켜기", bg="#F0F0F0")
        if self.on_doctor_audio_toggle:
            self.on_doctor_audio_toggle(self.doctor_audio_active)

    def _on_enter_key(self, event):
        if not (event.state & 0x1):
            self._send_message()
            return "break"

    def _send_message(self):
        msg = self.msg_entry.get("1.0", tk.END).strip()
        if not msg or not self.on_send_message:
            return
        if msg.startswith("/s "):
            self.sentence_counter += 1
            text = msg[3:].strip()
            self.on_send_message(f"SENTENCE:{self.sentence_counter}:{text}")
        else:
            self.on_send_message(f"INSTRUCT:{self.sentence_counter}:{msg}")
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
        save_path = os.path.abspath("data/sessions")
        messagebox.showinfo("세션 저장 완료", f"저장 위치:\n{save_path}")

    def update_patient_frame(self, frame: np.ndarray):
        if not self.patient_camera_active:
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.patient_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.patient_canvas.image = photo

    def _clear_patient_frame(self):
        self.patient_camera_active = False
        self.patient_canvas.delete("all")
        self.patient_canvas.create_text(320, 240, text="환자 카메라 꺼짐",
                                        fill="#AAAAAA", font=("맑은 고딕", 10))

    def _on_patient_camera_on(self):
        self.patient_camera_active = True

    def update_doctor_frame(self, frame: np.ndarray):
        if not self.camera_active:
            return
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