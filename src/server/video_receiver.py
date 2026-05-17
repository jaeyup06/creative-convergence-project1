import cv2
import socket
import struct
import threading
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.config import UDP_VIDEO_PORT


class VideoReceiver:
    def __init__(self, frame_callback=None):
        """
        frame_callback: 프레임 수신 시 호출할 함수 (GUI 연동용)
                        None이면 자체 cv2.imshow로 출력
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', UDP_VIDEO_PORT))
        self.sock.settimeout(2.0)  # 2초 응답 없으면 루프 재시도

        self.running = False
        self.frame_callback = frame_callback

        # 분할 패킷 조립용 버퍼
        # key: total_chunks, value: {index: chunk_data}
        self.buffer = {}

    def start(self):
        """수신 스레드 시작"""
        self.running = True
        print(f"[VideoReceiver] 수신 대기 중... PORT {UDP_VIDEO_PORT}")

        thread = threading.Thread(target=self._recv_loop, daemon=True)
        thread.start()

    def _recv_loop(self):
        """UDP 패킷을 수신하고 프레임을 복원하는 루프"""
        # 분할 패킷 조립 버퍼: (total, seq_id) 기반으로 관리
        chunk_buffer = {}  # {total: {index: bytes}}

        while self.running:
            try:
                # 최대 MAX_PACKET_SIZE(60000) + 헤더(4) 크기로 수신
                packet, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue  # 타임아웃은 정상, 루프 재시도
            except OSError:
                break

            if len(packet) < 4:
                continue  # 헤더보다 작으면 무시

            # 헤더 파싱: total_chunks(2bytes) + chunk_index(2bytes)
            total, index = struct.unpack('>HH', packet[:4])
            data = packet[4:]

            if total == 1:
                # 단일 패킷: 바로 프레임 복원
                self._decode_and_emit(data)
            else:
                # 분할 패킷: 버퍼에 쌓고 다 모이면 조립
                if total not in chunk_buffer:
                    chunk_buffer[total] = {}
                chunk_buffer[total][index] = data

                if len(chunk_buffer[total]) == total:
                    # 모든 조각 수신 완료 → 순서대로 합치기
                    full_data = b''.join(
                        chunk_buffer[total][i] for i in range(total)
                    )
                    chunk_buffer.pop(total)
                    self._decode_and_emit(full_data)

    def _decode_and_emit(self, data: bytes):
        """JPEG 바이트를 numpy 배열(프레임)로 변환 후 콜백 또는 출력"""
        np_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return  # 손상된 패킷 무시

        if self.frame_callback:
            # GUI 연동 시 콜백으로 프레임 전달
            self.frame_callback(frame)
        else:
            # 단독 실행 시 직접 출력
            cv2.imshow('Patient Video (Server)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

    def stop(self):
        """수신 중지 및 자원 해제"""
        self.running = False
        self.sock.close()
        cv2.destroyAllWindows()
        print("[VideoReceiver] 수신 중지")


# 단독 실행 테스트용
if __name__ == '__main__':
    receiver = VideoReceiver()
    receiver.start()

    print("수신 중... 영상 창에서 'q'를 누르면 종료됩니다.")
    try:
        while receiver.running:
            pass
    except KeyboardInterrupt:
        pass

    receiver.stop()
