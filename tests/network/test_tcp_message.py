# tests/network/test_tcp_message.py
# TCP 문장 송수신 및 분석 결과 송수신 테스트

import socket
import threading
import time
from src.common.config import SERVER_IP, TCP_PORT

def test_send_message():
    # 의료진 → 환자 재활 문장 송신 테스트
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_IP, TCP_PORT))
        print(f"TCP 연결 성공 - {SERVER_IP}:{TCP_PORT}")

        test_msg = "아-에-이-오-우 를 천천히 반복하세요"
        sock.sendall(test_msg.encode())
        print(f"문장 송신 완료: {test_msg}")
    except Exception as e:
        print(f"문장 송신 실패: {e}")
    finally:
        sock.close()

def test_send_result():
    # 환자 → 의료진 분석 결과 송신 테스트
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_IP, TCP_PORT))
        print(f"TCP 연결 성공 - {SERVER_IP}:{TCP_PORT}")

        test_result = "비대칭 지수: 0.12 / 발음 정확도: 87%"
        sock.sendall(test_result.encode())
        print(f"분석 결과 송신 완료: {test_result}")
    except Exception as e:
        print(f"분석 결과 송신 실패: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    threading.Thread(target=test_send_message, daemon=True).start()
    time.sleep(1)
    threading.Thread(target=test_send_result,  daemon=True).start()
    time.sleep(3)
    print("테스트 종료")