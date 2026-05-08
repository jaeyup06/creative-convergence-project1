# 커밋 가이드

## 분기 구조

| 분기 | 용도 |
|------|------|
| `main` | 완성된 코드만 병합하는 공식 분기 |
| `server_demo` | 서버 개발 분기 |
| `dlib_test` | dlib 관련 실험 분기 |

> 새 기능 개발 시 `main`에서 직접 작업하지 말고 분기를 따로 만들어 작업하세요.

---

## 커밋 메시지 규칙

| 태그 | 의미 | 예시 |
|------|------|------|
| `feat` | 새 기능 추가 | `feat: UDP 영상 수신 구현` |
| `fix` | 버그 수정 | `fix: 포트 번호 오류 수정` |
| `chore` | 설정, 환경 관련 | `chore: .gitignore 추가` |
| `docs` | 문서 수정 | `docs: README 업데이트` |
| `refactor` | 기능 변경 없이 코드 정리 | `refactor: server.py 함수 분리` |
| `test` | 테스트 코드 추가 | `test: UDP 수신 테스트 추가` |
| `style` | 포맷, 띄어쓰기 등 | `style: 들여쓰기 정리` |

---

## 폴더 구조

```
CREATIVE-CONVERGENCE-PROJECT1/
├── src/
│   ├── server/         # 서버 메인 코드
│   ├── client/         # 클라이언트 메인 코드
│   ├── recognition/    # 안면 비대칭 · 음성 분석 모듈
│   └── common/         # 공통 상수 (포트번호, 패킷 구조)
├── data/               # Git 제외 (모델, 세션, 캐시 등)
├── scripts/            # 초기 세팅 스크립트
├── tests/              # 테스트 코드
└── 개발일지/           # 팀원 개인 실험 공간
```

---

## 초기 세팅

```bash
pip install -r requirements.txt
python scripts/download_models.py
```

---

## 주의사항

- 포트번호, IP 등 공통 값은 반드시 `src/common/config.py` 에서만 수정

---

## 대용량 파일 공유 방법

- `.dat`, 이미지 등 대용량 파일은 Git에 올리지 않고 카카오톡/구글 드라이브로 별도 공유
- 모델 파일은 `python scripts/download_models.py` 로 자동 다운로드 가능
- 실수로 올라간 경우 바로 알려주세요