# 배포 구성 · 2026-09-13 보완

이 구성은 과거 Netlify 배포 파일의 복원이 아닙니다. 작성자가 당시 사이트를 삭제했다고 확인한 뒤, 보관된 Python 코드를 바탕으로 새로 구성했습니다.

## 역할

- `web/`: HTML/CSS/JS 대화 화면. 최근 3턴을 API로 전달하고 답변·출처·모드를 표시합니다.
- `netlify.toml`, `scripts/build-netlify.mjs`: 정적 사이트 빌드와 `/api/*` HTTPS 프록시.
- `backend/core.py`: 노트북의 실제 청킹·검색·멀티턴·답변 함수를 분리. Kiwi, 기존 ko-sroberta 모델, FAISS, 180자/top-k 5/0.53 유지.
- `backend/app.py`: FastAPI HTTP 계층, 입력 길이·이력 제한, 동시 실행 제한, 오류 처리.
- `Dockerfile`: Python 서버 실행 이미지. 당시 환경이 아니라 보완용 환경입니다.

## 로컬 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8787
```

`http://127.0.0.1:8787`에서 화면과 API를 함께 사용합니다. 최초 시작 시 임베딩 모델 다운로드와 전체 문서 인덱싱이 필요하므로 준비까지 시간이 걸립니다. 외부 공개 서버는 아닙니다.

API 키가 없으면 **검색 자료 안내 모드**입니다. 이는 원본 노트북의 키 없는 분기를 따르며, 생성형 LLM 답변을 실행했다고 표시하지 않습니다.

## 공개 배포 순서

1. Python 실행이 가능한 호스팅에서 이 저장소의 Dockerfile을 배포합니다. 사용 플랫폼과 비용은 별도로 확정해야 합니다. 이 문서가 유료 서비스 신청을 의미하지 않습니다.
2. 포트는 `PORT` 환경변수, 기본 7860. `/api/health`가 ready=true가 될 때까지 초기 인덱싱을 기다립니다.
3. Netlify에서 GitHub 저장소 `Sunjae-L22/rag-founder-chatbot`을 연결합니다.
4. Netlify 빌드 환경변수 `RAG_BACKEND_URL`에 **실제 Python 서버의 HTTPS 원점**을 설정합니다(경로·비밀번호·쿼리 없음).
5. 빌드 명령은 `node scripts/build-netlify.mjs`, 게시 폴더는 `dist`. 설정 파일이 자동 지정합니다.
6. 배포 후 상태 확인, 단일 질문, 후속 질문, 자료 밖 질문을 실제 공개 URL에서 검사합니다.

Python 서버 URL이 없으면 빌드는 의도적으로 실패합니다. 작동하지 않는 예시 URL을 실제 운영 주소로 배포하지 않습니다. 프록시는 별도 CORS 설정 없이 같은 출처의 `/api` 경로를 사용합니다.

## LLM 생성 모드

선택적으로 Python 서버에만 `OPENAI_API_KEY`를 설정합니다. 프런트엔드나 Netlify 빌드 산출물에 API 키를 넣지 않습니다. API 사용료가 생길 수 있습니다. 공개 서버의 무제한 사용을 피하도록 `DEMO_ACCESS_CODE`도 설정해야 하며, 두 값 중 코드가 빠지면 생성 모드의 요청은 거절합니다. 사용자는 웹 화면의 ‘체험 코드가 있나요?’에서 코드를 입력합니다. 공개 제품용 사용자 인증·과금 관리 시스템을 구현한 것은 아닙니다.

## 확인 범위

새 API의 입력 제한·이력 전달·출처 중복 제거·접근 코드·오류 응답·동시 처리 제한은 자동 검사 대상입니다. 기존 노트북의 저장 성능 수치는 별도 자료이며, 새 배포의 정확도 결과로 재표기하지 않습니다. Docker 실행과 실제 Netlify-Python 서버 간 연결은 호스팅이 정해진 뒤 검증해야 합니다.

[배포 구조도](deployment-architecture.html) · [기존 RAG 논리 구조](architecture.html)

## 이번 로컬 검증 결과

2026-09-13, macOS에서 실제 ko-sroberta 모델을 로드하고 문서 726건을 검색했습니다. 미용실 창업 질문 후 ‘시설 기준은 어떻게 돼?’에서 미용실 시설·설비 문서가 검색되었고, ‘비트코인 지금 사도 돼?’는 근거 없이 답하지 않고 자료 없음으로 안내했습니다. 브라우저에서도 첫 질문과 후속 질문, 출처 표시를 확인했습니다. API 키 없는 검색 모드이며 유료 LLM을 호출하지 않았습니다. [실행 기록](local-smoke-results.json)

자동 검사 9개 통과. Netlify 빌드 성공 및 잘못된 서버 설정 4종 거부 확인. Docker 이미지는 이 환경에 Docker가 없어 빌드 실행하지 않았습니다. 현재 공개 호스팅은 하지 않았으며, 사용자 요청으로 비용 없이 로컬 데모·배포 파일까지만 완성했습니다.
