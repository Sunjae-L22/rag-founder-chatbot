"""Notebook-derived RAG implementation. Deployment adaptation: 2026-09-13."""
import os
import threading
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 180
TOP_K = 5
SIM_THRESHOLD = 0.53
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
kiwi = None
retriever = None
_init_lock = threading.Lock()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_sents=1):
    """문장 경계를 지키면서 chunk_size(글자) 이하로 분할. 이웃 청크와 overlap_sents 문장 중첩.
    문장부호가 없는 초장문(표를 펴낸 크롤링 텍스트 등)은 chunk_size 단위로 강제 분할."""
    sents = []
    for s in kiwi.split_into_sents(text):
        t = s.text.strip()
        while len(t) > chunk_size:
            sents.append(t[:chunk_size])
            t = t[chunk_size - 30:]
        if t:
            sents.append(t)
    chunks, cur, cur_len = ([], [], 0)
    for s in sents:
        if cur and cur_len + len(s) > chunk_size:
            chunks.append(' '.join(cur))
            keep = cur[-overlap_sents:] if overlap_sents else []
            if sum((len(x) for x in keep)) + len(s) > chunk_size:
                keep = []
            cur, cur_len = (keep, sum((len(x) for x in keep)))
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append(' '.join(cur))
    return chunks

def make_chunks(df, chunk_size=CHUNK_SIZE):
    """데이터프레임 전체를 청크 단위 데이터프레임으로 변환"""
    rows = []
    for _, r in df.iterrows():
        for j, ch in enumerate(chunk_text(r['content'], chunk_size)):
            rows.append({'doc_id': r['doc_id'], 'category': r['category'], 'title': r['title'], 'source': r['source'], 'chunk_no': j, 'chunk': ch})
    return pd.DataFrame(rows)

class Retriever:
    """FAISS 기반 의미 검색 리트리버"""

    def __init__(self, chunk_df, embedder):
        self.chunk_df = chunk_df.reset_index(drop=True)
        self.embedder = embedder
        texts = (self.chunk_df['title'] + ': ' + self.chunk_df['chunk']).tolist()
        cache_dir = Path(__file__).resolve().parents[1] / '.cache'
        cache_dir.mkdir(exist_ok=True)
        digest = hashlib.sha256(('\n'.join(texts) + EMBEDDING_MODEL).encode()).hexdigest()
        cache_path = cache_dir / f'embeddings-{digest}.npy'
        if cache_path.exists():
            emb = np.load(cache_path, allow_pickle=False)
        else:
            emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            np.save(cache_path, emb, allow_pickle=False)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype='float32'))

    def search(self, query, top_k=3):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.asarray(q_emb, dtype='float32'), top_k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            row = self.chunk_df.iloc[int(i)]
            results.append({'doc_id': int(row['doc_id']), 'category': row['category'], 'title': row['title'], 'source': row['source'], 'chunk': row['chunk'], 'score': float(s)})
        return results

SYSTEM_PROMPT = '당신은 창업하는 사장님을 돕는 법률·세무 안내 챗봇 \'사장님, 전설이 되다\'입니다.\n\n[답변 규칙]\n1. 반드시 아래 [참고 자료]의 내용만 근거로 답변합니다.\n2. [참고 자료]에 없는 내용은 절대 추측하거나 지어내지 않습니다. 자료에 없으면\n   "제가 가진 자료에서는 확인할 수 없는 내용이에요."라고 답합니다.\n3. 금액·기간·비율·업종코드 등 숫자는 참고 자료에 적힌 그대로 정확하게 답합니다.\n4. 한국어로 2~5문장 이내, 처음 창업하는 사장님도 이해할 수 있게 쉽고 친절하게 답합니다.\n5. 답변 끝에 근거 문서를 (출처: 문서 제목) 형식으로 표기합니다. "자료 1" 같은 번호가 아니라\n   실제 문서 제목(예: 출처: 등록 의무)을 적습니다.'

SYSTEM_PROMPT_NO_GUARDRAIL = '당신은 창업과 사업에 대해 답변하는 친절한 챗봇입니다. 질문에 자유롭게 답하세요.'

INTRO_MESSAGE = '안녕하세요 사장님! 저는 창업 법률·세무 안내 챗봇 **사장님, 전설이 되다**예요. 🏆\n사업자등록, 부가가치세·종합소득세, 통신판매업 신고, 창업 세액감면, 정부 지원사업까지 —\n국세청·중소벤처기업부 공식 자료를 근거로만 답해 드려요. 무엇이 궁금하세요?'

REFUSAL_MESSAGE = '죄송해요 사장님, 제가 가진 공식 자료(국세청·중소벤처기업부)에서는 관련 내용을 찾지 못했어요. 🙏\n국세상담센터(국번없이 126), 중소기업 통합콜센터(1357), 정부24에 문의하시면 정확한 안내를 받으실 수 있어요.'

DISCLAIMER = '\n\n⚠️ 세법·지원사업 요건은 수시로 개정됩니다. 실제 신고·신청 전에 국세청 홈택스, 정부24, K-Startup에서 최신 기준을 꼭 확인하세요. 본 답변은 일반 정보 안내이며 세무·법률 자문이 아닙니다.'

NO_CONTEXT_PROMPT = "당신은 창업 안내 챗봇 '사장님, 전설이 되다'입니다.\n지금 사용자의 입력과 관련된 참고 자료를 찾지 못한 상태입니다. 다음 규칙으로 짧게(1~3문장) 답하세요.\n1. 인사·감정 표현·잡담(예: 배고파, 힘들다, 떨려)이면 따뜻하게 공감해 주고, 당신이 도울 수 있는 주제\n   (사업자등록·세금·통신판매업 신고·창업 세액감면·정부 지원사업)를 가볍게 알려 주세요.\n2. 정보를 묻는 질문이면 가진 자료에 없어 정확히 답할 수 없다고 솔직히 말하고,\n   국세상담센터(국번없이 126), 중소기업 통합콜센터(1357), 정부24를 안내하세요.\n3. 어떤 경우에도 법령·세금·지원사업의 구체적 수치·절차를 기억에 의존해 답하지 마세요. 지어내면 안 됩니다.\n4. 한국어로 친근하게, 사용자를 '사장님'이라고 부르세요."

SMALLTALK_EXACT = {'하이', 'ㅎㅇ', '헬로', '안녕'}

SMALLTALK_PATTERNS = ['안녕하', '반가워', '반갑습니다', '고마워', '감사합니다', '감사해요', '누구야', '누구니', '누구세요', '뭐 할 수', '뭘 할 수', '자기소개']

def is_smalltalk(q):
    """짧은 인사/잡담 감지. '안녕'이 포함된 일반 질문을 오분류하지 않도록
    ①인사 단어 단독 ②짧은 문장(15자 이하)+인사 패턴 두 경우만 인정"""
    qs = q.strip().rstrip('!?.~^ ')
    return qs in SMALLTALK_EXACT or (len(qs) <= 15 and any((p in qs for p in SMALLTALK_PATTERNS)))

EMOTION_PATTERNS = ['배고파', '배고프', '힘들', '피곤', '졸려', '졸리', '심심', '지쳤', '지친다', '떨려', '긴장']

EMOTION_MESSAGE = '사장님, 창업 준비하시느라 고생이 많으세요! 잠깐 쉬어 가도 괜찮아요. ☕\n기운 차리시면 사업자등록·세금·지원사업처럼 제가 도울 수 있는 것들, 언제든 물어봐 주세요!'

def is_emotion(q):
    qs = q.strip()
    return len(qs) <= 15 and any((p in qs for p in EMOTION_PATTERNS))

def call_llm(system_prompt, user_prompt, history_msgs=None, temperature=0.2, max_tokens=500):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0, max_retries=0)
    messages = [{'role': 'system', 'content': system_prompt}]
    if history_msgs:
        messages += history_msgs
    messages.append({'role': 'user', 'content': user_prompt})
    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()

def msg_text(content):
    """Gradio 버전마다 메시지 content가 str / list / dict로 달라짐(예: Gradio 6은
    [{'text':..., 'type':'text'}] 리스트). 어떤 형식이든 안전하게 순수 텍스트만 추출"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join((msg_text(c) for c in content)).strip()
    if isinstance(content, dict):
        return content.get('text') or content.get('value') or ''
    return str(content) if content is not None else ''

CONDENSE_SYSTEM = "당신은 대화형 검색 도우미입니다. 아래 이전 대화를 참고하여, 사용자의 '현재 질문'을 앞 맥락 없이도 그 자체로 이해되는 독립적인 검색 질문 한 문장으로 바꾸세요. 현재 질문이 이미 독립적이면 그대로 두세요. 다른 설명 없이 질문 문장만 출력하세요."

def condense_query(question, history):
    """후속 질문을 '이전 맥락이 반영된 독립형 검색 질문'으로 변환 (멀티턴의 핵심).
    - LLM 모드: 대화를 보고 질의 재작성(query rewriting) — 후속/주제전환 모두 똑똑하게 처리
    - 키 없음: 직전 사용자 질문 1개를 붙여 맥락 보강 (간단 fallback)"""
    prev_users = [msg_text(m.get('content')) for m in history if m.get('role') == 'user']
    prev_users = [u for u in prev_users if u]
    if not prev_users:
        return question
    if OPENAI_API_KEY:
        convo = '\n'.join((f'- {u}' for u in prev_users[-3:])) + f'\n- (현재) {question}'
        try:
            return call_llm(CONDENSE_SYSTEM, f'[이전 대화]\n{convo}\n\n[독립형 검색 질문]', temperature=0.0, max_tokens=80)
        except Exception:
            pass
    return prev_users[-1] + ' ' + question

def build_history_messages(history, max_turns=3):
    """생성 LLM에 넘길 최근 대화. assistant 답변에 붙은 출처(📚)/고지문(⚠️)은 떼어 토큰을 아낌"""
    msgs = []
    for m in history[-max_turns * 2:]:
        content = msg_text(m.get('content'))
        if m.get('role') == 'assistant':
            content = content.split('📚')[0].split('⚠️')[0].strip()
        if content:
            msgs.append({'role': m['role'], 'content': content})
    return msgs

def extractive_answer(hits):
    """API 키가 없을 때: 검색된 공식 자료 원문을 정리해 안내"""
    lines = ['사장님 질문과 관련된 공식 자료를 찾았어요. 📋\n']
    for i, h in enumerate(hits, 1):
        lines.append(f"**{i}. {h['title']}** ({h['category']})\n{h['chunk']}\n")
    return '\n'.join(lines)

def rag_answer(question, history=None, top_k=None, threshold=None, use_guardrail=True):
    """RAG 파이프라인 본체: (멀티턴)질의 재작성 → 검색 → 임계값 검사 → 프롬프트(+대화이력) → 생성 → 고지"""
    history = history or []
    top_k = TOP_K if top_k is None else top_k
    threshold = SIM_THRESHOLD if threshold is None else threshold
    question = question.strip()
    if not question:
        return ('질문을 입력해 주세요.', [])
    if is_smalltalk(question):
        return (INTRO_MESSAGE, [])
    if is_emotion(question):
        return (EMOTION_MESSAGE, [])
    search_query = condense_query(question, history)
    hits = retriever.search(search_query, top_k=top_k)
    hits = [h for h in hits if h['score'] >= threshold]
    if not hits:
        if OPENAI_API_KEY:
            try:
                return (call_llm(NO_CONTEXT_PROMPT, question, history_msgs=build_history_messages(history)), [])
            except Exception:
                return (REFUSAL_MESSAGE, [])
        return (REFUSAL_MESSAGE, [])
    if not OPENAI_API_KEY:
        return (extractive_answer(hits) + DISCLAIMER, hits)
    context = '\n\n'.join((f"[자료 {i}] 제목: {h['title']} (분류: {h['category']})\n{h['chunk']}" for i, h in enumerate(hits, 1)))
    system = SYSTEM_PROMPT if use_guardrail else SYSTEM_PROMPT_NO_GUARDRAIL
    user = f'[참고 자료]\n{context}\n\n[질문]\n{question}'
    try:
        return (call_llm(system, user, history_msgs=build_history_messages(history)) + DISCLAIMER, hits)
    except Exception as e:
        return ('⚠️ 답변 생성에 연결하지 못해 검색 결과를 안내합니다.\n\n' + extractive_answer(hits) + DISCLAIMER, hits)

demo_hist = []

def initialize():
    global kiwi, retriever
    with _init_lock:
        if retriever is None:
            kiwi = Kiwi()
            data = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
            df = pd.read_csv(data)
            embedder = SentenceTransformer(EMBEDDING_MODEL)
            retriever = Retriever(make_chunks(df, CHUNK_SIZE), embedder)
    return retriever
