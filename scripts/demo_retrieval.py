# -*- coding: utf-8 -*-
"""챗봇 데모 미리보기 (포트 7861). 최초 1회 임베딩 후 /tmp 캐시 저장."""
import os, pickle, re
from pathlib import Path
import faiss, numpy as np, pandas as pd
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
import gradio as gr

CACHE = str(Path(__file__).resolve().parent.parent / ".cache" / "demo.pkl")
Path(CACHE).parent.mkdir(exist_ok=True)
DATA = str(Path(__file__).resolve().parent.parent / "data" / "dataset.csv")
TOP_K, SIM_THRESHOLD, OPENAI_API_KEY = 5, 0.53, os.environ.get("OPENAI_API_KEY", "")
kiwi = Kiwi()
df = pd.read_csv(DATA)

def chunk_text(text, chunk_size=180, overlap_sents=1):
    sents = []
    for s in kiwi.split_into_sents(text):
        t = s.text.strip()
        while len(t) > chunk_size:
            sents.append(t[:chunk_size]); t = t[chunk_size - 30:]
        if t: sents.append(t)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) > chunk_size:
            chunks.append(" ".join(cur))
            keep = cur[-overlap_sents:] if overlap_sents else []
            if sum(len(x) for x in keep) + len(s) > chunk_size: keep = []
            cur, cur_len = keep, sum(len(x) for x in keep)
        cur.append(s); cur_len += len(s)
    if cur: chunks.append(" ".join(cur))
    return chunks

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
if os.path.exists(CACHE):
    cdf, emb = pickle.load(open(CACHE, "rb"))
else:
    rows = []
    for _, r in df.iterrows():
        for ch in chunk_text(r["content"]):
            rows.append({"doc_id": r["doc_id"], "title": r["title"], "chunk": ch})
    cdf = pd.DataFrame(rows)
    emb = embedder.encode((cdf["title"] + ": " + cdf["chunk"]).tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    pickle.dump((cdf, emb), open(CACHE, "wb"))
cdf = cdf.merge(df[["doc_id", "category", "source"]], on="doc_id", how="left")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.asarray(emb, dtype="float32"))

def search(query, top_k=5):
    q = embedder.encode([query], normalize_embeddings=True)
    s, i = index.search(np.asarray(q, dtype="float32"), top_k)
    return [{"doc_id": int(cdf.iloc[int(ix)]["doc_id"]), "category": cdf.iloc[int(ix)]["category"],
             "title": cdf.iloc[int(ix)]["title"], "source": cdf.iloc[int(ix)]["source"],
             "chunk": cdf.iloc[int(ix)]["chunk"], "score": float(sc)} for sc, ix in zip(s[0], i[0]) if ix != -1]

INTRO = ("안녕하세요 사장님! 저는 창업 법률·세무 안내 챗봇 **사장님, 전설이 되다**예요. 🏆\n"
         "사업자등록·세금·통신판매업 신고·창업 세액감면·정부 지원사업까지 — 공식 자료를 근거로만 답해 드려요.")
REFUSAL = ("죄송해요 사장님, 제가 가진 공식 자료(국세청·중소벤처기업부)에서는 관련 내용을 찾지 못했어요. 🙏\n"
           "국세상담센터(126), 중소기업 통합콜센터(1357), 정부24에 문의하시면 정확한 안내를 받으실 수 있어요.")
EMOTION = ("사장님, 창업 준비하시느라 고생이 많으세요! 잠깐 쉬어 가도 괜찮아요. ☕\n"
           "기운 차리시면 사업자등록·세금·지원사업처럼 제가 도울 수 있는 것들, 언제든 물어봐 주세요!")
DISC = "\n\n⚠️ 세법·지원사업 요건은 수시로 개정됩니다. 신고·신청 전 국세청 홈택스·정부24·K-Startup에서 최신 기준을 꼭 확인하세요. 본 답변은 일반 정보 안내이며 세무·법률 자문이 아닙니다."

def msg_text(c):
    if isinstance(c, str): return c
    if isinstance(c, list): return " ".join(msg_text(x) for x in c).strip()
    if isinstance(c, dict): return c.get("text") or c.get("value") or ""
    return str(c) if c is not None else ""

def condense(question, history):
    prev = [msg_text(m.get("content")) for m in history if m.get("role") == "user"]
    prev = [p for p in prev if p]
    if not prev: return question
    return prev[-1] + " " + question

def rag_answer(question, history=None, threshold=SIM_THRESHOLD):
    history = history or []
    q = question.strip()
    if not q: return "질문을 입력해 주세요.", []
    qs = q.rstrip("!?.~^ ")
    if qs in {"안녕","하이","헬로"} or (len(qs) <= 15 and any(p in qs for p in ["안녕하","누구야","누구니"])): return INTRO, []
    if len(qs) <= 15 and any(p in qs for p in ["배고파","떨려","힘들","피곤","졸려","긴장"]): return EMOTION, []
    hits = [h for h in search(condense(q, history), TOP_K) if h["score"] >= threshold]
    if not hits: return REFUSAL, []
    lines = ["사장님 질문과 관련된 공식 자료를 찾았어요. 📋\n"]
    for i, h in enumerate(hits[:3], 1):
        lines.append(f"**{i}. {h['title']}** ({h['category']})\n{h['chunk']}\n")
    return "\n".join(lines) + DISC, hits

def format_sources(hits):
    seen, out = set(), []
    for h in hits:
        if h["doc_id"] in seen: continue
        seen.add(h["doc_id"])
        out.append(f"- [{h['category']}] **{h['title']}** — {h['source']} (유사도 {h['score']:.2f})")
    return "\n".join(out)

def respond(message, chat_history, threshold):
    if not message.strip(): return "", chat_history
    answer, hits = rag_answer(message, history=chat_history, threshold=float(threshold))
    if hits: answer += "\n\n---\n📚 **참고한 공식 자료**\n" + format_sources(hits)
    return "", chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]

THEME = gr.themes.Soft(primary_hue="amber", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "system-ui", "sans-serif"])
CSS = """
.gradio-container {max-width: 960px !important; margin: 0 auto !important;}
#hero {background: linear-gradient(135deg, #1c2844 0%, #2b3d68 60%, #8a6d1f 135%);
       border-radius: 18px; padding: 24px 30px; color: #f4f6fb; margin-bottom: 6px;}
#hero h1 {color: #ffd24d; margin: 0 0 6px; font-size: 1.7em;}
#hero p {color: #e9edf7; margin: 0; font-size: .92em;}
#chatbox {border-radius: 16px;}
.message.user, .user-row .message {background: rgba(240,178,41,.16) !important; border: 1px solid rgba(240,178,41,.38) !important;}
.message.bot, .bot-row .message {background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important;}
#chatbox .message .message {background: transparent !important; border: none !important; box-shadow: none !important;}
#send-btn {background: linear-gradient(135deg, #f3b62b, #dc9117) !important; color: #2b2105 !important; font-weight: 700; border: none !important; border-radius: 12px !important;}
footer {visibility: hidden;}
"""
def make_component(cls, **kw):
    while True:
        try: return cls(**kw)
        except TypeError as e:
            m = re.search(r"'(\w+)'", str(e))
            if m and m.group(1) in kw: kw.pop(m.group(1))
            else: raise

mode = "검색 결과 안내 모드 (이 미리보기는 LLM을 호출하지 않음)"
with gr.Blocks(title="사장님, 전설이 되다", theme=THEME, css=CSS) as demo:
    gr.HTML(f'<div id="hero"><h1>🏆 사장님, 전설이 되다</h1>'
            f'<p>창업 법률·세무 RAG 챗봇 · 공식 자료 726건 기반 · 대화 맥락 기억 &nbsp;|&nbsp; <b>{mode}</b></p></div>')
    chatbot = make_component(gr.Chatbot, type="messages", height=460, label="상담 내용", elem_id="chatbox", show_copy_button=True)
    with gr.Row():
        msg = gr.Textbox(placeholder="예: 사업자등록은 언제까지 해야 하나요?", scale=5, container=False, autofocus=True, elem_id="msg-box")
        send_btn = gr.Button("질문하기 📨", variant="primary", scale=1, elem_id="send-btn")
    gr.Examples(examples=["사업자등록은 언제까지 해야 하나요?", "네일샵 차리려면 위생교육 받아야 해?",
                          "나라에서 창업자한테 지원해주는 거 있어?", "미용실 창업하려고 하는데 뭐부터 준비해?",
                          "비트코인 지금 사도 돼?"], inputs=msg, label="예시 질문")
    with gr.Accordion("🔧 검색 설정 (할루시네이션 라이브 데모)", open=False):
        th = gr.Slider(0.0, 0.8, value=SIM_THRESHOLD, step=0.01, label="SIM_THRESHOLD — 0으로 내리면 무관한 질문에도 답변 시도")
    msg.submit(respond, [msg, chatbot, th], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot, th], [msg, chatbot])

try:
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, theme=THEME, css=CSS)
except TypeError:
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
