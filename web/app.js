const $ = (id) => document.getElementById(id);
let history = [], busy = false, ready = false;
function message(text, role, sources = []) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const speaker = document.createElement('span'); speaker.className = 'speaker';
  speaker.textContent = role === 'user' ? '나의 질문' : role === 'error' ? '연결 안내' : '안내 데스크';
  const p = document.createElement('p'); p.textContent = text.replace(/\*\*(.*?)\*\*/g, '$1'); article.append(speaker, p);
  if (sources.length) {
    const details = document.createElement('details'); details.className = 'sources';
    const title = document.createElement('summary'); title.textContent = `참고한 공식 자료 ${sources.length}건`;
    const ul = document.createElement('ul');
    for (const item of sources) { const li = document.createElement('li'); li.textContent = `${item.title} · 유사도 ${Number(item.score).toFixed(2)}\n${item.source}`; ul.append(li); }
    details.append(title, ul); article.append(details);
  }
  $('messages').append(article); $('messages').scrollTop = $('messages').scrollHeight;
  return article;
}
async function health() {
  try {
    const response = await fetch('/api/health', {signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error();
    const info = await response.json(); ready = info.ready;
    $('status').textContent = ready ? '체험 가능' : '서버 준비 중';
    $('mode').textContent = info.mode === 'generation' ? '공식 자료에 근거한 AI 답변' : '공식 자료 검색 모드 · 생성형 답변 미사용';
  } catch { ready = false; $('status').textContent = '다시 연결'; $('mode').textContent = '서버가 준비되면 다시 연결해 주세요'; }
  $('send').disabled = !ready || busy;
}
$('chat-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const question = $('question').value.trim(); if (!question || busy || !ready) return;
  busy = true; $('send').disabled = true; $('clear').disabled = true;
  message(question, 'user'); $('question').value = '';
  const pending = message('관련 자료를 찾고 있어요…', 'assistant');
  try {
    const headers = {'Content-Type':'application/json'};
    if ($('access-code').value) headers['X-Demo-Code'] = $('access-code').value;
    const response = await fetch('/api/chat', {method:'POST', headers, body:JSON.stringify({question, history:history.slice(-6)}), signal:AbortSignal.timeout(65000)});
    const body = await response.json(); if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '질문을 확인하고 다시 시도해 주세요.');
    pending.remove(); message(body.answer, 'assistant', body.sources);
    history.push({role:'user',content:question},{role:'assistant',content:body.answer.slice(0,6000)});
    history = history.slice(-6);
  } catch(error) {pending.remove(); message(error.name === 'TimeoutError' ? '응답이 늦어지고 있어요. 잠시 후 다시 시도해 주세요.' : error.message, 'error'); $('question').value = question;}
  finally {busy=false; $('send').disabled=!ready; $('clear').disabled=false; $('question').focus();}
});
$('question').addEventListener('keydown', event => {if(event.key === 'Enter' && !event.shiftKey && !event.isComposing){event.preventDefault();$('chat-form').requestSubmit();}});
$('clear').addEventListener('click', () => {if(busy)return;history=[];$('messages').replaceChildren();message('새로운 상담을 시작할게요. 궁금한 내용을 알려주세요.','assistant');});
for(const button of document.querySelectorAll('.example')) button.addEventListener('click', () => {$('question').value=button.textContent;$('question').focus();});
$('status').addEventListener('click', health);
health();
