"""
학교 공지 맞춤 AI Agent (KCC 미니 캡스톤)

[Agent 모드]  목표만 말하면 AI가 스스로 계획하고, 도구(게시판 읽기·공지 열람·기억 확인·캘린더 등록)를
              골라 쓰면서 일하고, 그 과정을 화면에 보여줍니다.
[홈] 공지 전체를 한 번에 판단해서 D-day 순 카드로 보여줍니다. (Agent가 안 될 때 백업)

실행: streamlit run app.py
"""

import json
import os
from datetime import date

import streamlit as st

from agent_tools import (TOOL_DECLARATIONS, Toolbox, collect_notices, d_day, d_day_badge, load_memory,
                         load_sample_notices, make_ics, parse_notices, remember_feedback,
                         remember_note, save_memory)

# 내 키로 쓸 수 있는 모델을 Google에 물어봐서 자동으로 골라요. 목록을 못 받을 때만 아래 이름을 써요.
MODELS = ["gemini-3.8-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview"]
MAX_STEPS = 15          # Agent가 도구를 쓰는 최대 횟수 (무한 반복 방지)
CACHE_FILE = "last_result.json"

TOOL_LABELS = {
    "list_boards": "📋 게시판 목록 확인",
    "read_board": "🔍 게시판 살펴보기",
    "read_notice": "📄 공지 자세히 읽기",
    "read_notices": "📄 공지 여러 개 자세히 읽기",
    "get_user_memory": "🧠 내 취향 기억 꺼내기",
    "add_to_calendar": "📅 캘린더에 마감일 등록",
}


# ──────────────────────────────────────────────
# AI 연결
# ──────────────────────────────────────────────
def get_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


MAX_WAIT = 15   # 무료 한도에 걸렸을 때 최대로 기다리는 시간(초). 1분 안에 끝나도록 짧게


def _ver(name):
    import re
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def model_list(client):
    """내 키로 지금 쓸 수 있는 Flash 모델을 최신순으로 (예전 모델이 막혀도 자동으로 새 모델 사용).
    lite·preview는 뒤로. 한 번 알아내면 이 접속 동안 기억해요."""
    if st.session_state.get("model_list"):
        return st.session_state["model_list"]
    names = []
    try:
        for m in client.models.list():
            n = str(getattr(m, "name", "")).split("/")[-1]
            acts = getattr(m, "supported_actions", None) or ["generateContent"]
            bad = ("image", "tts", "audio", "live", "embed", "vision", "thinking-exp", "learnlm")
            if n.startswith("gemini-") and "flash" in n and "generateContent" in acts and not any(b in n for b in bad):
                names.append(n)
    except Exception:
        names = []
    # 무료 한도가 넉넉한 lite를 먼저 → 그다음 일반 모델 (각각 최신순, 시험판 preview·exp는 맨 뒤)
    # 한 모델이 한도에 걸리면 다음 모델로 넘어가요. 한도는 모델마다 따로라 더 많이 쓸 수 있어요.
    names = sorted(set(names), key=lambda n: ("preview" in n or "exp" in n, "lite" not in n, -_ver(n), len(n)))
    out = names + [m for m in MODELS if m not in names]
    st.session_state["model_list"] = out
    return out


def generate(client, contents, config, log=None, budget=None):
    """모델을 순서대로 시도합니다.
    무료 사용량(1분당 요청 수)을 넘기면 바로 다른 모델로 바꾸고, 모두 막혔을 때만 잠깐(최대 15초) 기다려요."""
    import re
    import time

    t_start = time.time()
    last, waited, daily, out_of_time = None, 0, set(), False
    for attempt in range(3):
        # 지난번에 성공한 모델을 먼저 시도
        models = sorted(model_list(client), key=lambda m: m != st.session_state.get("model"))
        wait = 0
        for m in models:
            if m in daily:          # 오늘 사용량을 다 쓴 모델은 건너뛰기
                continue
            if budget and last is not None and time.time() - t_start > budget:
                out_of_time = True   # 시간 제한: 몇 분씩 돌지 않게 여기서 멈춰요
                break
            try:
                resp = client.models.generate_content(model=m, contents=contents, config=config)
                st.session_state["model"] = m
                return resp
            except Exception as e:
                last = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:  # 사용량 초과 → 다른 모델 시도
                    if "PerDay" in msg or "per day" in msg.lower():   # 하루 한도: 기다려도 소용없음
                        daily.add(m)
                        continue
                    found = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay': '(\d+)s", msg)
                    wait = max(wait, float(found.group(1)) if found else 30)
                if any(k in msg for k in ("503", "UNAVAILABLE", "overloaded", "high demand", "500 INTERNAL")):
                    wait = max(wait, 4)   # Google 서버가 잠깐 붐빔 → 다른 모델 시도, 다 안 되면 몇 초 뒤 다시
                    continue
                if "thinking" in msg.lower() and ("400" in msg or "INVALID" in msg):
                    raise RuntimeError("THINKING_UNSUPPORTED: " + msg)   # 모델을 바꾸지 말고 '생각 끄기' 없이 다시
                # 그 밖의 오류(모델 이름 없음·예전 모델 막힘 등)도 다음 모델로
                if "NOT_FOUND" in msg or "no longer available" in msg:
                    st.session_state["model_list"] = [x for x in model_list(client) if x != m]
        if out_of_time or not wait or waited >= MAX_WAIT:
            break
        wait = min(wait + 1, MAX_WAIT - waited)
        if log:
            busy = last and any(k in str(last) for k in ("503", "UNAVAILABLE", "high demand"))
            log(f"⏳ {'Google AI 서버가 붐벼서' if busy else '무료 사용량 한도에 걸려서'} {int(wait)}초만 기다렸다가 이어서 할게요...")
        time.sleep(wait)
        waited += wait
    if daily and all(m in daily for m in model_list(client)):
        raise RuntimeError("오늘 무료 사용량을 모두 썼어요. 오후 5시쯤 초기화돼요.")
    if last and ("429" in str(last) or "RESOURCE_EXHAUSTED" in str(last)):
        raise RuntimeError("AI 무료 사용량 한도에 걸렸어요. 30초~1분 뒤에 다시 눌러 주세요. "
                           "(그동안 이미 판단한 결과는 그대로 볼 수 있어요)")
    if last and any(k in str(last) for k in ("503", "UNAVAILABLE", "overloaded", "high demand", "500 INTERNAL")):
        raise RuntimeError("지금 Google AI 서버에 사람이 몰려서 잠깐 응답을 못 해요. 10~30초 뒤에 다시 눌러 주세요. "
                           "(내 키나 사용량 문제는 아니에요)")
    raise RuntimeError(f"AI 호출 실패: {last}")


def profile_text(p):
    return (f"- 학과: {p['major']}\n- 학년: {p['grade']}\n- 관심사: {', '.join(p['interests'])}\n"
            f"- 추가 정보: {p['extra'] or '없음'}")


# ──────────────────────────────────────────────
# Agent 모드: 생각 → 도구 사용 → 결과 확인 → 반복
# ──────────────────────────────────────────────
def agent_system_prompt(profile, today, boards):
    return f"""너는 대학생의 학교 공지를 챙겨주는 비서 AI Agent야. 오늘은 {today.isoformat()}이야.

## 사용자 프로필
{profile_text(profile)}

## 확인할 게시판
{', '.join(boards)}

## 일하는 방법 (AI 호출 횟수가 제한돼 있으니, 도구는 한 번에 여러 개를 동시에 불러)
1. 도구를 쓰기 전에 무엇을 할지 한 줄로 계획을 말해.
2. 첫 번째 차례에 get_user_memory와 모든 게시판의 read_board를 한꺼번에 동시에 호출해.
3. 제목을 보고 사용자와 관련 있어 보이는 공지만 골라 read_notices 한 번으로 같이 열어 (최대 5개).
4. 본문에서 지원 대상에 사용자가 해당하는지 확인해. 학과 이름이 없어도 의미로 판단해
   (예: '공과대학 재학생', '전기·전자 계열'은 전자공학부 포함).
5. 추천하는 공지 중 마감일이 있는 것은 add_to_calendar를 한꺼번에 동시에 호출해 등록해.
6. 기억에서 싫어한 공지와 비슷한 건 제외하고, 좋아한 공지와 비슷한 건 우선해.
7. 공지에 없는 내용은 절대 지어내지 마.

## 최종 답변 형식 (한국어, 마크다운)
**한 줄 요약**
### 추천 공지 (마감 급한 순)
각 공지마다: 제목 · D-day · 마감일
- 추천 이유: 프로필이나 기억을 근거로 한 문장
- 준비물 체크리스트: 신청에 필요한 것 (공지에 적힌 것만)
### 제외한 공지
제목 — 이유 (짧게)
"""


def run_agent(api_key, profile, boards, today, goal, log):
    from google.genai import types

    client = get_client(api_key)
    toolbox = Toolbox(boards, today)
    config = types.GenerateContentConfig(
        system_instruction=agent_system_prompt(profile, today, boards),
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )
    contents = [types.Content(role="user", parts=[types.Part(text=goal)])]
    steps = []

    for _ in range(MAX_STEPS):
        resp = generate(client, contents, config, log)
        content = resp.candidates[0].content
        contents.append(content)

        # Agent의 생각(계획) 보여주기
        calls = resp.function_calls or []
        thought = " ".join(p.text for p in (content.parts or []) if getattr(p, "text", None)).strip()
        if calls and thought:
            steps.append(("💭 생각", thought))
            log(f"💭 **생각**: {thought}")

        if not calls:  # 더 쓸 도구가 없으면 최종 답변
            return {"answer": resp.text or thought, "steps": steps,
                    "calendar": toolbox.calendar, "new": toolbox.new_found}

        # 도구 실행
        parts = []
        for fc in calls:
            args = dict(fc.args or {})
            result = toolbox.call(fc.name, args)
            label = TOOL_LABELS.get(fc.name, fc.name)
            detail = summarize_tool(fc.name, args, result)
            steps.append((label, detail))
            log(f"{label} — {detail}")
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))

    return {"answer": "⚠️ 작업이 너무 길어져서 멈췄어요. 목표를 더 구체적으로 말해 주세요.",
            "steps": steps, "calendar": toolbox.calendar, "new": toolbox.new_found}


def summarize_tool(name, args, result):
    if "error" in result:
        return f"⚠️ {result['error']}"
    if name == "list_boards":
        return ", ".join(result["boards"])
    if name == "read_board":
        new = sum(1 for n in result["notices"] if n.get("new"))
        return f"{args.get('board')}: 공지 {result['count']}개 (새 공지 {new}개)"
    if name == "read_notices":
        ok = [n for n in result["notices"] if "content" in n]
        return f"{len(ok)}개 열람: " + " / ".join(n["content"].splitlines()[0].replace("제목:", "").strip()[:30] for n in ok)
    if name == "read_notice":
        first = result["content"].splitlines()[0][:50]
        return first
    if name == "get_user_memory":
        return f"좋아함 {len(result['좋아한 공지'])}개 · 싫어함 {len(result['싫어한 공지'])}개 · 메모 {len(result['사용자 메모'])}개"
    if name == "add_to_calendar":
        return f"{args.get('title')} → {args.get('deadline')}" + (f" (D-{result['d_day']})" if "d_day" in result else "")
    return ""


# ──────────────────────────────────────────────
# 홈 모드 (전체 공지를 AI가 한 번에 판단 → 대시보드로 표시)
# ──────────────────────────────────────────────
CACHE_DIR = "cache"   # 판단 결과를 '프로필 + 게시판(공지)' 조합마다 따로 저장 → 접속한 사람마다 자기 결과만 봐요
TAG_SET = "취업|자격증|장학|연구|대외활동|학사|행사|기타"


def batch_prompt(profile, notices, today):
    notice_text = "\n\n".join(f"[공지 {n['id']}]\n{n['text']}" for n in notices)
    return f"""너는 대학생의 학교 공지를 대신 읽어주는 비서야. 오늘은 {today.isoformat()}이야.

## 사용자 프로필
{profile_text(profile)}


## 판단 규칙
1. 지원 대상에 사용자가 해당하는지 먼저 확인. 학과 이름이 없어도 의미로 판단해
   (예: '공과대학 재학생', '전기·전자 계열'은 전자공학부 포함). 대상이 아니면 relevant=false.
2. 관심사 관련도 0~100점. 신청할 게 없는 안내성 공지는 relevant=false.
3. 사용자 추가 정보에 적힌 희망 진로·관심을 점수에 반영해.
4. reason은 프로필을 근거로 한 문장, 40자 이내. 공지에 없는 내용은 지어내지 마.
5. deadline은 신청 마감일 YYYY-MM-DD, 없으면 null.
6. category는 {TAG_SET} 중 하나.
7. short는 타임라인에 쓸 6자 이내 짧은 이름 (예: 채용박람회, 전기기사, 장학생).
8. kind는 이 공지의 유형을 2~6자로 (예: 채용박람회, 공모전, 봉사, 성적장학금). 사용자가 '관심 없음'을 누르면 같은 유형을 거를 때 써.
9. checklist는 신청에 필요한 준비물, 공지에 적힌 것만 최대 3개.

## 출력: JSON 배열만, 모든 공지에 대해
[{{"id": "공지번호", "relevant": true, "score": 0, "category": "취업",
"short": "", "kind": "", "reason": "", "summary": "2문장 이내",
"deadline": "YYYY-MM-DD 또는 null", "target": "", "how_to_apply": "", "checklist": ["준비물"]}}]

## 공지 목록
{notice_text}
"""




def ai_call(api_key, prompt, read_links=False, json_mode=False, log=None, fast=False, budget=None):
    """Gemini 부르기. 빠르게: '생각하는 시간'을 끄고(thinking_budget=0) 바로 답하게 해요.
    read_links=True면 Gemini가 글 속의 링크(URL)를 직접 열어서 읽어요 (URL context 기능)."""
    from google.genai import types

    base = {"temperature": 0.2}
    if read_links:
        base["tools"] = [types.Tool(url_context=types.UrlContext())]
    elif json_mode:
        base["response_mime_type"] = "application/json"
    client = get_client(api_key)
    prev = st.session_state.get("model")
    if fast:   # 빠른 모델(lite)을 먼저 고르게 (안 되면 generate가 다른 모델로 넘어가요)
        ml = model_list(client)
        st.session_state["model"] = next((m for m in ml if "lite" in m), ml[0] if ml else None)
    try:
        if st.session_state.get("no_think"):
            return generate(client, prompt, types.GenerateContentConfig(**base), log, budget)
        try:
            cfg = types.GenerateContentConfig(**base, thinking_config=types.ThinkingConfig(thinking_budget=0))
            return generate(client, prompt, cfg, log, budget)
        except Exception as e:
            if "THINKING_UNSUPPORTED" not in str(e):
                raise
            # 이 모델은 '생각 끄기'를 지원하지 않아요 → 이번 접속 동안은 기본 설정으로
            st.session_state["no_think"] = True
            return generate(client, prompt, types.GenerateContentConfig(**base), log, budget)
    finally:
        if fast:   # 판단 같은 중요한 일은 원래 모델로
            st.session_state["model"] = prev


def parse_json_list(text):
    """AI 답에서 JSON 배열을 꺼내요.
    앞뒤에 설명 글이 붙거나, ```json``` 으로 감싸거나, {"results": [...]} 모양이어도 찾아내요."""
    import re
    t = re.sub(r"```(?:json)?", "", text or "")
    dec = json.JSONDecoder()
    best = []
    for m in re.finditer(r"[\[{]", t):
        try:
            v, _ = dec.raw_decode(t[m.start():])
        except Exception:
            continue
        if isinstance(v, dict):
            v = next((x for x in v.values() if isinstance(x, list)), None)
        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
            items = [x for x in v if isinstance(x, dict)]
            if len(items) > len(best):
                best = items
    return best


def failed_links(resp):
    """Gemini가 열지 못한 링크 목록 (답에 함께 오는 정보)"""
    out = []
    try:
        for m in resp.candidates[0].url_context_metadata.url_metadata or []:
            if "SUCCESS" not in str(getattr(m, "url_retrieval_status", "SUCCESS")):
                out.append(getattr(m, "retrieved_url", ""))
    except Exception:
        pass
    return [u for u in out if u]


MAX_LINKS = 20   # Gemini가 한 번에 읽을 수 있는 링크 수
MAX_JUDGE = 150  # 한 번에 판단하는 공지 수 상한 (게시판마다 최대 30개)


def ai_find_posts(api_key, boards, today=None, log=None):
    """1단계: Gemini가 게시판 페이지를 직접 읽고 최신 공지 목록(제목·주소)을 찾아요.
    마감이 이미 지난 글은 빼고, 아직 신청할 수 있는 글 위주로 게시판마다 최대 4개."""
    from urllib.parse import urljoin
    items = list(boards.items())[:MAX_LINKS]
    per_board = max(2, min(4, MAX_LINKS // max(1, len(items))))
    today_txt = today.isoformat() if today else "오늘"
    listing = "\n".join(f"- {n}: {u}" for n, u in items)
    prompt = f"""오늘은 {today_txt}이야. 아래 학교 게시판 페이지를 직접 열어서 읽고, 게시판마다 아직 신청할 수 있는 최근 공지 글을 최대 {per_board}개씩 골라 제목과 글 주소를 알려줘.
## 게시판
{listing}
## 규칙
- 게시판 페이지에 실제로 보이는 공지 글만. 메뉴, 페이지 번호, 첨부파일 링크는 빼.
- url은 그 글을 누르면 열리는 주소를 https:// 로 시작하는 전체 주소로.
- 맨 위 고정 공지보다 최근에 올라온 글을 우선해.
- 제목이나 목록에 마감일이 보이고 그 날짜가 오늘보다 앞이면(이미 마감) 빼고, 목록을 더 내려서 다른 글을 골라.
- 결과·합격자 발표, 단순 안내처럼 신청할 게 없는 글보다 모집·신청·채용·장학·특강 글을 우선해.
- 절대 지어내지 마. 페이지를 못 읽은 게시판은 빼.
## 출력: JSON 배열만
[{{"board": "게시판 이름", "title": "글 제목", "date": "게시일(있으면)", "url": "https://..."}}]"""
    resp = ai_call(api_key, prompt, read_links=True, log=log, fast=True, budget=40)
    posts, seen, count = [], set(), {}
    for p in parse_json_list(resp.text):
        if not isinstance(p, dict) or not p.get("title"):
            continue
        board = p.get("board") or items[0][0]
        url = str(p.get("url") or "")
        if url and not url.startswith("http"):
            url = urljoin(boards.get(board, items[0][1]), url)
        if url in seen or count.get(board, 0) >= per_board:
            continue
        seen.add(url)
        count[board] = count.get(board, 0) + 1
        posts.append({"id": str(len(posts) + 1), "board": board, "title": str(p["title"]).strip(),
                      "date": p.get("date", "") or "", "url": url})
    return posts[:MAX_LINKS], failed_links(resp)


PER_BOARD = 30   # 게시판마다 최근 두 달 치 공지를 최대 몇 개까지 볼지
MAX_PAGES = 4    # 게시판마다 최대 몇 페이지까지 넘겨 볼지


def _get_soup(url):
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (student project)"})
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup


def _page_posts(soup, name, url):
    """목록 페이지 하나에서 공지 글(제목·주소·게시일)을 페이지에 나온 순서대로 꺼내요"""
    import re
    from urllib.parse import urljoin, urlparse
    base = urlparse(url).path.rsplit("/index.do", 1)[0].rstrip("/")
    date_re = re.compile(r"(20\d{2})[.\-/]\s?(\d{1,2})[.\-/]\s?(\d{1,2})")
    found, seen = [], set()
    for a in soup.find_all("a", href=True):
        title = " ".join(a.get_text().split())
        href = urljoin(url, a["href"]).split("#")[0]
        path = urlparse(href).path
        if len(title) < 6 or not href.startswith("http") or href in seen:
            continue
        if not (path.startswith(base + "/") and re.search(r"/\d{2,}(/|$)", path[len(base):])):
            continue
        seen.add(href)
        row = a.find_parent(["tr", "li"]) or a.parent
        m = date_re.search(" ".join(row.get_text(" ").split())) if row else None
        posted = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""
        found.append({"board": name, "title": title[:120], "date": posted, "url": href})
    return found


def _page_urls(soup, url, n):
    """n번째 페이지 주소 후보: 페이지에 있는 '2','3' 링크를 먼저, 없으면 흔한 주소 모양으로"""
    import re
    from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
    out = []
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True) == str(n):
            h = a["href"]
            if not h.lower().startswith("javascript") and h != "#":
                out.append(urljoin(url, h))
    u = urlparse(url)
    q = dict(parse_qsl(u.query))
    for key in ("page", "pageIndex", "currentPageNo", "pg", "pageNo"):
        out.append(urlunparse(u._replace(query=urlencode({**q, key: n}))))
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:7]


def _recent(p, today):
    from datetime import date as _d
    try:
        y, m, d = map(int, p["date"].split("-"))
        return (today - _d(y, m, d)).days <= OLD_DAYS
    except Exception:
        return None   # 게시일 모름


def _read_one_board(name, url, today):
    """게시판 하나에서 최근 두 달 치 공지를 모아요 (AI 없이). 마지막 글도 두 달 안쪽이면 다음 페이지로."""
    soup = _get_soup(url)
    page = _page_posts(soup, name, url)
    posts, seen = list(page), {p["url"] for p in page}
    pattern = None   # 2페이지에서 통한 주소 모양을 3·4페이지에도 써요
    for n in range(2, MAX_PAGES + 1):
        if len([p for p in posts if _recent(p, today) is not False]) >= PER_BOARD:
            break
        if not page or _recent(page[-1], today) is not True:   # 마지막 글이 두 달 넘었거나 날짜를 모르면 그만
            break
        cands = [pattern.replace("__N__", str(n))] if pattern else _page_urls(soup, url, n)
        page = []
        for cu in cands:
            try:
                sp = _get_soup(cu)
            except Exception:
                continue
            new = [p for p in _page_posts(sp, name, url) if p["url"] not in seen]
            if new:
                page, soup = new, sp
                if not pattern:
                    import re
                    m = re.search(rf"([?&][A-Za-z]+=){n}(?=&|$)", cu)
                    pattern = cu[:m.start()] + m.group(1) + "__N__" + cu[m.end():] if m else None
                break
        if not page:
            break
        posts += page
        seen |= {p["url"] for p in page}
    import re
    posts = [p for p in posts if _recent(p, today) is not False or re.search(r"상시|수시", p["title"])]   # 두 달 넘은 글은 빼요
    posts.sort(key=lambda p: p["date"] or "", reverse=True)
    return posts[:PER_BOARD]


def quick_find_posts(boards, today):
    """1단계(빠른 길): 모든 게시판을 동시에 열어 최근 두 달 치 목록을 읽어요. 못 읽은 게시판은 따로 돌려줘요."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(boards.items())[:MAX_LINKS]
    posts, failed = [], {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [(n, u, ex.submit(_read_one_board, n, u, today)) for n, u in items]
        for n, u, f in futs:
            try:
                got = f.result(timeout=40)
            except Exception:
                got = []
            if got:
                posts.extend(got)
            else:
                failed[n] = u
    for i, p in enumerate(posts):
        p["id"] = str(i + 1)
    return posts, failed


def ai_judge_posts(api_key, profile, posts, today, log=None):
    """2단계: Gemini가 공지 글 주소를 직접 열어 읽고 판단해요."""
    listing = "\n".join(f"[공지 {p['id']}] {p['title']} ({p['board']}) {p['url']}" for p in posts)
    prompt = batch_prompt(profile, [], today).replace("## 공지 목록", "## 공지 목록 (각 주소를 직접 열어 본문을 읽고 판단해)") \
        + listing + "\n\n본문을 못 읽은 공지는 제목만 보고 판단하고, reason 끝에 '(본문 확인 필요)'라고 붙여.\n"
    resp = ai_call(api_key, prompt, read_links=True, log=log)
    return match_results(parse_json_list(resp.text), posts), failed_links(resp)


def match_results(results, posts):
    """AI 판단 결과를 공지와 짝지어요. 번호("공지 3", 3, "3")로, 안 되면 제목으로."""
    import re
    by_id = {p["id"]: p for p in posts}
    norm = lambda x: re.sub(r"\s+", "", str(x or ""))
    out, used = [], set()
    for r in results:
        m = re.search(r"\d+", str(r.get("id", "")))
        p = by_id.get(m.group(0)) if m else None
        if not p and r.get("title"):   # 번호가 없으면 제목이 서로 포함되는 공지로
            t = norm(r["title"])
            p = next((q for q in posts if q["id"] not in used and (norm(q["title"]) in t or t in norm(q["title"]))), None)
        if not p or p["id"] in used:
            continue
        used.add(p["id"])
        r.update(id=p["id"], title=p["title"], url=p["url"], board=p["board"])
        out.append(r)
    return out


TITLE_CHUNK = 25   # 제목 판단을 몇 개씩 나눠 보낼지 (한 번에 너무 많으면 AI가 일부만 답해요)


def _title_prompt(profile, posts, today):
    """제목 판단용 짧은 프롬프트. 답도 짧게 받아서 빨라요 (요약·준비물은 나중에 본문에서)."""
    lines = "\n".join(f"[{p['id']}] {p['title']} | {p['board']} | {p.get('date', '')}" for p in posts)
    return f"""너는 대학생의 학교 공지를 골라주는 비서야. 오늘은 {today.isoformat()}이야.
## 사용자 프로필
{profile_text(profile)}
## 규칙 (게시판 목록의 제목·게시판·게시일만 보고 판단)
- relevant: 사용자가 신청·참여할 수 있고 관심사·진로와 관련 있으면 true. 대상이 분명히 다르거나(다른 학과·학년 전용 등) 신청할 게 없는 단순 안내·결과 발표면 false.
- score: 관련도 0~100
- category: {TAG_SET} 중 하나
- short: 6자 이내 짧은 이름, kind: 공지 유형 2~6자
- reason: 프로필을 근거로 한 문장 30자 이내
- deadline: 제목에 마감일·기간(~11/12 등)이 보이면 YYYY-MM-DD, 없으면 null. 마감이 오늘보다 앞이면 relevant=false.
## 출력: JSON 배열만. 공지 {len(posts)}개 모두 빠짐없이.
[{{"id":"번호","relevant":true,"score":0,"category":"","short":"","kind":"","reason":"","deadline":null}}]
## 공지
{lines}"""


def _judge_titles_once(api_key, profile, posts, today, log=None):
    resp = ai_call(api_key, _title_prompt(profile, posts, today), json_mode=True, log=log, fast=True, budget=40)
    return match_results(parse_json_list(resp.text), posts)


def _parallel_texts(api_key, prompts, on_done=None):
    """여러 프롬프트를 동시에 Gemini에 보내요. 스레드 안에서는 st를 쓰지 않아요."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google.genai import types
    client = get_client(api_key)
    models = model_list(client)
    models = sorted(models, key=lambda m: ("lite" not in m))[:3]
    base = {"temperature": 0.2, "response_mime_type": "application/json"}
    no_think = st.session_state.get("no_think")

    def work(prompt):
        for m in models:
            for think in ([False] if no_think else [True, False]):
                cfg = types.GenerateContentConfig(**base, **({"thinking_config": types.ThinkingConfig(thinking_budget=0)} if think else {}))
                try:
                    return client.models.generate_content(model=m, contents=prompt, config=cfg).text
                except Exception as e:
                    if think and "thinking" in str(e).lower():
                        continue   # 생각 끄기를 못 하는 모델 → 기본 설정으로 다시
                    break          # 한도·서버 붐빔 등 → 다음 모델
        return None

    out = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(work, p): i for i, p in enumerate(prompts)}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:
                pass
            if on_done:
                on_done()
    return out


def ai_judge_titles(api_key, profile, posts, today, log=None, progress=None):
    """제목·게시일만 보고 판단. 묶음을 동시에 보내고, AI가 빠뜨린 공지만 한 번 더 물어봐요."""
    chunks = [posts[i:i + TITLE_CHUNK] for i in range(0, len(posts), TITLE_CHUNK)]
    done_n = [0]

    def tick():
        done_n[0] += 1
        if progress:
            progress(min(done_n[0] * TITLE_CHUNK, len(posts)), len(posts))

    texts = _parallel_texts(api_key, [_title_prompt(profile, c, today) for c in chunks], tick)
    results = []
    for c, t in zip(chunks, texts):
        if t:
            results += match_results(parse_json_list(t), c)
    missing = [p for p in posts if p["id"] not in {r["id"] for r in results}]
    if missing:   # 빠졌거나 실패한 공지만 다시 (한 번)
        try:
            results += _judge_titles_once(api_key, profile, missing[:TITLE_CHUNK * 2], today, log)
        except RuntimeError:
            if not results:
                raise
    # 마감이 이미 지난 공지는 아예 없애요 (숨김 창에도 안 나와요)
    past = {r["id"] for r in results if len(str(r.get("deadline") or "")) == 10 and str(r["deadline"]) < today.isoformat()}
    results = [r for r in results if r["id"] not in past]
    posts = [p for p in posts if p["id"] not in past]
    # 그래도 답이 없는 공지는 숨김 창에서 볼 수 있게 남겨요
    done = {r["id"] for r in results}
    for p in posts:
        if p["id"] not in done:
            results.append({"id": p["id"], "title": p["title"], "url": p["url"], "board": p["board"],
                            "relevant": False, "score": 0, "category": "기타", "short": p["title"][:6], "kind": "",
                            "reason": "AI가 판단하지 못했어요 (다시 판단하면 바뀔 수 있어요)", "summary": "",
                            "deadline": None, "target": "", "how_to_apply": "", "checklist": []})
    order = {p["id"]: i for i, p in enumerate(posts)}
    return sorted(results, key=lambda r: order.get(r["id"], 0))


MAX_FILL = 8   # 추천된 공지 중 본문을 열어 볼 최대 개수
OLD_DAYS = 60  # 게시일이 이보다 오래된 공지는 마감됐을 가능성이 높아서 미리 빼요 (상시·수시 모집은 남겨요)


def drop_old_posts(posts, today):
    """게시일이 두 달(OLD_DAYS)보다 오래된 공지를 빼요. 게시일을 모르거나 제목에 '상시·수시'가 있으면 남겨요."""
    import re
    from datetime import date as _d
    keep, old = [], 0
    for p in posts:
        m = re.search(r"(20\d{2})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})", str(p.get("date") or ""))
        try:
            posted = _d(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
        except ValueError:
            posted = None
        if posted and (today - posted).days > OLD_DAYS and not re.search(r"상시|수시", p.get("title", "")):
            old += 1
            continue
        keep.append(p)
    return keep, old


def _body_text(url, limit=2500):
    """공지 글 페이지에서 본문 글자만 꺼내요 (메뉴·머리말 빼고, 가장 긴 본문 영역)"""
    soup = _get_soup(url)
    best = None
    for sel in (".view_cont", ".view-con", ".board_view", ".bbs_view", ".board-view", "article",
                "div[class*=view]", "td.content", "#content", "main", ".content", "div[class*=cont]"):
        for t in soup.select(sel):
            if len(t.get_text(" ", strip=True)) >= 150:   # 본문처럼 충분히 긴 첫 영역
                best = t
                break
        if best:
            break
    best = best or soup.body or soup
    text = " ".join(best.get_text(" ").split())
    return text[:limit]


def ai_fill_details(api_key, profile, results, today, log=None):
    """2단계(가볍게): 제목으로 추천된 공지 중 마감일이 안 보이는 것만 Gemini가 본문을 열어
    마감일·대상·신청 방법·준비물을 채워요. 실패하면 제목 판단 결과를 그대로 둬요."""
    no_dl = lambda r: str(r.get("deadline") or "").strip().lower() in ("", "null", "none", "없음")
    todo = [r for r in results if r.get("relevant") and no_dl(r) and r.get("url")]
    todo = sorted(todo, key=lambda r: -(r.get("score") or 0))[:MAX_FILL]
    if not todo:
        return 0
    # 본문은 코드가 동시에 열어서 글자만 넘겨요 (Gemini가 링크를 하나씩 여는 것보다 훨씬 빨라요)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [(r, ex.submit(_body_text, r["url"])) for r in todo]
        bodies = {}
        for r, f in futs:
            try:
                bodies[r["id"]] = f.result(timeout=12)
            except Exception:
                pass
    got_text = [r for r in todo if bodies.get(r["id"])]
    if got_text:
        listing = "\n\n".join(f"[공지 {r['id']}] {r.get('title', '')}\n{bodies[r['id']]}" for r in got_text)
        how = "아래 공지 본문을 읽고"
        todo = got_text
    else:   # 코드로 못 열었으면 예전처럼 Gemini가 링크를 직접 열어요
        listing = "\n".join(f"[공지 {r['id']}] {r.get('title', '')} {r['url']}" for r in todo)
        how = "아래 공지 주소를 직접 열어 본문을 읽고"
    prompt = f"""오늘은 {today.isoformat()}이야. {how}, 공지마다 정보를 채워줘.
## 사용자 프로필
{profile_text(profile)}
## 규칙
- deadline: 신청 마감일 YYYY-MM-DD, 본문에도 없으면 null
- target: 지원 대상 한 줄, how_to_apply: 신청 방법 한 줄, checklist: 준비물 최대 3개 (본문에 적힌 것만)
- summary: 2문장 이내
- relevant: 본문을 보니 사용자가 대상이 아니거나 이미 마감이면 false, 아니면 true
- 본문을 못 읽은 공지는 빼. 절대 지어내지 마.
## 출력: JSON 배열만
[{{"id": "공지번호", "relevant": true, "deadline": null, "target": "", "how_to_apply": "", "checklist": [], "summary": ""}}]
## 공지
{listing}"""
    resp = ai_call(api_key, prompt, read_links=not got_text, json_mode=bool(got_text), log=log, fast=True, budget=35)
    by_id = {r["id"]: r for r in todo}
    import re
    filled = 0
    for x in parse_json_list(resp.text):
        m = re.search(r"\d+", str(x.get("id", "")))
        r = by_id.get(m.group(0)) if m else None
        if not r:
            continue
        for k in ("deadline", "target", "how_to_apply", "summary"):
            if x.get(k) and str(x[k]).lower() != "null":
                r[k] = x[k]
        if isinstance(x.get("checklist"), list) and x["checklist"]:
            r["checklist"] = x["checklist"][:3]
        if x.get("relevant") is False:
            r["relevant"] = False
            r["reason"] = (r.get("reason") or "") + " (본문 확인: 대상 아님/마감)"
        filled += 1
    return filled


def run_batch(api_key, profile, notices, today):
    """예시 공지·붙여넣은 공지 판단 (글이 이미 있으니 링크를 읽을 필요 없음)"""
    resp = ai_call(api_key, batch_prompt(profile, notices, today), json_mode=True, log=st.caption)
    results = parse_json_list(resp.text)
    by_id = {n["id"]: n for n in notices}
    for r in results:
        r["id"] = str(r["id"])
        n = by_id.get(r["id"], {})
        r["title"] = n.get("title", f"공지 {r['id']}")
        r["url"] = n.get("url", "")
        r["board"] = n.get("board", "")
    return results


def batch_key(profile, boards, notice_text):
    """프로필과 공지 출처로 만든 열쇠. 같은 프로필·같은 게시판이면 같은 결과를 다시 써요 (AI 절약)."""
    import hashlib
    src = sorted(boards.values()) if boards else (notice_text or "")
    raw = json.dumps([profile["major"], profile["grade"], sorted(profile["interests"]), profile["extra"], src],
                     ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_batch(key):
    """먼저 이 접속(세션)의 결과, 없으면 같은 열쇠로 저장해 둔 결과를 찾아요."""
    if st.session_state.get("batch_key") == key and st.session_state.get("batch"):
        return st.session_state["batch"]
    try:
        b = json.load(open(os.path.join(CACHE_DIR, key + ".json"), encoding="utf-8"))
        st.session_state.update(batch=b, batch_key=key)
        return b
    except Exception:
        return None


def save_batch(key, batch):
    st.session_state.update(batch=batch, batch_key=key)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(batch, open(os.path.join(CACHE_DIR, key + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass  # 저장이 안 돼도 이 접속에서는 계속 보여요


def profile_label(p):
    return f"{p.get('major', '')} {p.get('grade', '')} · {', '.join(p.get('interests', [])) or '-'}"


def standalone_html(entries):
    """제출용: 판단 결과를 담은 HTML 파일 하나 (서버·인터넷·API 키 없이 더블클릭으로 열림)"""
    tpl = open("dashboard.html", encoding="utf-8").read()
    data = json.dumps({"standalone": True, "profiles": entries}, ensure_ascii=False).replace("</", "<\\/")
    return tpl.replace('/*__DATA__*/{"today": "2026-10-30", "interests": [], "results": []}', data)


def export_entries(current_batch, today):
    """지금 결과 + 이 컴퓨터에 저장된 다른 프로필 결과(같은 프로필은 최신 1개)"""
    items = []
    if current_batch and current_batch.get("profile"):
        items.append(current_batch)
    try:
        files = sorted((os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".json")),
                       key=os.path.getmtime, reverse=True)
        for f in files:
            b = json.load(open(f, encoding="utf-8"))
            if b.get("profile") and b.get("results"):
                items.append(b)
    except Exception:
        pass
    seen, out = set(), []
    for b in items:
        lab = profile_label(b["profile"])
        if lab in seen:
            continue
        seen.add(lab)
        out.append({"label": lab, "today": b.get("today") or today.isoformat(),
                    "interests": b["profile"].get("interests", []), "results": b["results"]})
    return out


def show_dashboard(results, today, interests, height=1450, mode="list", all_tags=False):
    import streamlit.components.v1 as components

    tpl = open("dashboard.html", encoding="utf-8").read()
    data = json.dumps({"today": today.isoformat(), "interests": interests, "results": results, "allTags": all_tags},
                      ensure_ascii=False).replace("</", "<\\/")
    html = tpl.replace('/*__DATA__*/{"today": "2026-10-30", "interests": [], "results": []}', data)
    if mode in ("calendar", "hidden", "memory"):
        html = html.replace('/*__MODE__*/"list"', f'"{mode}"')
    components.html(html, height=height, scrolling=True)


# 자주 쓰는 학교 게시판은 주소만 보고 바로 이름을 붙여요
KNOWN_BOARDS = [
    ("ee.kookmin.ac.kr/community/board/notice", "전자공학부 학부공지"),
    ("ee.kookmin.ac.kr/community/board/ee_recruit", "전자공학부 취업정보"),
    ("ee.kookmin.ac.kr/community/board/contest", "전자공학부 경진대회"),
    ("ee.kookmin.ac.kr/community/board/job_reviews", "전자공학부 취업후기"),
    ("ee.kookmin.ac.kr/community/board/ee_news", "전자공학부 소식"),
    ("ee.kookmin.ac.kr/community/kookmin/academic", "국민대 학사공지"),
    ("ee.kookmin.ac.kr/community/kookmin/administration", "국민대 행정공지"),
    ("ee.kookmin.ac.kr/community/kookmin/special_lecture", "국민대 특강공지"),
    ("ee.kookmin.ac.kr/community/kookmin/scholarship", "국민대 장학공지"),
    ("ee.kookmin.ac.kr/community/kookmin/social_service", "국민대 사회봉사"),
    ("ee.kookmin.ac.kr/community/kookmin/recruit", "국민대 교내채용"),
    ("ee.kookmin.ac.kr/community/kookmin/suburb", "국민대 교외채용"),
    ("kookmin.ac.kr/user/kmuNews/notice/4/", "국민대 학사공지"),
    ("kookmin.ac.kr/user/kmuNews/notice/5/", "국민대 행정공지"),
    ("kookmin.ac.kr/user/kmuNews/notice/6/", "국민대 특강공지"),
    ("kookmin.ac.kr/user/kmuNews/notice/7/", "국민대 장학공지"),
    ("kookmin.ac.kr/user/kmuNews/notice/8/", "국민대 사회봉사"),
    ("kookmin.ac.kr/user/kmuNews/notice/9/", "국민대 공모·행사"),
    ("kookmin.ac.kr/user/kmuNews/notice/10/", "국민대 교내채용"),
    ("kookmin.ac.kr/user/kmuNews/notice/11/", "국민대 교외채용"),
    ("kookmin.ac.kr/user/kmuNews/notice/index.do", "국민대 공지사항"),
]


def guess_board_name(url: str) -> str:
    """링크만 보고 어떤 게시판인지 이름을 정해요.
    ① 알고 있는 학교 게시판이면 바로 이름을 붙이고
    ② 처음 보는 곳이면 페이지에 들어가 제목(<title>)을 읽어서 정해요."""
    import re
    from urllib.parse import urlparse

    for key, name in KNOWN_BOARDS:
        if key in url:
            return name
    host = urlparse(url).netloc.replace("www.", "")
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (student project)"})
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        og = soup.find("meta", attrs={"property": "og:title"})
        title = (soup.title.get_text() if soup.title else "") or (og.get("content", "") if og else "")
        parts = [p.strip() for p in re.split(r"\s*[|\-–—:·>»]\s*", " ".join(title.split())) if p.strip()]
        parts = list(dict.fromkeys(parts))  # 중복 제거
        if not parts:
            return host
        # '국민대학교', '○○학부', '○○센터' 같은 기관 이름을 앞에, 게시판 이름을 뒤에
        org = [p for p in parts if re.search(r"(대학교|대학|학부|학과|센터|재단|공사|협회)$", p)]
        rest = [p for p in parts if p not in org]
        name = " ".join((org[:1] + rest[:1]) or parts[:2])
        return name[:30]
    except Exception:
        return host


@st.cache_data(show_spinner="게시판 이름 알아보는 중...", ttl=86400)
def board_name(url):
    return guess_board_name(url)


# ── 처음 접속할 때 뜨는 프로필 입력 창 ─────────
GRADES = ["1학년", "2학년", "3학년", "4학년", "대학원"]
INTEREST_OPTS = ["취업", "장학", "자격증", "대외활동", "연구"]
PRESETS = {
    "직접 입력": None,
    "예시: 전자공학부 4학년 (취업·자격증)": ("전자공학부", "4학년", ["취업", "자격증"], "공기업(전력·에너지) 취업 희망, 2027년 2월 졸업예정"),
    "예시: 경영학과 2학년 (장학)": ("경영학과", "2학년", ["장학"], ""),
}


def load_profile_from_url():
    """새로고침해도 다시 묻지 않도록, 주소창(?major=...)에 저장해 둔 프로필을 읽어요."""
    q = st.query_params
    if "p_major" not in st.session_state and q.get("major"):
        st.session_state["p_major"] = q.get("major", "")
        st.session_state["p_grade"] = q.get("grade", "1학년") if q.get("grade") in GRADES else "1학년"
        st.session_state["p_interests"] = [x for x in q.get("int", "").split(",") if x in INTEREST_OPTS]
        st.session_state["p_extra"] = q.get("extra", "")


def apply_preset():
    v = PRESETS.get(st.session_state.get("f_preset"))
    if v:
        st.session_state.f_major, st.session_state.f_grade, st.session_state.f_interests, st.session_state.f_extra = v[0], v[1], list(v[2]), v[3]


def profile_form():
    st.caption("입력한 학과·학년·관심사에 맞춰 공지를 골라드려요. 나중에 사이드바에서 언제든 바꿀 수 있어요.")
    st.session_state.setdefault("f_major", st.session_state.get("p_major", ""))
    st.session_state.setdefault("f_grade", st.session_state.get("p_grade", "1학년"))
    st.session_state.setdefault("f_interests", st.session_state.get("p_interests", []))
    st.session_state.setdefault("f_extra", st.session_state.get("p_extra", ""))
    st.selectbox("빠르게 채우기", list(PRESETS), key="f_preset", on_change=apply_preset)
    st.text_input("학과 *", key="f_major", placeholder="예: 전자공학부, 경영학과")
    st.selectbox("학년 *", GRADES, key="f_grade")
    st.multiselect("관심사 * (여러 개 가능)", INTEREST_OPTS, key="f_interests")
    st.text_area("추가 정보 (선택)", key="f_extra", height=80,
                 placeholder="예: 공기업 취업 희망, 2027년 2월 졸업예정, 전기기사 준비 중")
    if st.button("시작하기 →", type="primary", use_container_width=True):
        m, it = st.session_state.f_major.strip(), st.session_state.f_interests
        if not m or not it:
            st.error("학과와 관심사(1개 이상)를 입력해 주세요.")
            return
        st.session_state.update(p_major=m, p_grade=st.session_state.f_grade, p_interests=list(it),
                                p_extra=st.session_state.f_extra.strip(), profile_open=False)
        st.query_params.update(major=m, grade=st.session_state.f_grade, int=",".join(it),
                               extra=st.session_state.f_extra.strip()[:80])
        st.rerun()


if hasattr(st, "dialog"):
    profile_dialog = st.dialog("👋 내 프로필을 알려주세요")(profile_form)
else:  # 오래된 Streamlit이면 창 대신 화면에 바로 표시
    def profile_dialog():
        st.subheader("👋 내 프로필을 알려주세요")
        profile_form()


def open_profile():
    for k in ("f_major", "f_grade", "f_interests", "f_extra", "f_preset"):
        st.session_state.pop(k, None)
    st.session_state["profile_open"] = True


def server_key():
    """컴퓨터(setx GEMINI_API_KEY) 또는 Streamlit Cloud의 Secrets에 저장한 키. 화면으로는 보내지 않아요."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


KMU = ("https://ee.kookmin.ac.kr/community/board/notice/\n"
       "https://ee.kookmin.ac.kr/community/board/ee_recruit/\n"
       "https://ee.kookmin.ac.kr/community/board/contest/\n"
       "https://www.kookmin.ac.kr/user/kmuNews/notice/7/index.do\n"
       "https://www.kookmin.ac.kr/user/kmuNews/notice/6/index.do\n"
       "https://www.kookmin.ac.kr/user/kmuNews/notice/11/index.do")

# ── 관련 게시판 추천 ───────────────────────────
BOARD_HINT = ("공지", "채용", "장학", "특강", "공모", "행사", "취업", "모집", "게시판", "소식", "안내",
              "현장실습", "대회", "인턴", "교육", "프로그램", "학사", "봉사", "연구")
INTEREST_WORDS = {
    "취업": ("취업", "채용", "현장실습", "인턴", "recruit", "job", "career", "suburb"),
    "장학": ("장학", "scholar", "근로"),
    "자격증": ("자격", "특강", "교육", "lecture", "special"),
    "대외활동": ("공모", "대회", "행사", "contest", "봉사", "social"),
    "연구": ("연구", "lab", "research", "news"),
}


# 국민대 학과·단과대 사이트에 공통으로 있는 게시판 (전자공학부·경영대학 등 같은 틀)
KMU_SITE_BOARDS = [
    ("/community/notice/", "공지사항"), ("/community/board/notice/", "학부공지"),
    ("/community/kookmin/academic/", "국민대 학사공지"), ("/community/kookmin/scholarship/", "국민대 장학공지"),
    ("/community/kookmin/special_lecture/", "국민대 특강공지"), ("/community/kookmin/recruit/", "국민대 교내채용"),
    ("/community/kookmin/suburb", "국민대 교외채용"), ("/community/kookmin/social_service/", "국민대 사회봉사"),
    ("/community/kookmin/administration/", "국민대 행정공지"),
]


def name_for(url, text=""):
    """게시판 링크에 붙일 이름: 알고 있는 게시판 → 공통 게시판 → 링크 글자"""
    from urllib.parse import urlparse
    for k, n in KNOWN_BOARDS:
        if k in url:
            return n
    path = urlparse(url).path
    for k, n in KMU_SITE_BOARDS:
        if path.rstrip("/") == k.rstrip("/"):
            return n if n.startswith("국민대") else f"{board_name(url.split('/community')[0] + '/')} {n}"
    return text or urlparse(url).netloc


@st.cache_data(show_spinner=False, ttl=86400)
def probe_kmu_boards(url):
    """국민대 학과 사이트면 공통 게시판 주소가 실제로 열리는지 확인해서 후보로 돌려줘요."""
    from urllib.parse import urlparse

    import requests
    u = urlparse(url)
    if not u.netloc.endswith("kookmin.ac.kr") or u.netloc in ("kookmin.ac.kr", "www.kookmin.ac.kr"):
        return []
    base = f"{u.scheme}://{u.netloc}"
    out = []
    for path, _ in KMU_SITE_BOARDS:
        try:
            r = requests.get(base + path, timeout=6, headers={"User-Agent": "Mozilla/5.0 (student project)"})
            if r.status_code == 200 and len(r.text) > 2000:
                out.append({"url": base + path, "text": ""})
        except Exception:
            pass
    return out


@st.cache_data(show_spinner=False, ttl=86400)
def find_related_boards(url):
    """게시판 페이지의 메뉴에서 다른 게시판 링크를 모아요 (공지 글 하나짜리 링크는 빼고)."""
    import re
    from urllib.parse import urljoin, urlparse

    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (student project)"})
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    site = ".".join(urlparse(url).netloc.split(".")[-3:])   # 예: kookmin.ac.kr
    found = {}
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text().split())
        href = urljoin(url, a["href"]).split("#")[0]
        u = urlparse(href)
        if not (2 <= len(text) <= 20) or not href.startswith("http") or not u.netloc.endswith(site):
            continue
        path = u.path.lower()
        if re.search(r"(/view|view\.do)", path) or re.search(r"/\d{3,}(/|$|\.)", path) or re.search(r"\.(pdf|hwp|hwpx|docx?|xlsx?|zip|jpg|png)$", path):
            continue   # 공지 글 하나, 첨부파일은 제외
        if any(w in path for w in ("login", "logout", "javascript")):
            continue
        looks_board = any(w in path for w in ("board", "notice", "bbs", "news", "community", "recruit", "scholar")) \
            or any(w in text for w in BOARD_HINT)
        if looks_board and href not in found:
            found[href] = text
    return [{"url": h, "text": t} for h, t in list(found.items())[:60]]


# 국민대 사이트를 넣으면 항상 같이 후보에 넣는 본교 게시판 (전자공학부 기본 조합과 같은 방식)
KMU_MAIN_BOARDS = [
    "https://www.kookmin.ac.kr/user/kmuNews/notice/7/index.do",   # 장학공지
    "https://www.kookmin.ac.kr/user/kmuNews/notice/6/index.do",   # 특강공지
    "https://www.kookmin.ac.kr/user/kmuNews/notice/11/index.do",  # 교외채용
    "https://www.kookmin.ac.kr/user/kmuNews/notice/4/index.do",   # 학사공지
    "https://www.kookmin.ac.kr/user/kmuNews/notice/9/index.do",   # 공모·행사
]
# 학생에게 쓸모 있는 게시판인지 판단하는 단어 (점수용)
USEFUL_WORDS = ("공지", "취업", "채용", "장학", "특강", "행사", "공모", "대회", "현장실습", "인턴", "모집", "학사",
                "recruit", "job", "career", "contest", "scholar", "notice", "lecture", "suburb")
NOT_BOARD = ("소개", "인사말", "오시는", "찾아오", "교수", "연혁", "조직", "갤러리", "사진", "동영상", "사이트맵",
             "개인정보", "로그인", "faq", "자료실", "서식")


def site_label(url):
    """'국민대학교 경영대학' → '경영대학' 처럼 게시판 이름 앞에 붙일 학과 이름"""
    root = url.split("/community")[0].split("/user/")[0].rstrip("/") + "/"
    return board_name(root).replace("국민대학교", "").strip() or "학과"


def rule_score(c, interest_words):
    text = (c["name"] + " " + c["url"]).lower()
    if any(w in text for w in NOT_BOARD):
        return -1
    return 3 * sum(w in text for w in interest_words) + sum(w in text for w in USEFUL_WORDS)


def rule_reason(c, interests):
    text = (c["name"] + " " + c["url"]).lower()
    for it in interests:
        if any(w in text for w in INTEREST_WORDS.get(it, ())):
            return f"{it} 관심사와 관련된 게시판"
    if "kmunews" in text or c["name"].startswith("국민대"):
        return "국민대 전체 학생 대상 공지"
    return "학과 학생에게 필요한 공지가 올라오는 곳"


def prepare_candidates(current, candidates):
    """찾은 링크에 이름을 붙이고 중복을 없앤 '전체 후보' 목록 (국민대면 본교 게시판도 포함)"""
    from urllib.parse import urlparse
    cands = [dict(c) for c in candidates]
    if any(urlparse(u).netloc.endswith("kookmin.ac.kr") for u in current.values()):
        cands += [{"url": u, "text": ""} for u in KMU_MAIN_BOARDS]
    cands = [c for c in cands if c["url"] not in current.values()]
    for c in cands:
        n = name_for(c["url"], c["text"])
        if not n.startswith(("국민대", "전자공학부")) and "kookmin.ac.kr" in c["url"] and "www.kookmin" not in c["url"]:
            lab = site_label(c["url"])
            n = n if n.startswith(lab) else f"{lab} {n}"
        c["name"] = n
    uniq = {}   # 같은 게시판(이름이 같은 것)은 하나만
    for c in cands:
        if c["name"] not in current and c["name"] not in uniq:
            uniq[c["name"]] = c
    return list(uniq.values())


def recommend_boards(api_key, profile, current, candidates, top=5, at_least=3):
    """넣은 학과 사이트 기준으로, 그 학과 학생에게 유용한 게시판을 골라요.
    국민대 사이트면 본교 장학·특강·채용 게시판도 후보에 넣고, 최소 at_least개는 추천해요."""
    cands = [c for c in prepare_candidates(current, candidates) if rule_score(c, ()) >= 0]
    if not cands:
        return []
    sites = ", ".join(sorted({site_label(u) for u in current.values() if "kookmin.ac.kr" in u})) or "넣은 사이트"
    out = []
    if api_key:
        try:
            from google.genai import types
            listing = "\n".join(f"{i}. {c['name']} ({c['url']})" for i, c in enumerate(cands, 1))
            prompt = f"""학생이 학교 공지를 놓치지 않도록 같이 보면 좋은 게시판을 골라줘.
## 사용자가 넣은 게시판 (= 사용자가 다니는 학과·단과대 사이트: {sites})
{chr(10).join(f"- {n}" for n in current)}
## 사용자 정보 (참고용, 넣은 사이트와 학과가 달라도 넣은 사이트를 기준으로 판단해)
{profile_text(profile)}
## 후보 (같은 학교 사이트에서 찾은 게시판 링크)
{listing}
## 규칙
- 공지가 꾸준히 올라오는 '게시판'만 골라. 소개·교수진·자료실 같은 일반 페이지는 빼.
- 넣은 학과 사이트의 주요 게시판(공지, 취업, 행사, 경진대회 등)과 국민대 본교 게시판(장학, 특강, 채용)을 골고루.
- 관심사({', '.join(profile['interests']) or '없음'})와 관련된 것을 앞에. 최소 {at_least}개, 최대 {top}개.
- reason은 왜 도움이 되는지 25자 이내 한 문장.
## 출력: JSON 배열만
[{{"no": 후보번호, "reason": ""}}]"""
            config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            client, picks = get_client(api_key), None
            for m in sorted(model_list(client), key=lambda m: m != st.session_state.get("model")):  # 한도에 걸려도 기다리지 않고 다음으로
                try:
                    picks = json.loads(client.models.generate_content(model=m, contents=prompt, config=config).text)
                    st.session_state["model"] = m
                    break
                except Exception:
                    continue
            if isinstance(picks, dict):  # {"추천": [...]} 같은 모양으로 와도 처리
                picks = next((v for v in picks.values() if isinstance(v, list)), [])
            for pk in (picks or [])[:top]:
                try:
                    i = int(pk.get("no", 0)) - 1
                except (TypeError, ValueError, AttributeError):
                    continue
                if 0 <= i < len(cands) and all(o["url"] != cands[i]["url"] for o in out):
                    out.append(dict(cands[i], reason=pk.get("reason", ""), by="AI"))
        except Exception:
            pass  # 한도 초과 등 → 아래 규칙으로 채움
    # AI가 못 골랐거나 적게 골랐으면 규칙으로 채우기 (관심사 단어 > 공지·취업·장학 같은 쓸모 있는 단어)
    if len(out) < at_least:
        words = [w for it in profile["interests"] for w in INTEREST_WORDS.get(it, ())]
        ranked = sorted((c for c in cands if all(o["url"] != c["url"] for o in out)),
                        key=lambda c: -rule_score(c, words))
        target = at_least if out else top   # AI가 하나도 못 골랐으면 규칙으로 최대 top개까지
        for c in ranked:
            if len(out) >= target or rule_score(c, words) <= 0:
                break
            out.append(dict(c, reason=rule_reason(c, profile["interests"]), by="규칙"))
    return out[:top]


def add_board(*urls):
    cur = st.session_state.get("board_text", "").rstrip()
    have = set(cur.split())
    new = [u for u in urls if u not in have]
    st.session_state["board_text"] = (cur + "\n" + "\n".join(new)).strip()
    for key in ("board_suggest", "board_all"):
        st.session_state[key] = [c for c in st.session_state.get(key, []) if c["url"] not in urls]


def toggle_kmu():
    st.session_state["board_text"] = KMU if st.session_state.get("use_kmu") else ""
    st.session_state.pop("board_suggest", None)
    st.session_state.pop("board_note", None)
    st.session_state.pop("board_all", None)


def parse_board_lines(text):
    """한 줄에 하나씩 적은 게시판 링크를 읽어요.
    '링크만' / '이름 | 링크' / '이름 링크' 모두 가능. 이름이 없으면 링크를 보고 알아서 정해요."""
    import re
    boards = {}
    for line in text.splitlines():
        m = re.search(r"https?://\S+", line)
        if not m:
            continue
        url = m.group(0).rstrip(",;)")
        name = (line[:m.start()] + line[m.end():]).replace("|", " ").strip()
        name = " ".join(name.split()) or board_name(url)
        base, i = name, 2
        while name in boards:          # 이름이 겹치면 (2), (3)…
            name = f"{base} ({i})"; i += 1
        boards[name] = url
    return boards


# ──────────────────────────────────────────────
# 화면
# ──────────────────────────────────────────────
st.set_page_config(page_title="공지콕", page_icon="📢", layout="wide")
st.markdown("""<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@900&display=swap');
  /* 위쪽 Streamlit 메뉴 줄(Deploy)에 제목이 가려지지 않게 충분히 띄우기 */
  .block-container, [data-testid="stMainBlockContainer"] { padding-top: 5.5rem !important; max-width: 1400px; }
  header[data-testid="stHeader"] { background: #F4F5F8; }
  .app-title { font-size: 4.6rem; font-weight: 900; color: #15213B; margin: 0; line-height: 1.25;
    font-family: "Noto Sans KR", sans-serif; -webkit-text-stroke: 2.5px #15213B; letter-spacing: 1px; }
  .app-title b { font-weight: 900; }
  .app-sub { color: #6B7280; font-size: 1.05rem; margin: .2rem 0 0; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("👤 내 프로필")
    load_profile_from_url()
    major = st.session_state.get("p_major", "")
    grade = st.session_state.get("p_grade", "1학년")
    interests = st.session_state.get("p_interests", [])
    extra = st.session_state.get("p_extra", "")
    if major:
        st.markdown(f"**{major} {grade}**  \n관심사: {', '.join(interests) or '-'}"
                    + (f"  \n<span style='font-size:13px;color:#6B7280'>{extra}</span>" if extra else ""),
                    unsafe_allow_html=True)
    else:
        st.caption("아직 프로필을 입력하지 않았어요.")
    st.button("✏️ 프로필 수정" if major else "프로필 입력하기", use_container_width=True, on_click=open_profile)
    today = st.date_input("기준 날짜", date.today())
    typed_key = st.text_input("Gemini API 키", key="api_key", type="password",
                              help="저장해 둔 키가 있으면 비워둬도 돼요. 입력한 키는 이 브라우저 창에서만 쓰여요.")
    api_key = typed_key or server_key()
    if not typed_key and api_key:
        st.caption("🔒 저장된 키를 사용 중이에요 (화면에는 보이지 않아요)")

    st.divider()
    st.header("🌐 게시판")
    st.session_state.setdefault("board_text", "")
    st.toggle("국민대 전자공학부 게시판", key="use_kmu", on_change=toggle_kmu)
    board_lines = st.text_area(
        "게시판 링크", key="board_text",
        placeholder="https://ee.kookmin.ac.kr/community/board/notice/\nhttps://...", height=130)
    boards = parse_board_lines(board_lines)
    if boards:
        st.caption("인식한 게시판\n" + "\n".join(f"- ✅ **{n}**" for n in boards))
        if st.button("🔎 관련 게시판 추천받기", use_container_width=True,
                     help="붙여넣은 게시판 사이트의 메뉴에서 다른 게시판을 찾아, 내 학과·관심사와 가까운 곳을 골라줘요."):
            with st.spinner("같은 사이트의 다른 게시판을 살펴보는 중..."):
                cands, notes = [], []
                for u in boards.values():
                    try:
                        found = find_related_boards(u)
                    except Exception as e:
                        found = []
                        notes.append(f"⚠️ {u} 을(를) 열지 못했어요 ({type(e).__name__})")
                    if len(found) < 3:  # 메뉴에서 거의 못 찾으면 학과 사이트 공통 게시판을 직접 확인
                        found += probe_kmu_boards(u)
                    cands += found
                st.session_state["board_suggest"] = recommend_boards(
                    st.session_state.get("api_key", "") or server_key(),
                    {"major": major, "grade": grade, "interests": interests, "extra": extra}, boards, cands)
                st.session_state["board_all"] = prepare_candidates(boards, cands)
                st.session_state["board_note"] = notes + [
                    f"다른 게시판 후보 {len(st.session_state['board_all'])}개 중 "
                    + (f"{len(st.session_state['board_suggest'])}개를 골랐어요." if st.session_state["board_suggest"]
                       else "추천할 만한 곳을 찾지 못했어요. 메인 페이지보다 게시판 링크를 넣으면 더 잘 찾아요.")]
        for n in st.session_state.get("board_note", []):
            st.caption(n)
        sug = st.session_state.get("board_suggest", [])
        if sug:
            st.markdown("**⭐ 추천 게시판**")
        for c in sug:
            col1, col2 = st.columns([4, 1.3], vertical_alignment="center")
            col1.markdown(f"**{c['name']}**  \n<span style='font-size:12.5px;color:#6B7280'>{c['reason']}</span>",
                          unsafe_allow_html=True)
            col2.button("➕ 추가", key="add_" + c["url"], on_click=add_board, args=(c["url"],))
        rest = [c for c in st.session_state.get("board_all", []) if all(c["url"] != x["url"] for x in sug)]
        if rest:
            with st.expander(f"찾은 게시판 전체 보기 ({len(rest)}개 더)"):
                st.button(f"➕ {len(rest)}개 전체 추가", key="add_all", use_container_width=True,
                          on_click=add_board, args=tuple(c["url"] for c in rest),
                          help="게시판이 많을수록 'AI로 공지 판단하기'가 오래 걸려요 (게시판마다 공지 4개씩 읽어요).")
                for c in rest:
                    col1, col2 = st.columns([4, 1.3], vertical_alignment="center")
                    col1.markdown(f"{c['name']}  \n<span style='font-size:11.5px;color:#9AA3B5'>{c['url']}</span>",
                                  unsafe_allow_html=True)
                    col2.button("➕", key="addall_" + c["url"], on_click=add_board, args=(c["url"],))


profile = {"major": major, "grade": grade, "interests": interests, "extra": extra}
if not major or st.session_state.get("profile_open"):
    profile_dialog()
    st.session_state["profile_open"] = False  # 창을 X로 닫아도 계속 뜨지 않게 (프로필이 없으면 다시 떠요)

# 게시판 설정 (사이드바에서 읽은 링크)
real_boards = bool(boards)
if not boards:
    for n in load_sample_notices():
        boards.setdefault(n["board"], "sample")


def ready():
    if not api_key:
        st.error("사이드바에 Gemini API 키를 넣어주세요.")
        return False
    if not major or not interests:
        st.error("먼저 사이드바의 '프로필 입력하기'로 학과와 관심사를 입력해 주세요.")
        return False
    return True


# ── 머리말 + 화면 고르기 ───────────────────────
# Agent 화면을 다시 보이게 하려면 SHOW_AGENT_PAGE = True 로 바꾸세요
SHOW_AGENT_PAGE = False
PAGES = (["Agent에게 맡기기"] if SHOW_AGENT_PAGE else []) + ["홈", "캘린더", "기억", "숨김"]
h_left, h_right = st.columns([3, 2], vertical_alignment="center")
with h_left:
    st.markdown(f'<p class="app-title"><b>공지콕</b></p>'
                f'<p class="app-sub">{major or "학과 미입력"} {grade} · 기준일 {today.month}/{today.day}</p>',
                unsafe_allow_html=True)
with h_right:
    try:
        page = st.segmented_control("화면", PAGES, default="홈", key="page",
                                    label_visibility="collapsed")
    except AttributeError:  # Streamlit이 오래된 버전이면 일반 선택 버튼으로
        page = st.radio("화면", PAGES, index=PAGES.index("홈"), horizontal=True, key="page", label_visibility="collapsed")
page = page or "홈"
st.write("")

# 이 사람의 결과를 찾는 열쇠 (프로필 + 게시판 또는 공지 내용)
st.session_state.setdefault("notice_text", open("notices.txt", encoding="utf-8").read() if os.path.exists("notices.txt") else "")
MY_KEY = batch_key(profile, boards if real_boards else {}, st.session_state["notice_text"])

# ── 홈 ────────────────────────────────
if page == "홈":
    if real_boards:
        source_note = ""
        notices = None
    else:
        with st.expander("공지 내용 보기 / 직접 붙여넣기 (단톡방 공지도 여기에)"):
            notice_text = st.text_area("공지는 --- 로 구분", key="notice_text", height=220)
        notices = parse_notices(notice_text)
        source_note = f"예시 공지 {len(notices)}개를 판단해요."

    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    batch = load_batch(MY_KEY)
    run = c1.button("🔄 다시 판단하기" if batch else "🤖 AI로 공지 판단하기", type="primary", use_container_width=True,
                    help="같은 프로필·게시판으로 이미 판단한 결과가 있으면 바로 보여줘요. 새 공지를 반영하려면 다시 판단하세요.")
    c2.caption("  ·  ".join(x for x in (source_note, f"마지막 판단: {batch['saved_at']}" if batch else "아직 판단한 결과가 없어요.") if x))

    if run and ready():
        import time as _time
        t0 = _time.time()
        with st.status("AI가 일하는 중...", expanded=True) as status:
            bar = st.progress(0, text="0%")
            step = lambda pct, txt: bar.progress(min(100, int(pct)) / 100, text=f"**{int(pct)}%** · {txt}")
            try:
                if notices is None:
                    # 실제 게시판: 게시판 목록은 동시에 바로 읽고(몇 초), 판단은 Gemini가 해요
                    if len(boards) > MAX_LINKS:
                        st.write(f"⚠️ 게시판이 많아서 앞의 {MAX_LINKS}곳만 읽어요.")
                    n_b = min(len(boards), MAX_LINKS)
                    st.write("⏱️ 예상 시간 약 30~70초 (공지가 많을수록 조금 더 걸려요)")
                    step(3, f"🔍 게시판 {n_b}곳을 동시에 열어 최신 공지를 찾는 중...")
                    posts, missed = quick_find_posts(dict(list(boards.items())[:MAX_LINKS]), today)
                    if missed:   # 바로 못 읽은 게시판만 Gemini가 직접 열어 읽어요
                        step(15, f"🔍 바로 못 읽은 게시판 {len(missed)}곳은 Gemini가 직접 열어 읽는 중...")
                        try:
                            extra, bad = ai_find_posts(api_key, missed, today, log=st.caption)
                        except RuntimeError as e:
                            if not posts:
                                raise
                            extra, bad = [], list(missed)
                        if bad:
                            st.write("⚠️ 열지 못한 게시판: " + ", ".join(bad))
                        posts += extra
                        for i, p in enumerate(posts):
                            p["id"] = str(i + 1)
                    posts, _ = drop_old_posts(posts, today)
                    posts = posts[:MAX_JUDGE]
                    for i, p in enumerate(posts):
                        p["id"] = str(i + 1)
                    if not posts:
                        raise RuntimeError("게시판에서 공지를 찾지 못했어요. 게시판 링크가 맞는지 확인하거나, "
                                           "사이드바 스위치를 꺼서 예시 공지로 시연해 주세요.")
                    step(25, f"🤖 최신 공지 {len(posts)}개를 찾았어요. Gemini가 판단하는 중...")
                    results = ai_judge_titles(api_key, profile, posts, today, log=st.caption,
                                              progress=lambda d, t: step(25 + 55 * d / t,
                                                                         f"🤖 Gemini가 공지 {d}/{t}개 판단 완료..."))
                    need = [r for r in results if r.get("relevant")
                            and str(r.get("deadline") or "").strip().lower() in ("", "null", "none", "없음")]
                    if need:
                        step(82, f"📄 마감일이 안 보이는 추천 공지 {min(len(need), MAX_FILL)}개의 본문을 확인하는 중...")
                        try:
                            ai_fill_details(api_key, profile, results, today, log=st.caption)
                        except Exception:
                            st.caption("⏭️ 본문 읽기가 오래 걸려서 건너뛰고, 제목으로 판단한 결과를 보여줄게요.")
                else:
                    st.write("⏱️ 예상 시간 약 5~15초")
                    step(30, f"🤖 Gemini가 공지 {len(notices)}개를 판단하는 중...")
                    results = run_batch(api_key, profile, notices, today)
                if not results:
                    raise RuntimeError("AI 답을 읽지 못했어요. 잠시 뒤 다시 눌러 주세요.")
                from datetime import datetime
                batch = {"saved_at": datetime.now().strftime("%m/%d %H:%M"), "results": results,
                         "profile": profile, "today": today.isoformat(),
                         "source": f"실제 게시판 {len(boards)}곳" if real_boards else "예시 공지"}
                save_batch(MY_KEY, batch)
                step(100, "완료!")
                status.update(label=f"완료! 공지 {len(posts) if notices is None else len(notices)}개를 읽고 {sum(1 for r in results if r.get('relevant'))}개를 추천했어요 ({_time.time() - t0:.0f}초) · 나머지는 숨김 창에",
                              state="complete", expanded=False)
            except Exception as e:
                status.update(label="오류", state="error")
                st.error(str(e))

    if batch and batch.get("results"):
        show_dashboard(batch["results"], today, interests, all_tags=str(batch.get("source", "")).startswith("실제"))
        st.caption("☆ 중요 표시, 관심 없음, 관심 키워드, 제외 키워드 설정은 이 브라우저에 저장돼요. 숨긴 공지는 위쪽 '숨김'에서 볼 수 있어요.")
    else:
        st.info("위의 **🤖 AI로 공지 판단하기**를 누르면 마감 타임라인과 맞춤 공지 카드가 여기에 나타나요.")

# ── 숨김 (필터로 숨긴 공지 모아보기) ───────────
elif page == "숨김":
    batch = load_batch(MY_KEY)
    if batch and batch.get("results"):
        show_dashboard(batch["results"], today, interests, height=1000, mode="hidden", all_tags=str(batch.get("source", "")).startswith("실제"))
    else:
        st.info("먼저 **홈**에서 🤖 AI로 공지 판단하기를 눌러 주세요.")

# ── 캘린더 (캘린더만 있는 화면) ───────────
elif page == "캘린더":
    batch = load_batch(MY_KEY)
    if batch and batch.get("results"):
        show_dashboard(batch["results"], today, interests, height=1080, mode="calendar", all_tags=str(batch.get("source", "")).startswith("실제"))
    else:
        st.info("먼저 **홈**에서 🤖 AI로 공지 판단하기를 눌러 주세요. 판단한 공지의 마감일이 여기 캘린더에 표시돼요.")

# ── Agent에게 맡기기 ───────────────────────────
elif page == "Agent에게 맡기기":
    st.write(f"확인할 게시판: **{', '.join(boards)}**")
    goal = st.text_input("Agent에게 시킬 일", "이번 주에 내가 놓치면 안 되는 공지 정리해줘")
    c1, c2 = st.columns(2)
    go = c1.button("🤖 Agent 실행", type="primary", use_container_width=True)
    morning = c2.button("🌅 아침 브리핑 (자동 실행)", use_container_width=True,
                        help="매일 아침 자동으로 돌아가는 상황을 시연해요. 아직 안 읽은 새 공지만 확인합니다.")
    if morning:
        goal = ("아침 브리핑이야. 각 게시판에서 new=true인 새 공지만 확인해서 나에게 해당하는 것만 알려줘. "
                "새 공지가 없거나 해당되는 게 없으면 짧게 그렇다고 말해.")

    if (go or morning) and ready():
        with st.status("Agent가 일하는 중...", expanded=True) as status:
            try:
                out = run_agent(api_key, profile, boards, today, goal, st.markdown)
                status.update(label=f"완료! 도구를 {len([s for s in out['steps'] if not s[0].startswith('💭')])}번 사용했어요",
                              state="complete", expanded=False)
                st.session_state["agent_out"] = out
                json.dump({"agent_out": out}, open(CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False)
            except Exception as e:
                status.update(label="오류", state="error")
                st.error(str(e))

    if st.button("📂 지난 결과 불러오기", help="시연 중 인터넷/API 문제가 생기면 사용"):
        try:
            st.session_state["agent_out"] = json.load(open(CACHE_FILE, encoding="utf-8"))["agent_out"]
        except Exception:
            st.warning("저장된 결과가 없어요. 먼저 한 번 실행해 주세요.")

    out = st.session_state.get("agent_out")
    if out:
        with st.expander(f"🔎 Agent가 일한 과정 ({len(out['steps'])}단계)"):
            for i, (label, detail) in enumerate(out["steps"], 1):
                st.markdown(f"{i}. **{label}** — {detail}")
        if out.get("new"):
            st.success(f"🆕 이번에 새로 읽은 공지 {len(out['new'])}개")
        st.markdown(out["answer"])
        if out["calendar"]:
            st.subheader("📅 Agent가 등록한 마감일")
            for c in sorted(out["calendar"], key=lambda c: c["deadline"]):
                st.markdown(f"- {d_day_badge(d_day(c['deadline'], today))} **{c['title']}** ({c['deadline']}) {c.get('memo', '')}")
            st.download_button("내 캘린더에 추가 (.ics, D-3·D-1 알림)", make_ics(out["calendar"]),
                               file_name="공지_마감일.ics", mime="text/calendar")

        fb = st.text_input("💬 Agent에게 피드백 (기억해 둬요)", placeholder="예: 대외활동 공지는 관심 없어 / 공기업 관련은 꼭 알려줘")
        if st.button("기억시키기") and fb:
            remember_note(fb)
            st.success("기억했어요! 다음 추천부터 반영돼요.")

# ── 기억 (이 브라우저에 저장된 내 설정) ─────────
else:
    batch = load_batch(MY_KEY)
    show_dashboard((batch or {}).get("results", []), today, interests, height=900, mode="memory", all_tags=str((batch or {}).get("source", "")).startswith("실제"))
