"""
Agent가 쓰는 '도구'들과 '기억' 저장소.
AI는 이 도구들 중 무엇을, 어떤 순서로 쓸지 스스로 정합니다.
(화면 코드와 분리해 두어서, 이 파일만 따로 테스트할 수 있어요.)
"""

import json
import os
from datetime import date, datetime
from urllib.parse import urljoin

NOTICE_FILE = "notices.txt"
MEMORY_FILE = "memory.json"


# ──────────────────────────────────────────────
# 공지 데이터
# ──────────────────────────────────────────────
def parse_notices(text: str) -> list[dict]:
    """'---'로 구분된 공지 텍스트를 목록으로 바꿉니다."""
    notices = []
    for block in text.split("---"):
        block = block.strip()
        if not block:
            continue
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields.setdefault(k.strip(), v.strip())
        notices.append({
            "id": str(len(notices) + 1),
            "title": fields.get("제목", block.splitlines()[0]),
            "board": fields.get("게시판", "학교 공지"),
            "date": fields.get("게시일", ""),
            "text": block,
        })
    return notices


def load_sample_notices() -> list[dict]:
    if not os.path.exists(NOTICE_FILE):
        return []
    return parse_notices(open(NOTICE_FILE, encoding="utf-8").read())


# ──────────────────────────────────────────────
# 기억 (사용자 피드백, 이미 본 공지)
# ──────────────────────────────────────────────
def load_memory() -> dict:
    try:
        return json.load(open(MEMORY_FILE, encoding="utf-8"))
    except Exception:
        return {"likes": [], "dislikes": [], "notes": [], "seen": []}


def save_memory(mem: dict) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


def remember_feedback(title: str, liked: bool, why: str = "") -> None:
    mem = load_memory()
    key = "likes" if liked else "dislikes"
    item = f"{title}" + (f" ({why})" if why else "")
    if item not in mem[key]:
        mem[key].append(item)
    save_memory(mem)


def remember_note(note: str) -> None:
    mem = load_memory()
    if note and note not in mem["notes"]:
        mem["notes"].append(note)
    save_memory(mem)


# ──────────────────────────────────────────────
# 웹페이지 읽기 (실제 학교 게시판용)
# ──────────────────────────────────────────────
def fetch_page(url: str) -> tuple[str, list[dict]]:
    import requests
    from bs4 import BeautifulSoup

    import time
    time.sleep(0.5)  # 학교 서버에 부담을 주지 않도록 잠깐 쉬기
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (student project)"})
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    links = []
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text().split())
        href = urljoin(url, a["href"])
        if len(text) >= 8 and href.startswith("http") and not href.endswith("#"):
            links.append({"title": text[:80], "url": href})
    text = " ".join(soup.get_text(" ").split())
    return text, links


def list_board_posts(src: str, limit: int = 20) -> list[dict]:
    """게시판 목록 페이지에서 공지 글 링크만 골라냅니다 (메뉴 링크는 제외)."""
    import re
    from urllib.parse import urlparse

    _, links = fetch_page(src)
    # 국민대 본교: .../notice/7/12563/view.do, 전자공학부: .../board/notice/2663
    base = urlparse(src).path.rsplit("/index.do", 1)[0].rstrip("/")
    posts = [l for l in links
             if urlparse(l["url"]).path.startswith(base + "/")
             and re.search(r"/\d{2,}(/|$)", urlparse(l["url"]).path[len(base):])]
    seen_urls, uniq = set(), []
    for l in (posts or links):
        if l["url"] not in seen_urls:
            seen_urls.add(l["url"])
            uniq.append(l)
    return uniq[:limit]


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


def collect_notices(boards: dict, per_board: int = 4, log=None) -> list[dict]:
    """'한눈에 보기'용: 게시판마다 최신 공지 몇 개씩 본문까지 모아옵니다."""
    samples = load_sample_notices()
    out = []
    for name, src in boards.items():
        if src == "sample":
            picked = [n for n in samples if n["board"] == name]
            out += [dict(n, url="") for n in picked]
            continue
        try:
            posts = list_board_posts(src)[:per_board]
        except Exception as e:
            if log:
                log(f"⚠️ {name} 게시판을 열 수 없어요: {e}")
            continue
        for p in posts:
            try:
                text, _ = fetch_page(p["url"])
            except Exception:
                continue
            out.append({"title": p["title"], "board": name, "date": "", "url": p["url"],
                        "text": f"제목: {p['title']}\n게시판: {name}\n내용: {text[:1500]}"})
        if log:
            log(f"🔍 {name}: 공지 {len(posts)}개 읽음")
    for i, n in enumerate(out, 1):
        n["id"] = str(i)
    return out


# ──────────────────────────────────────────────
# Agent 도구 상자
# ──────────────────────────────────────────────
class Toolbox:
    """boards: {'게시판 이름': 'sample' 또는 'https://...'}"""

    def __init__(self, boards: dict, today: date):
        self.boards = boards
        self.today = today
        self.samples = load_sample_notices()
        self.calendar: list[dict] = []   # Agent가 등록한 마감일
        self.new_found: list[str] = []   # 이번에 새로 발견한 공지

    # 1) 게시판 목록 보기
    def list_boards(self) -> dict:
        return {"boards": list(self.boards.keys())}

    # 2) 게시판에서 최신 공지 제목 보기
    def read_board(self, board: str) -> dict:
        src = self.boards.get(board)
        if src is None:
            return {"error": f"'{board}' 게시판이 없어요. list_boards로 확인하세요."}
        mem = load_memory()
        if src == "sample":
            items = [{"ref": n["id"], "title": n["title"], "date": n["date"]}
                     for n in self.samples if n["board"] == board]
        else:
            try:
                items = [{"ref": l["url"], "title": l["title"]} for l in list_board_posts(src)]
            except Exception as e:
                return {"error": f"게시판을 열 수 없어요: {e}"}
        for it in items:
            it["new"] = it["ref"] not in mem["seen"]
        return {"board": board, "count": len(items), "notices": items}

    # 3) 공지 하나 자세히 읽기
    def read_notice(self, ref: str) -> dict:
        ref = str(ref)
        content = next((n["text"] for n in self.samples if n["id"] == ref), None)
        if content is None and ref.startswith("http"):
            try:
                text, _ = fetch_page(ref)
                content = text[:3000]
            except Exception as e:
                return {"error": f"공지를 열 수 없어요: {e}"}
        if content is None:
            return {"error": f"'{ref}' 공지를 찾을 수 없어요."}
        mem = load_memory()
        if ref not in mem["seen"]:  # 읽은 공지로 기억 (아침 브리핑에서 '새 공지' 구분용)
            mem["seen"].append(ref)
            save_memory(mem)
            self.new_found.append(ref)
        return {"ref": ref, "content": content}

    # 3-2) 공지 여러 개 한 번에 읽기 (AI 호출 횟수 절약)
    def read_notices(self, refs: list) -> dict:
        return {"notices": [self.read_notice(r) for r in list(refs)[:5]]}

    # 4) 사용자에 대한 기억 꺼내기
    def get_user_memory(self) -> dict:
        mem = load_memory()
        return {"좋아한 공지": mem["likes"], "싫어한 공지": mem["dislikes"], "사용자 메모": mem["notes"]}

    # 5) 마감일 캘린더에 등록
    def add_to_calendar(self, title: str, deadline: str, memo: str = "") -> dict:
        try:
            d = datetime.strptime(deadline, "%Y-%m-%d").date()
        except ValueError:
            return {"error": "deadline은 YYYY-MM-DD 형식이어야 해요."}
        if any(c["title"] == title for c in self.calendar):
            return {"ok": True, "note": "이미 등록됨"}
        self.calendar.append({"title": title, "deadline": deadline, "memo": memo})
        return {"ok": True, "d_day": (d - self.today).days}

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name.startswith("_") or name == "call" or fn is None:
            return {"error": f"없는 도구: {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"도구 사용법 오류: {e}"}


# AI에게 알려주는 도구 설명서
TOOL_DECLARATIONS = [
    {"name": "list_boards", "description": "확인할 수 있는 공지 게시판 목록을 봅니다.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "read_board", "description": "게시판의 최신 공지 제목 목록을 봅니다. new=true는 아직 안 읽은 새 공지입니다.",
     "parameters": {"type": "object", "properties": {"board": {"type": "string", "description": "게시판 이름"}},
                    "required": ["board"]}},
    {"name": "read_notices", "description": "관련 있어 보이는 공지 여러 개(최대 5개)의 본문을 한 번에 읽습니다. refs는 read_board 결과의 ref 값 목록입니다.",
     "parameters": {"type": "object", "properties": {"refs": {"type": "array", "items": {"type": "string"}}},
                    "required": ["refs"]}},
    {"name": "get_user_memory", "description": "사용자가 예전에 좋아하거나 싫어한 공지, 남긴 메모를 확인합니다. 추천 전에 꼭 확인하세요.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "add_to_calendar", "description": "추천한 공지의 신청 마감일을 사용자 캘린더에 등록합니다.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string"},
         "deadline": {"type": "string", "description": "YYYY-MM-DD"},
         "memo": {"type": "string", "description": "신청 방법·준비물 한 줄"}},
         "required": ["title", "deadline"]}},
]


# ──────────────────────────────────────────────
# 캘린더 파일 (.ics)
# ──────────────────────────────────────────────
def make_ics(items: list[dict]) -> str:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//KCC//Notice Agent//KO"]
    for i, it in enumerate(items):
        if not it.get("deadline"):
            continue
        day = it["deadline"].replace("-", "")
        lines += [
            "BEGIN:VEVENT", f"UID:notice-{i}-{day}@notice-agent",
            f"DTSTART;VALUE=DATE:{day}",
            f"SUMMARY:[마감] {it['title']}",
            f"DESCRIPTION:{it.get('memo') or it.get('how_to_apply', '')}",
            "BEGIN:VALARM", "TRIGGER:-P3D", "ACTION:DISPLAY", "DESCRIPTION:마감 3일 전", "END:VALARM",
            "BEGIN:VALARM", "TRIGGER:-P1D", "ACTION:DISPLAY", "DESCRIPTION:마감 1일 전", "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


def d_day(deadline, today: date):
    try:
        return (datetime.strptime(deadline, "%Y-%m-%d").date() - today).days
    except (TypeError, ValueError):
        return None


def d_day_badge(d) -> str:
    if d is None:
        return "⚪ 마감일 없음"
    if d < 0:
        return "⚫ 마감 지남"
    if d == 0:
        return "🔴 오늘 마감!"
    if d <= 3:
        return f"🔴 D-{d}"
    if d <= 7:
        return f"🟠 D-{d}"
    return f"🟢 D-{d}"
