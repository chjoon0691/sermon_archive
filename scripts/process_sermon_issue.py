import datetime as dt
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from github import Github
from google import genai
from google.genai import types
from slugify import slugify

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")
ISSUE_NUMBER = int(os.environ.get("ISSUE_NUMBER", "0"))
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "gemini-2.5-flash-lite,gemini-2.5-flash").split(",") if m.strip()]

def korea_today():
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).date().isoformat()

def issue_value(label: str) -> str:
    m = re.search(rf"### {re.escape(label)}\s*\n\s*(.*?)(?=\n### |\Z)", ISSUE_BODY, re.DOTALL)
    if not m:
        return ""
    value = m.group(1).strip()
    return "" if value in {"_No response_", "No response"} else value

def to_int(value, default, lo, hi):
    try:
        n = int(str(value).strip())
    except Exception:
        n = default
    return max(lo, min(n, hi))

def mmss(sec):
    return f"{sec//60:02d}:{sec%60:02d}"

def split_text(text: str, max_chars=9000) -> List[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + 1000:
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start + 1000:
            boundary = end
        chunks.append(text[start:boundary].strip())
        start = boundary
    return [c for c in chunks if c]

def safe_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    raise RuntimeError("Gemini 분석 결과를 JSON으로 해석하지 못했습니다.")

def retryable(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in ["503", "unavailable", "high demand", "429", "resource_exhausted", "try again later", "timeout"])

def client():
    return genai.Client(api_key=GEMINI_API_KEY)

def call_gemini(fn, purpose):
    last = None
    for model in GEMINI_MODELS:
        for attempt in range(1, 5):
            try:
                print(f"{purpose}: {model}, attempt {attempt}", flush=True)
                return fn(model)
            except Exception as e:
                last = e
                if not retryable(e):
                    raise
                sleep = 15 * (2 ** (attempt - 1)) + random.randint(0, 6)
                print(f"retry after {sleep}s: {e}", flush=True)
                time.sleep(sleep)
    raise RuntimeError(f"Gemini 요청 실패: {last}")

def youtube_request(url, prompt, start, end):
    c = client()
    part = types.Part(
        file_data=types.FileData(file_uri=url),
        video_metadata=types.VideoMetadata(start_offset=f"{int(start)}s", end_offset=f"{int(end)}s"),
    )
    def run(model):
        r = c.models.generate_content(model=model, contents=types.Content(parts=[part, types.Part(text=prompt)]))
        return (r.text or "").strip()
    return call_gemini(run, "YouTube 구간 전사")

def text_request(prompt, purpose):
    c = client()
    def run(model):
        r = c.models.generate_content(model=model, contents=prompt)
        return (r.text or "").strip()
    return call_gemini(run, purpose)

def transcribe(url, max_minutes, segment_minutes):
    chunks = []
    total = (max_minutes + segment_minutes - 1) // segment_minutes
    for i in range(total):
        start = i * segment_minutes * 60
        end = min((i + 1) * segment_minutes * 60, max_minutes * 60)
        s, e = mmss(start), mmss(end)
        prompt = f"""
당신은 한국어 기독교 설교 전문 전사자입니다.

이 YouTube 영상의 {s}~{e} 구간에서 들리는 설교 음성을 가능한 한 충실하게 한국어 문장으로 전사하십시오.
- 요약하지 마십시오.
- 해설을 붙이지 마십시오.
- 영상 분석 설명을 하지 마십시오.
- 이 구간이 영상 길이를 넘어가거나 실제 음성이 없으면 정확히 [END]라고만 출력하십시오.
- 결과는 전사문 본문만 출력하십시오.
"""
        t = youtube_request(url, prompt, start, end)
        if not t or "[END]" in t[:80].upper():
            break
        chunks.append(f"[{s}~{e}]\n{t.strip()}")
    raw = "\n\n".join(chunks).strip()
    if len(raw) < 200:
        raise RuntimeError("전사 결과가 너무 짧습니다. 영상 접근 제한 또는 Gemini 처리 실패일 수 있습니다.")
    return raw

def correct(raw):
    out = []
    chunks = split_text(raw)
    for i, chunk in enumerate(chunks, 1):
        prompt = f"""
다음 한국어 기독교 설교 전사문을 오타, 띄어쓰기, 문장부호, 성경 인명/지명/본문 표기만 자연스럽게 바로잡으십시오.
원문의 흐름과 설교자의 어투를 보존하고, 요약하거나 내용을 추가하지 마십시오.
구간 표시는 유지하십시오. 결과는 수정된 설교문 본문만 출력하십시오.

전사문 조각 {i}/{len(chunks)}:
{chunk}
"""
        out.append(text_request(prompt, f"오타 수정 {i}/{len(chunks)}"))
    return "\n\n".join(out).strip()

def analyze(corrected, meta):
    prompt = f"""
다음 설교문을 설교 아카이브용으로 분석하십시오. 반드시 JSON 객체만 출력하십시오.

JSON 형식:
{{
  "title": "설교 제목",
  "speaker": "설교자",
  "date": "YYYY-MM-DD 또는 미상",
  "bibleText": "본문",
  "church": "교회 또는 채널",
  "summary": "설교 전체 요약. 800~1200자 정도.",
  "mainMessage": "핵심 메시지 한 문단",
  "outline": [{{"title": "대지 제목", "summary": "해당 대지 요약"}}],
  "topics": ["주제색인1", "주제색인2"],
  "applications": ["적용점1", "적용점2"],
  "illustrations": [{{"title": "예화 제목", "summary": "예화 요약", "topics": ["연결 주제"]}}]
}}

사용자 입력:
- 설교 제목: {meta.get("title") or "미입력"}
- 설교자: {meta.get("speaker") or "미입력"}
- 본문: {meta.get("bibleText") or "미입력"}
- 교회/채널: {meta.get("church") or "미입력"}
- 설교 날짜: {meta.get("date") or "미입력"}
- 유튜브 주소: {meta.get("youtubeUrl")}

설교문:
{corrected[:70000]}
"""
    data = safe_json(text_request(prompt, "설교 분석"))
    for k in ["title", "speaker", "bibleText", "church", "date"]:
        if meta.get(k):
            data[k] = meta[k]
    data.setdefault("title", "제목 미상")
    data.setdefault("speaker", "설교자 미상")
    data.setdefault("bibleText", "본문 미상")
    data.setdefault("church", "미상")
    if not data.get("date") or data.get("date") == "미상":
        data["date"] = korea_today()
    for k in ["outline", "topics", "applications", "illustrations"]:
        if not isinstance(data.get(k), list):
            data[k] = []
    data["sourceType"] = "github_actions_gemini_youtube_url"
    return data

def sermon_id(date, title):
    return f"{date}-{slugify(title or 'sermon', lowercase=True) or 'sermon'}"[:120].strip("-")

def card_md(item):
    outline = "\n".join(f"{i+1}. **{o.get('title','대지')}** — {o.get('summary','')}" for i,o in enumerate(item.get("outline", []))) or "대지 정보가 없습니다."
    apps = "\n".join(f"- {a}" for a in item.get("applications", [])) or "적용점 정보가 없습니다."
    ills = "\n".join(f"- **{x.get('title','예화')}**: {x.get('summary','')}" for x in item.get("illustrations", [])) or "추출된 예화가 없습니다."
    topics = ", ".join(item.get("topics", [])) or "주제 정보가 없습니다."
    return f"""# {item.get('title','제목 미상')}

- 설교자: {item.get('speaker','설교자 미상')}
- 날짜: {item.get('date','날짜 미상')}
- 본문: {item.get('bibleText','본문 미상')}
- 교회/채널: {item.get('church','미상')}
- 영상: {item.get('videoUrl','')}
- 원문 수집 방식: {item.get('sourceType','')}

## 핵심 메시지

{item.get('mainMessage','')}

## 설교 요약

{item.get('summary','')}

## 대지

{outline}

## 적용점

{apps}

## 주제 색인

{topics}

## 예화

{ills}
"""

def save(raw, corrected, analysis, meta):
    date, title = analysis.get("date") or korea_today(), analysis.get("title") or "제목 미상"
    sid = sermon_id(date, title)
    base = Path("sermons") / sid
    base.mkdir(parents=True, exist_ok=True)
    item = {
        "id": sid,
        "videoUrl": meta.get("youtubeUrl", ""),
        "title": title,
        "speaker": analysis.get("speaker", "설교자 미상"),
        "date": date,
        "bibleText": analysis.get("bibleText", "본문 미상"),
        "church": analysis.get("church", "미상"),
        "summary": analysis.get("summary", ""),
        "mainMessage": analysis.get("mainMessage", ""),
        "outline": analysis.get("outline", []),
        "topics": analysis.get("topics", []),
        "applications": analysis.get("applications", []),
        "illustrations": analysis.get("illustrations", []),
        "sourceType": analysis.get("sourceType", ""),
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "issueNumber": ISSUE_NUMBER,
        "files": {
            "rawTranscript": f"sermons/{sid}/raw_transcript.md",
            "correctedTranscript": f"sermons/{sid}/corrected_transcript.md",
            "analysis": f"sermons/{sid}/analysis.json",
            "illustrations": f"sermons/{sid}/illustrations.json",
            "sermonCard": f"sermons/{sid}/sermon_card.md"
        }
    }
    (base / "raw_transcript.md").write_text(raw, encoding="utf-8")
    (base / "corrected_transcript.md").write_text(corrected, encoding="utf-8")
    (base / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "illustrations.json").write_text(json.dumps(item["illustrations"], ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "sermon_card.md").write_text(card_md(item), encoding="utf-8")
    Path("data").mkdir(exist_ok=True)
    idx = Path("data/sermons.json")
    sermons = []
    if idx.exists():
        try:
            old = json.loads(idx.read_text(encoding="utf-8"))
            sermons = old.get("sermons", old if isinstance(old, list) else [])
        except Exception:
            sermons = []
    replaced = False
    for i, old in enumerate(sermons):
        if old.get("id") == item["id"] or old.get("videoUrl") == item["videoUrl"]:
            sermons[i] = item
            replaced = True
            break
    if not replaced:
        sermons.insert(0, item)
    idx.write_text(json.dumps({"sermons": sermons}, ensure_ascii=False, indent=2), encoding="utf-8")
    return item

def comment(msg):
    repo = Github(GITHUB_TOKEN).get_repo(GITHUB_REPOSITORY)
    repo.get_issue(number=ISSUE_NUMBER).create_comment(msg)

def main():
    url = issue_value("유튜브 주소")
    if not url:
        raise RuntimeError("유튜브 주소를 찾지 못했습니다.")
    meta = {
        "youtubeUrl": url,
        "title": issue_value("설교 제목"),
        "speaker": issue_value("설교자"),
        "bibleText": issue_value("본문"),
        "church": issue_value("교회/채널"),
        "date": issue_value("설교 날짜"),
        "memo": issue_value("메모"),
    }
    max_minutes = to_int(issue_value("최대 처리 시간(분)"), 30, 5, 180)
    segment_minutes = to_int(issue_value("구간 길이(분)"), 5, 3, 20)
    try:
        comment(f"설교 아카이브 처리를 시작합니다.\\n\\n- 최대 처리 시간: {max_minutes}분\\n- 구간 길이: {segment_minutes}분")
        raw = transcribe(url, max_minutes, segment_minutes)
        corrected = correct(raw)
        analysis = analyze(corrected, meta)
        item = save(raw, corrected, analysis, meta)
        comment(f"✅ 설교 아카이브 생성이 완료되었습니다.\\n\\n- 제목: {item['title']}\\n- 설교자: {item['speaker']}\\n- 본문: {item['bibleText']}\\n- 저장 ID: `{item['id']}`\\n\\n잠시 후 GitHub Pages에서 확인할 수 있습니다.")
    except Exception as e:
        comment(f"❌ 설교 아카이브 생성 중 오류가 발생했습니다.\\n\\n```text\\n{e}\\n```\\n\\nGemini 503 혼잡 오류라면 10~30분 뒤 Issue를 다시 편집하거나, 최대 처리 시간을 줄여 다시 시도해 보세요.")
        raise

if __name__ == "__main__":
    main()
