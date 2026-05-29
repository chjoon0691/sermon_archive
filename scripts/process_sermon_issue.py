import datetime as dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from github import Github, GithubException
from google import genai
from slugify import slugify


GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]
ISSUE_NUMBER = int(os.environ.get("ISSUE_NUMBER", "0"))

GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get("GEMINI_MODELS", "gemini-2.5-flash-lite,gemini-2.5-flash").split(",")
    if m.strip()
]

LABELS = {
    "request": ("sermon-request", "ededed"),
    "processing": ("processing", "fbca04"),
    "done": ("done", "0e8a16"),
    "failed": ("failed", "b60205"),
}


def repo():
    return Github(GITHUB_TOKEN).get_repo(GITHUB_REPOSITORY)


def ensure_labels(r):
    existing = {label.name for label in r.get_labels()}
    for name, color in LABELS.values():
        if name not in existing:
            try:
                r.create_label(name=name, color=color)
            except GithubException:
                pass


def label_names(issue):
    return {label.name for label in issue.get_labels()}


def add_labels(issue, *names):
    current = label_names(issue)
    for name in names:
        if name and name not in current:
            issue.add_to_labels(name)


def remove_labels(issue, *names):
    current = label_names(issue)
    for name in names:
        if name and name in current:
            issue.remove_from_labels(name)


def korea_today():
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).date().isoformat()


def issue_value(body: str, label: str) -> str:
    m = re.search(rf"### {re.escape(label)}\s*\n\s*(.*?)(?=\n### |\Z)", body or "", re.DOTALL)
    if not m:
        return ""
    value = m.group(1).strip()
    return "" if value in {"_No response_", "No response"} else value


def clean_transcript(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def retryable_error(error: Exception) -> bool:
    s = str(error).lower()
    return any(x in s for x in ["503", "unavailable", "high demand", "429", "resource_exhausted", "try again later", "timeout"])


def client():
    return genai.Client(api_key=GEMINI_API_KEY)


def call_gemini(prompt: str, purpose: str) -> str:
    c = client()
    last = None

    for model in GEMINI_MODELS:
        for attempt in range(1, 5):
            try:
                print(f"{purpose}: {model}, attempt {attempt}", flush=True)
                r = c.models.generate_content(model=model, contents=prompt)
                return (r.text or "").strip()
            except Exception as e:
                last = e
                if not retryable_error(e):
                    raise
                sleep = 15 * (2 ** (attempt - 1)) + random.randint(0, 6)
                print(f"retry after {sleep}s: {e}", flush=True)
                time.sleep(sleep)

    raise RuntimeError(f"Gemini 요청 실패: {last}")


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


def correct(raw: str) -> str:
    out = []
    chunks = split_text(raw)

    for i, chunk in enumerate(chunks, 1):
        prompt = f"""
다음 한국어 기독교 설교 전사문을 오타, 띄어쓰기, 문장부호, 성경 인명/지명/본문 표기만 자연스럽게 바로잡으십시오.
원문의 흐름과 설교자의 어투를 보존하고, 요약하거나 내용을 추가하지 마십시오.
구간 표시는 있으면 유지하십시오. 결과는 수정된 설교문 본문만 출력하십시오.

전사문 조각 {i}/{len(chunks)}:
{chunk}
"""
        out.append(call_gemini(prompt, f"오타 수정 {i}/{len(chunks)}"))

    return "\n\n".join(out).strip()


def analyze(corrected: str, meta: Dict[str, str]) -> Dict[str, Any]:
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
- 유튜브 주소: {meta.get("youtubeUrl") or "미입력"}
- 원문 수집 방식: issue_transcript_success_mode

설교문:
{corrected[:70000]}
"""
    data = safe_json(call_gemini(prompt, "설교 분석"))

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

    data["sourceType"] = "issue_transcript_success_mode"
    return data


def sermon_id(date: str, title: str) -> str:
    return f"{date}-{slugify(title or 'sermon', lowercase=True) or 'sermon'}"[:120].strip("-")


def card_md(item: Dict[str, Any]) -> str:
    outline = "\n".join(
        f"{i+1}. **{o.get('title','대지')}** — {o.get('summary','')}"
        for i, o in enumerate(item.get("outline", []))
    ) or "대지 정보가 없습니다."

    apps = "\n".join(f"- {a}" for a in item.get("applications", [])) or "적용점 정보가 없습니다."

    ills = "\n".join(
        f"- **{x.get('title','예화')}**: {x.get('summary','')}"
        for x in item.get("illustrations", [])
    ) or "추출된 예화가 없습니다."

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


def save(raw: str, corrected: str, analysis: Dict[str, Any], meta: Dict[str, str]) -> Dict[str, Any]:
    date = analysis.get("date") or korea_today()
    title = analysis.get("title") or "제목 미상"
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
            "sermonCard": f"sermons/{sid}/sermon_card.md",
        },
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


def main():
    r = repo()
    ensure_labels(r)
    issue = r.get_issue(number=ISSUE_NUMBER)

    add_labels(issue, LABELS["request"][0], LABELS["processing"][0])
    remove_labels(issue, LABELS["failed"][0])

    body = issue.body or ""
    meta = {
        "youtubeUrl": issue_value(body, "유튜브 주소"),
        "title": issue_value(body, "설교 제목"),
        "speaker": issue_value(body, "설교자"),
        "bibleText": issue_value(body, "본문"),
        "church": issue_value(body, "교회/채널"),
        "date": issue_value(body, "설교 날짜"),
        "memo": issue_value(body, "메모"),
    }

    raw = clean_transcript(issue_value(body, "설교 스크립트"))

    if len(raw) < 100:
        remove_labels(issue, LABELS["processing"][0])
        add_labels(issue, LABELS["failed"][0])
        issue.create_comment(
            "❌ 설교 스크립트가 너무 짧거나 비어 있습니다.\n\n"
            "성공했던 실행과 동일하게 처리하려면 Issue의 `설교 스크립트` 칸에 실제 설교 전사문을 붙여넣어 주세요."
        )
        raise RuntimeError("설교 스크립트가 너무 짧거나 비어 있습니다.")

    try:
        issue.create_comment(
            "설교 아카이브 처리를 시작합니다.\n\n"
            "- 방식: 성공했던 실행과 동일한 스크립트 기반 처리\n"
            f"- 입력 분량: 약 {len(raw):,}자\n"
            "- 유튜브 영상 전사 단계는 실행하지 않습니다."
        )

        corrected = correct(raw)
        analysis = analyze(corrected, meta)
        item = save(raw, corrected, analysis, meta)

        remove_labels(issue, LABELS["processing"][0], LABELS["failed"][0])
        add_labels(issue, LABELS["done"][0])

        issue.create_comment(
            "✅ 설교 아카이브 생성이 완료되었습니다.\n\n"
            f"- 제목: {item['title']}\n"
            f"- 설교자: {item['speaker']}\n"
            f"- 본문: {item['bibleText']}\n"
            f"- 저장 ID: `{item['id']}`\n"
            f"- 처리 방식: {item['sourceType']}\n\n"
            "잠시 후 GitHub Pages에서 확인할 수 있습니다."
        )

    except Exception as e:
        remove_labels(issue, LABELS["processing"][0])
        add_labels(issue, LABELS["failed"][0])
        issue.create_comment(
            "❌ 설교 아카이브 생성 중 오류가 발생했습니다.\n\n"
            f"```text\n{e}\n```"
        )
        raise


if __name__ == "__main__":
    main()
