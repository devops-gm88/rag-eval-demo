"""Build the corpus from our own material.

Sources, as specified in SESSION_START_RAG_FASTAPI_2026-10-31.md:
  * the case studies in n8n_demo/**/CASE_STUDY_*.html
  * the case-study PDFs in proposal_assets_*/case_study_*.pdf
  * the specs in MARK/Projects/Build_Ideas/*.md
  * the durable sections of WORKING_KNOWLEDGE.md

Output: data/corpus/documents.jsonl, one JSON object per document:
  {"source": "path#section", "title": ..., "kind": "html|pdf|md",
   "text": ..., "chars": int, "sha256": ...}

The redaction pass removes client and personal names from the published
corpus. It is checked by tests/test_corpus.py.
"""
from __future__ import annotations

import hashlib
import html as htmllib
import json
import os
import re
from pathlib import Path

from .config import CORPUS_DIR, ROOT

SOURCE_ROOT_DEFAULT = Path(
    os.environ.get("RAG_SOURCE_ROOT", "/Users/moidchowdhury/Workspace")
)

# Names that must never reach the published corpus.
REDACTIONS = {
    "Natalie": "a client",
    "Juliana": "a client",
    "Devan": "a client",
    "Mark Sosik": "a client",
    "Owen Nordberg": "a client",
    "Raul": "a client",
    "Nerea": "a client",
    "Maria Simonetti": "a client",
    "Axess": "a client",
    "Kunal": "a client",
    "Patrick": "a client contact",
    "Vipul": "a competitor",
    "Reed": "a client",
    "Moid Chowdhury": "the owner",
    "SK Saimon": "the owner",
    # Client and vendor organisations named in the source specs.
    "Dream Big Marketing Services": "a client",
    "Boreal Global Trading": "a client",
    "The Equine Practice Company": "a client",
    "MarketLeague": "a client",
    "Raffall": "a client",
    "My Home Solutions AZ": "a client",
    "Hyper Solutions": "a vendor",
    "SYC": "a client",
    "Trace (regional underground/telecom construction)": "a client",
}

# Commercial facts that stay out of a public demo corpus.
REDACT_RE = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"https?://\S+"), "[url]"),
    (re.compile(r"devan_trial"), "[client-project]"),
]


PERSONAL_ORGS = [
    "Natalie", "Juliana", "Devan", "Mark Sosik", "Owen Nordberg", "Raul",
    "Nerea", "Maria Simonetti", "Axess", "Kunal", "Patrick", "Vipul", "Reed",
    "Owen", "Boreal",
    "Boreal", "Moid", "Saimon",
]


def redact(text: str) -> str:
    for name, replacement in REDACTIONS.items():
        text = re.sub(rf"\b{re.escape(name)}\b", replacement, text)
    for name in PERSONAL_ORGS:
        text = re.sub(rf"\b{re.escape(name)}\b", "a client", text, flags=re.IGNORECASE)
    text = re.sub(r"/Users/[\w.-]+/Workspace", "[workspace]", text)
    for pattern, replacement in REDACT_RE:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------- extraction

TAG_BLOCK = re.compile(r"<\s*(p|li|br|tr|div|section|h[1-6])[^>]*>", re.I)


def html_to_text(raw: str) -> str:
    """Turn a standalone case-study page into heading-preserving text."""
    raw = re.sub(r"(?is)<(script|style|nav|footer)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)

    def heading(match: re.Match) -> str:
        level = int(match.group(1))
        body = strip_tags(match.group(2)).strip()
        return f"\n\n{'#' * level} {body}\n\n" if body else "\n"

    raw = re.sub(r"(?is)<h([1-6])[^>]*>(.*?)</h\1>", heading, raw)
    raw = re.sub(r"(?i)<li[^>]*>", "\n- ", raw)
    raw = re.sub(r"(?i)</(p|li|tr|div|section|table|ul|ol)>", "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)<(td|th)[^>]*>", " | ", raw)
    text = strip_tags(raw)
    text = htmllib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\| *", " | ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def strip_tags(raw: str) -> str:
    return re.sub(r"(?s)<[^>]+>", " ", raw)


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def md_to_text(raw: str) -> str:
    # Strip YAML front matter only when it really is key: value pairs.
    raw = re.sub(r"\A---\n(?:[A-Za-z_][\w-]*:.*\n)+\s*---\n", "", raw)
    return raw.strip()


# ---------------------------------------------------------------- splitting


def split_sections(text: str, level: int = 2) -> list[tuple[str, str]]:
    """Split markdown-ish text at a heading level. Returns [(heading, body)]."""
    marker = "#" * level
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current: list[str] = []
    for line in lines:
        if re.match(rf"^{marker} [^#]", line):
            sections.append((current_heading, "\n".join(current).strip()))
            current_heading = line[len(marker) :].strip()
            current = []
        else:
            current.append(line)
    sections.append((current_heading, "\n".join(current).strip()))
    return [(h, b) for h, b in sections if b]


def slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:60] or "section"


def make_document(source_root: Path, rel: str, section: str, kind: str, text: str) -> dict:
    title = section or Path(rel).stem
    source = f"{rel}#{section}" if section else rel
    clean = redact(text)
    return {
        "source": redact(source),
        "title": redact(title),
        "kind": kind,
        "text": clean,
        "chars": len(clean),
        "sha256": hashlib.sha256(clean.encode()).hexdigest()[:16],
    }


# ---------------------------------------------------------------- sources


def collect(source_root: Path) -> list[dict]:
    docs: list[dict] = []

    # 1. Case-study HTML, split at level-2 headings (a section per document).
    for path in sorted(source_root.glob("MARK/Projects/Job_Search/n8n_demo/**/CASE_STUDY_*.html")):
        rel = str(path.relative_to(source_root))
        text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
        for heading, body in split_sections(text, level=2):
            if len(body) < 200:
                continue
            docs.append(make_document(source_root, rel, heading, "html", body))

    # 2. Case-study PDFs.
    for path in sorted(source_root.glob("MARK/Projects/Job_Search/proposal_assets_*/case_study_*.pdf")):
        rel = str(path.relative_to(source_root))
        docs.append(make_document(source_root, rel, "", "pdf", pdf_to_text(path)))

    # 3. Build specs.
    for path in sorted(source_root.glob("MARK/Projects/Build_Ideas/*.md")):
        rel = str(path.relative_to(source_root))
        text = md_to_text(path.read_text(encoding="utf-8", errors="replace"))
        docs.append(make_document(source_root, rel, "", "md", text))

    # 4. Durable sections of WORKING_KNOWLEDGE (sections 3-9 only:
    #    rules, capability, working style, lessons, what fails, layout, estimating).
    #    Section 3 is split into one document per rule: the rules are the facts.
    wk = source_root / "WORKING_KNOWLEDGE.md"
    if wk.exists():
        text = md_to_text(wk.read_text(encoding="utf-8", errors="replace"))
        for heading, body in split_sections(text, level=2):
            number = heading.split(".")[0].strip()
            if not number.isdigit() or not 3 <= int(number) <= 9:
                continue
            if int(number) == 3:
                for rule_heading, rule_body in split_rules(body):
                    docs.append(make_document(source_root, "WORKING_KNOWLEDGE.md", rule_heading, "md", rule_body))
            elif len(body) >= 200:
                docs.append(make_document(source_root, "WORKING_KNOWLEDGE.md", heading, "md", body))

    return [doc for doc in docs if doc["chars"] >= 100]


RULE_RE = re.compile(r"^- \*\*(Rule \d+)[^*]*\*\*", re.M)


def split_rules(body: str) -> list[tuple[str, str]]:
    """Split the standing-rules section into one document per rule."""
    starts = [match.start() for match in RULE_RE.finditer(body)]
    if not starts:
        return [("3. Standing rules — the gates", body)]
    parts: list[tuple[str, str]] = []
    preamble = body[: starts[0]].strip()
    if preamble:
        parts.append(("3. Standing rules — the gates", preamble))
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(body)
        chunk = body[start:end].strip()
        label = RULE_RE.match(chunk)
        heading = label.group(1) if label else f"rule-{index}"
        title_line = chunk.splitlines()[0]
        title = re.sub(r"^[-*] \*\*|\*\*.*$", "", title_line).strip()
        if title.startswith(heading):
            label = title
        elif title:
            label = f"{heading} — {title}"
        else:
            label = heading
        parts.append((label, chunk))
    return parts


def write_corpus(source_root: Path = SOURCE_ROOT_DEFAULT) -> Path:
    docs = collect(source_root)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out = CORPUS_DIR / "documents.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    all_text = "\n".join(d["text"] for d in docs)
    manifest = {
        "documents": len(docs),
        "chars": sum(d["chars"] for d in docs),
        "by_kind": {k: sum(1 for d in docs if d["kind"] == k) for k in ("html", "pdf", "md")},
        "corpus_sha256": hashlib.sha256(all_text.encode()).hexdigest(),
    }
    # Per-document identity stays local: the public repo ships the summary only.
    detail = {
        "source_root": str(source_root),
        "document_sha256": {d["source"]: d["sha256"] for d in docs},
    }
    (CORPUS_DIR / "manifest_detail.json").write_text(json.dumps(detail, indent=2) + "\n")
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def load_documents() -> list[dict]:
    path = CORPUS_DIR / "documents.jsonl"
    if not path.exists():
        raise SystemExit(
            f"Corpus missing: {path}\nRun `make corpus` first (see README)."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    target = write_corpus()
    docs = load_documents()
    print(f"wrote {len(docs)} documents -> {target}")
