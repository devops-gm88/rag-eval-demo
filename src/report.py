"""Render reports/eval.html from the eval JSON files.

    python -m src.report
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from .config import REPORTS_DIR

ORDER = ["baseline", "fixed_hybrid", "structural_vector", "improved"]
LABELS = {
    "baseline": "Baseline — fixed chunks, vector only",
    "fixed_hybrid": "Fixed chunks + full-text hybrid",
    "structural_vector": "Structural chunks + vector only",
    "improved": "Improved — structural chunks + full-text hybrid",
}
FILES = {
    "baseline": "eval_baseline.json",
    "fixed_hybrid": "eval_fixed_hybrid.json",
    "structural_vector": "eval_structural_vector.json",
    "improved": "eval_improved.json",
}


def load_reports() -> dict[str, dict]:
    reports = {}
    for name, filename in FILES.items():
        path = REPORTS_DIR / filename
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def esc(text: str) -> str:
    return html.escape(str(text))


def pct(part: int, total: int) -> str:
    return f"{round(100 * part / total)}%" if total else "0%"


def summary_rows(reports: dict[str, dict]) -> str:
    rows = []
    for name in ORDER:
        if name not in reports:
            continue
        report = reports[name]
        totals = report["totals"]
        bar = "PASS" if totals["pass_bar_met"] else "FAIL"
        css = "ok" if totals["pass_bar_met"] else "bad"
        rows.append(
            f"""<tr>
              <td class="name">{esc(LABELS[name])}</td>
              <td>{totals['recall_at_5']} / 30<span class="pct">{pct(totals['recall_at_5'], 30)}</span></td>
              <td>{totals['citation_correct']} / 30<span class="pct">{pct(totals['citation_correct'], 30)}</span></td>
              <td>{totals['answer_complete']} / 30<span class="pct">{pct(totals['answer_complete'], 30)}</span></td>
              <td class="strong">{totals['passed']} / 30</td>
              <td>{totals['refusals_correct']} / 3</td>
              <td class="{css}">{bar}</td>
            </tr>"""
        )
    return "\n".join(rows)


def question_table(reports: dict[str, dict]) -> str:
    baseline = reports.get("baseline")
    improved = reports.get("improved")
    if not baseline or not improved:
        return "<p class='muted'>Baseline and improved reports are required.</p>"
    improved_by_id = {row["id"]: row for row in improved["rows"]}
    rows = []
    for row in baseline["rows"]:
        other = improved_by_id.get(row["id"], {})

        def marks(item: dict) -> str:
            if not item:
                return "—"
            bits = []
            bits.append("<i class='ok'>R</i>" if item["recall_at_5"] else "<i class='bad'>R</i>")
            bits.append("<i class='ok'>C</i>" if item["citation_correct"] else "<i class='bad'>C</i>")
            bits.append("<i class='ok'>A</i>" if item["answer_complete"] else "<i class='bad'>A</i>")
            return " ".join(bits)

        verdict = "ok" if other.get("passed") else "bad"
        rows.append(
            f"""<tr>
              <td class="mono">{esc(row['id'])}</td>
              <td>{esc(row['question'])}</td>
              <td class="mono fact">{esc(row['key_fact'])}</td>
              <td class="checks">{marks(row)}</td>
              <td class="checks">{marks(other)}</td>
              <td class="{verdict}">{"pass" if other.get("passed") else "fail"}</td>
            </tr>"""
        )
    return "\n".join(rows)


def delta_block(reports: dict[str, dict]) -> str:
    if "baseline" not in reports or "improved" not in reports:
        return ""
    base = reports["baseline"]["totals"]
    improved = reports["improved"]["totals"]
    parts = []

    def line(label: str, key: str) -> str:
        delta = improved[key] - base[key]
        sign = "+" if delta >= 0 else ""
        return f"<li>{label}: <b>{base[key]} → {improved[key]}</b> <span class='delta'>({sign}{delta})</span></li>"

    parts.append(line("Questions passed", "passed"))
    parts.append(line("Retrieval recall@5", "recall_at_5"))
    parts.append(line("Citation correctness", "citation_correct"))
    parts.append(line("Answer completeness", "answer_complete"))
    parts.append(line("Refusals correct", "refusals_correct"))

    intermediate = []
    if "structural_vector" in reports:
        t = reports["structural_vector"]["totals"]
        intermediate.append(
            f"<li>Structural chunking alone, vector only: <b>{t['passed']} / 30</b> passed "
            f"(recall {t['recall_at_5']}, citation {t['citation_correct']}, complete {t['answer_complete']}).</li>"
        )
    if "fixed_hybrid" in reports:
        t = reports["fixed_hybrid"]["totals"]
        intermediate.append(
            f"<li>Full-text hybrid alone, fixed chunks: <b>{t['passed']} / 30</b> passed "
            f"(recall {t['recall_at_5']}, citation {t['citation_correct']}, complete {t['answer_complete']}).</li>"
        )
    return f"<ul class='delta'>{''.join(parts)}</ul><ul class='notes'>{''.join(intermediate)}</ul>"


def failures_block(reports: dict[str, dict]) -> str:
    report = reports.get("improved")
    if not report:
        return ""
    rows = []
    for row in report["rows"]:
        if row["passed"]:
            continue
        causes = []
        if not row["recall_at_5"]:
            causes.append("retrieval missed the expected source")
        if not row["citation_correct"]:
            causes.append("the cited chunk does not hold the key fact")
        if not row["answer_complete"]:
            causes.append("the answer omits the key fact")
        retrieved = ", ".join(esc(item["source"].split("/")[-1]) for item in row["retrieved"][:3])
        rows.append(
            f"""<tr>
              <td class="mono">{esc(row['id'])}</td>
              <td>{esc(row['question'])}</td>
              <td class="mono fact">{esc(row['key_fact'])}</td>
              <td>{esc('; '.join(causes))}</td>
              <td class="answer">{esc(row['answer'][:220])}</td>
              <td class="sources">{retrieved}</td>
            </tr>"""
        )
    if not rows:
        return "<p>No failures at the improved configuration.</p>"
    return "\n".join(rows)


def render(reports: dict[str, dict]) -> str:
    environment = next(iter(reports.values()))["environment"] if reports else {}
    baseline = reports.get("baseline", {}).get("totals", {})
    improved = reports.get("improved", {}).get("totals", {})
    delta_passed = improved.get("passed", 0) - baseline.get("passed", 0)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG evaluation report</title>
<style>
  :root {{
    --ink: #16181d; --muted: #5f6673; --line: #d9dde3; --wash: #f5f6f8;
    --ok: #0f6b3d; --bad: #a3231b; --acc: #1f4e79;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #fff; color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 48px 32px 96px; }}
  header {{ border-bottom: 2px solid var(--ink); padding-bottom: 20px; margin-bottom: 28px; }}
  .kicker {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
  h1 {{ font-size: 34px; line-height: 1.15; margin: 8px 0 10px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 20px; margin: 44px 0 12px; padding-top: 8px; border-top: 1px solid var(--line); }}
  p {{ margin: 0 0 12px; }}
  .muted {{ color: var(--muted); }}
  .lede {{ font-size: 17px; max-width: 74ch; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0 8px; }}
  .card {{ border: 1px solid var(--line); padding: 14px 16px; }}
  .card .n {{ font-size: 28px; font-weight: 650; letter-spacing: -0.02em; }}
  .card .l {{ font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0 8px; font-size: 13.5px; }}
  th {{ text-align: left; font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
        color: var(--muted); border-bottom: 1px solid var(--ink); padding: 8px 8px; font-weight: 600; }}
  td {{ border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }}
  tr:hover td {{ background: var(--wash); }}
  td.name {{ font-weight: 600; }}
  td.strong {{ font-weight: 700; }}
  .pct {{ display: block; color: var(--muted); font-size: 11px; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  .fact {{ color: var(--acc); }}
  .ok {{ color: var(--ok); }}
  .bad {{ color: var(--bad); }}
  .checks i {{ font-style: normal; font-weight: 700; font-size: 11px; padding: 1px 3px;
               border: 1px solid currentColor; }}
  .checks i.bad {{ opacity: .85; }}
  .answer {{ font-size: 12.5px; color: var(--muted); max-width: 34ch; }}
  .sources {{ font-size: 11.5px; color: var(--muted); }}
  ul.delta {{ list-style: none; padding: 0; display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px 24px; }}
  ul.delta li {{ border-left: 2px solid var(--ink); padding-left: 10px; }}
  ul.notes {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
  .note {{ border-left: 3px solid var(--ink); background: var(--wash); padding: 12px 16px; margin: 18px 0; }}
  footer {{ margin-top: 56px; border-top: 1px solid var(--line); padding-top: 14px;
            color: var(--muted); font-size: 12px; }}
  @media print {{ body {{ font-size: 11px; }} .wrap {{ padding: 0; }} h2 {{ break-after: avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="kicker">Retrieval-augmented generation · evaluation report</div>
    <h1>30 questions, four configurations, one refusal bar</h1>
    <p class="lede">A corpus of {environment.get('documents', '?')} documents, indexed two ways.
    Every question is scored on retrieval recall@5, citation correctness, answer completeness and
    refusal accuracy. Generated {esc(generated)}.</p>
  </header>

  <div class="cards">
    <div class="card"><div class="n">{baseline.get('passed', 0)} / 30</div><div class="l">Baseline passed</div></div>
    <div class="card"><div class="n">{improved.get('passed', 0)} / 30</div><div class="l">Improved passed</div></div>
    <div class="card"><div class="n">{'+' if delta_passed >= 0 else ''}{delta_passed}</div><div class="l">Delta</div></div>
    <div class="card"><div class="n">{improved.get('refusals_correct', 0)} / 3</div><div class="l">Refusals correct</div></div>
  </div>

  <h2>Every configuration</h2>
  <p class="muted">Pass bar: at least 26 of 30 answered correctly and 3 of 3 refusals.
  R = an expected source is in the top five chunks. C = a cited chunk contains the key fact.
  A = the answer contains the key fact.</p>
  <table>
    <thead><tr><th>Configuration</th><th>Recall@5</th><th>Citations</th><th>Complete</th><th>Passed</th><th>Refusals</th><th>Bar</th></tr></thead>
    <tbody>{summary_rows(reports)}</tbody>
  </table>

  <h2>Baseline to improved</h2>
  {delta_block(reports)}

  <h2>Question by question</h2>
  <table>
    <thead><tr><th>ID</th><th>Question</th><th>Key fact</th><th>Baseline</th><th>Improved</th><th>Improved</th></tr></thead>
    <tbody>{question_table(reports)}</tbody>
  </table>

  <h2>Refusals</h2>
  <p class="muted">Three questions the corpus cannot answer. The correct answer is <b>NOT IN SOURCES</b>.
  A model that guesses fails these.</p>

  <h2>Failure report — improved configuration</h2>
  <table>
    <thead><tr><th>ID</th><th>Question</th><th>Key fact</th><th>Cause</th><th>Answer</th><th>Top sources</th></tr></thead>
    <tbody>{failures_block(reports)}</tbody>
  </table>

  <div class="note">
    <b>How to read the delta.</b> Each configuration changes one thing at a time. The intermediate rows
    separate the effect of structural chunking from the effect of full-text hybrid retrieval. A component
    that does not move the pass count is not kept.
  </div>

  <footer>
    Embedding model: {esc(environment.get('embedding_model', '?'))} ({environment.get('chunks_structural', '?')} structural chunks,
    {environment.get('chunks_fixed', '?')} fixed chunks).
    Answer model: {esc(environment.get('chat_model', '?'))}. Temperature 0.
    Corpus and eval set are property of the author. The harness, the questions and the scoring rules are the artefact.
  </footer>
</div>
</body>
</html>
"""


def main() -> None:
    reports = load_reports()
    if not reports:
        raise SystemExit("no report JSON files found; run `make eval` first")
    target = REPORTS_DIR / "eval.html"
    target.write_text(render(reports), encoding="utf-8")
    print(f"wrote {target} from {len(reports)} configurations")


if __name__ == "__main__":
    main()
