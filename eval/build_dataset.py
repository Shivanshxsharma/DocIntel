"""
Build the QASPER evaluation subset
==================================
QASPER (Dasigi et al., NAACL 2021, AllenAI) is a QA benchmark over full-text NLP
research papers. Every answer is annotated by NLP practitioners together with the
exact paragraphs of the paper that support it ("evidence"). That evidence is what
makes a pipeline-agnostic retrieval metric possible: a retrieved chunk is relevant
if it contains an evidence paragraph, no matter how either pipeline chunked it.

This script:
  1. Picks a seeded, reproducible sample of papers from the QASPER test split.
  2. Downloads each paper's real PDF from arXiv (politely, 1 request / 3 s).
  3. Extracts page text with PyMuPDF — a NEUTRAL parser that neither pipeline
     under test uses — and keeps only questions whose gold evidence can actually
     be located in the PDF (guards against arXiv version drift).
  4. Writes eval/data/qasper_subset.json.

Run:  python eval/build_dataset.py --papers 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pymupdf

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from textmatch import match_score, trigrams  # noqa: E402

RAW_PARQUET = EVAL_DIR / "data" / "raw" / "qasper_test.parquet"
PAPERS_DIR = EVAL_DIR / "data" / "papers"
OUT_PATH = EVAL_DIR / "data" / "qasper_subset.json"

MAX_PAGES = 16            # keep the sample to normal-length conference papers
LOCATE_THRESHOLD = 0.6    # evidence trigram coverage needed to count as "in the PDF"


def _clean_evidence(text: str) -> str:
    # Table/figure evidence is stored as "FLOAT SELECTED: Table 2: <caption>".
    return re.sub(r"^FLOAT SELECTED:\s*", "", text.strip())


def _answer_text(ans: dict) -> str:
    if ans["unanswerable"]:
        return ""
    if ans["yes_no"] is not None:
        return "Yes" if ans["yes_no"] else "No"
    spans = [s.strip() for s in (ans["extractive_spans"] if ans["extractive_spans"] is not None else []) if s.strip()]
    if spans:
        return ", ".join(spans)
    return (ans["free_form_answer"] or "").strip()


def _answer_type(ans: dict) -> str:
    if ans["unanswerable"]:
        return "none"
    if ans["yes_no"] is not None:
        return "boolean"
    if ans["extractive_spans"] is not None and len(ans["extractive_spans"]) and any(s.strip() for s in ans["extractive_spans"]):
        return "extractive"
    return "abstractive"


def download_pdf(client: httpx.Client, arxiv_id: str) -> Path | None:
    dest = PAPERS_DIR / f"{arxiv_id}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        r = client.get(url, follow_redirects=True, timeout=60)
    except httpx.HTTPError as exc:
        print(f"    download failed: {exc}")
        return None
    time.sleep(3)  # arXiv asks for <= 1 request every 3 seconds
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        print(f"    not a PDF (HTTP {r.status_code})")
        return None
    dest.write_bytes(r.content)
    return dest


def page_texts(pdf_path: Path) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def locate(evidence: str, pages: list[str]) -> list[int]:
    """1-based pages that contain this evidence paragraph (single page or a page pair)."""
    ev = trigrams(evidence)
    if not ev:
        return []
    best, where = 0.0, []
    for i, txt in enumerate(pages):
        s = match_score(evidence, txt, ev_grams=ev, contain_only=True)
        if s > best:
            best, where = s, [i + 1]
    if best < LOCATE_THRESHOLD:  # paragraph may straddle a page break
        for i in range(len(pages) - 1):
            s = match_score(evidence, pages[i] + "\n" + pages[i + 1], ev_grams=ev, contain_only=True)
            if s > best:
                best, where = s, [i + 1, i + 2]
    return where if best >= LOCATE_THRESHOLD else []


def build(n_papers: int, seed: int) -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW_PARQUET)
    rows = list(df.itertuples(index=False))
    random.Random(seed).shuffle(rows)

    questions: list[dict] = []
    papers: list[dict] = []
    client = httpx.Client(headers={"User-Agent": "DocIntel-eval/1.0 (research benchmark)"})

    for row in rows:
        if len(papers) >= n_papers:
            break
        qas = row.qas
        if len(qas["question"]) < 2:
            continue
        print(f"[{len(papers) + 1}/{n_papers}] {row.id}  {row.title[:70]}")
        pdf = download_pdf(client, row.id)
        if pdf is None:
            continue
        try:
            pages = page_texts(pdf)
        except Exception as exc:  # noqa: BLE001
            print(f"    unreadable PDF: {exc}")
            continue
        if len(pages) > MAX_PAGES or sum(len(p) for p in pages) < 5_000:
            print(f"    skipped ({len(pages)} pages)")
            pdf.unlink(missing_ok=True)
            continue

        kept = 0
        for q, qid, ans_list in zip(qas["question"], qas["question_id"], qas["answers"]):
            annots = list(ans_list["answer"])
            n_unans = sum(1 for a in annots if a["unanswerable"])

            if n_unans == len(annots):
                # Every annotator agreed the paper doesn't answer it → abstention test.
                questions.append({
                    "qid": qid, "paper_id": row.id, "filename": pdf.name, "title": row.title,
                    "question": q.strip(), "answerable": False,
                    "answers": [], "answer_types": ["none"], "evidence_sets": [], "evidence_pages": [],
                })
                kept += 1
                continue
            if n_unans * 2 >= len(annots):
                continue  # annotators disagree on answerability → ambiguous, drop

            evidence_sets, answers, types, ev_pages, has_float = [], [], [], set(), False
            for a in annots:
                if a["unanswerable"]:
                    continue
                ev = [_clean_evidence(e) for e in a["evidence"] if _clean_evidence(e)]
                located = [(e, locate(e, pages)) for e in ev]
                located = [(e, p) for e, p in located if p]
                if ev and len(located) == len(ev):   # every evidence paragraph found in the PDF
                    evidence_sets.append([e for e, _ in located])
                    has_float |= any(e.strip().startswith("FLOAT SELECTED") for e in a["evidence"])
                    for _, p in located:
                        ev_pages.update(p)
                    answers.append(_answer_text(a))
                    types.append(_answer_type(a))
            if not evidence_sets:
                continue
            questions.append({
                "qid": qid, "paper_id": row.id, "filename": pdf.name, "title": row.title,
                "question": q.strip(), "answerable": True,
                "answers": answers, "answer_types": types,
                "evidence_sets": evidence_sets, "evidence_pages": sorted(ev_pages),
                "table_or_figure_evidence": has_float,
            })
            kept += 1

        if kept == 0:
            print("    no usable questions; skipped")
            continue
        papers.append({"paper_id": row.id, "filename": pdf.name, "title": row.title, "pages": len(pages)})
        print(f"    {len(pages)} pages, {kept} questions kept")

    client.close()
    out = {
        "source": "QASPER test split (allenai/qasper), arXiv PDFs",
        "seed": seed,
        "papers": papers,
        "questions": questions,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    n_ans = sum(q["answerable"] for q in questions)
    print(f"\nWrote {OUT_PATH}: {len(papers)} papers, {len(questions)} questions "
          f"({n_ans} answerable, {len(questions) - n_ans} unanswerable), "
          f"{sum(p['pages'] for p in papers)} pages")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build(args.papers, args.seed)
