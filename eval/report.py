"""
Turn the benchmark's cached outputs into metrics.

Writes:
  eval/results/summary.json   — every number + per-question detail (feeds the UI)
  eval/results/RESULTS.md     — human-readable tables

Statistics: 95% CIs from 10,000 paired bootstrap resamples over questions;
p-values are two-sided paired-bootstrap tests against the standard-RAG baseline.
"""

from __future__ import annotations

import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import textmatch as tm  # noqa: E402
from bench import CHUNKSETS, CONFIGS, DENSE_MODEL, FINAL_K, GEN_CONFIGS, RESULTS, WORK, load_chunks, load_dataset, _read  # noqa: E402

BASELINE = "naive"
RNG = np.random.default_rng(0)
N_BOOT = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def ci(values) -> dict:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    idx = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
    boots = a[idx].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": int(len(a))}


def paired(a, b) -> dict:
    """Difference a - b with bootstrap CI and two-sided p-value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "p": float(max(p, 1 / N_BOOT))}


# ─────────────────────────────────────────────────────────────────────────────
# Answer-level helpers
# ─────────────────────────────────────────────────────────────────────────────
_CITE = re.compile(r"\[[^\[\]]{1,60}\]")


def cited_pages(answer: str, contexts: list[dict]) -> list[set[int]]:
    """Pages behind each numbered citation [n] → the n-th retrieved chunk's page span."""
    from rag_engine import cited_block_ids

    out = []
    for n in sorted(cited_block_ids(answer)):
        if 1 <= n <= len(contexts):
            c = contexts[n - 1]
            out.append(set(range(c["page_start"], c["page_end"] + 1)))
    return out


def _norm_answer(s: str) -> list[str]:
    s = _CITE.sub(" ", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    """SQuAD/QASPER-style answer F1."""
    p, g = _norm_answer(pred), _norm_answer(gold)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return float(p == g)
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
def retrieval_metrics(ds: dict) -> tuple[dict, dict]:
    retrieval = _read(WORK / "retrieval.json", {})
    answerable = [q for q in ds["questions"] if q["answerable"]]

    # Pre-compute trigram sets once.
    chunk_grams: dict[str, set] = {}
    chunks_by_set_paper: dict[str, dict[str, list[str]]] = {}
    for name in CHUNKSETS:
        per_paper = defaultdict(list)
        for c in load_chunks(name):
            chunk_grams[c["id"]] = tm.trigrams(c["text"])
            per_paper[c["paper_id"]].append(c["id"])
        chunks_by_set_paper[name] = per_paper

    per_q: dict[str, dict] = defaultdict(dict)
    metrics: dict[str, dict] = {}
    for cname, cfg in CONFIGS.items():
        if cname not in retrieval:
            continue
        rows = defaultdict(list)
        for q in answerable:
            top = retrieval[cname].get(q["qid"])
            if top is None:
                continue
            ev_sets = [[tm.trigrams(e) for e in s] for s in q["evidence_sets"]]
            ev_all = [g for s in ev_sets for g in s]
            rel = [any(tm.is_relevant(chunk_grams[c["id"]], g) for g in ev_all) for c in top]
            first = next((i for i, r in enumerate(rel) if r), None)
            recall = max(
                sum(any(tm.is_relevant(chunk_grams[c["id"]], g) for c in top) for g in s) / len(s)
                for s in ev_sets)
            paper_chunks = chunks_by_set_paper[cfg["chunks"]][q["paper_id"]]
            n_rel_total = sum(any(tm.is_relevant(chunk_grams[cid], g) for g in ev_all) for cid in paper_chunks)
            dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(FINAL_K, n_rel_total)))
            vals = {
                "hit@1": float(any(rel[:1])), "hit@3": float(any(rel[:3])), "hit@5": float(any(rel[:5])),
                "mrr@5": 1.0 / (first + 1) if first is not None else 0.0,
                "recall@5": recall,
                "ndcg@5": dcg / idcg if idcg else 0.0,
                "evidence_recoverable": float(n_rel_total > 0),
                "context_chars": float(sum(len(c["text"]) for c in top)),
            }
            for k, v in vals.items():
                rows[k].append(v)
            rows["_qid"].append(q["qid"])
            per_q[q["qid"]][cname] = {"rel": rel, "hit@5": vals["hit@5"], "mrr": vals["mrr@5"],
                                      "pages": [c["page_start"] for c in top]}
        qids = rows.pop("_qid", [])
        metrics[cname] = {"label": cfg["label"], "values": dict(rows), "qids": qids}
    return metrics, per_q


def parse_fidelity(ds: dict) -> dict:
    """
    For every gold evidence paragraph (>=80% of its trigrams must match):
      text_recovered — present anywhere in the parsed document (parser quality)
      chunk_intact   — present inside ONE chunk (parser + chunker quality; a
                       paragraph split across chunks can't be retrieved whole)
    """
    out = {}
    for name in CHUNKSETS:
        per_paper = defaultdict(list)
        for c in load_chunks(name):
            per_paper[c["paper_id"]].append(tm.trigrams(c["text"]))
        full = {pid: set().union(*gs) for pid, gs in per_paper.items()}
        text_rec, intact = [], []
        for q in ds["questions"]:
            if not q["answerable"]:
                continue
            for s in q["evidence_sets"]:
                for e in s:
                    g = tm.trigrams(e)
                    if not g:
                        continue
                    text_rec.append(float(len(g & full[q["paper_id"]]) / len(g) >= 0.8))
                    best = max(len(g & cg) / len(g) for cg in per_paper[q["paper_id"]])
                    intact.append(float(best >= 0.8))
        out[name] = {"text_recovered": ci(text_rec), "chunk_intact": ci(intact)}
    return out


def answer_metrics(ds: dict) -> tuple[dict, dict]:
    gens = _read(WORK / "generation.json", {})
    judg = _read(WORK / "judgments.json", {})
    retrieval = _read(WORK / "retrieval.json", {})
    qmap = {q["qid"]: q for q in ds["questions"]}
    qids = [qid for qid in gens if qid in judg]
    rows = {cfg: defaultdict(list) for cfg in GEN_CONFIGS}
    per_q = {}
    wins = Counter()
    for qid in qids:
        q, g, j = qmap[qid], gens[qid], judg[qid]
        per_q[qid] = {"model": g["model"], "judge": j["judge"]}
        for cfg in GEN_CONFIGS:
            ans = g["answers"][cfg]["answer"]
            grade = j[cfg]
            r = rows[cfg]
            per_q[qid][cfg] = {"answer": ans, "grade": grade}
            r["faithful"].append(float(grade["faithful"]))
            if q["answerable"]:
                r["correctness"].append(SCORE[grade["correctness"]])
                r["accuracy"].append(float(grade["correctness"] == "correct"))
                r["token_f1"].append(max(token_f1(ans, ref) for ref in q["answers"]))
                r["false_abstention"].append(float(grade["abstained"]))
                cites = cited_pages(ans, retrieval[cfg][qid])
                if not grade["abstained"]:
                    r["has_citation"].append(float(bool(cites)))
                    ev = set(q["evidence_pages"])
                    if cites:
                        # Per answer: share of its citations that land on an evidence page.
                        r["citation_precision"].append(sum(bool(p & ev) for p in cites) / len(cites))
                        r["answer_cites_evidence_page"].append(float(any(p & ev for p in cites)))
            else:
                r["correct_abstention"].append(float(grade["abstained"]))
        if q["answerable"]:
            a, b = SCORE[j["docintel"]["correctness"]], SCORE[j["naive"]["correctness"]]
            wins["docintel" if a > b else "naive" if b > a else "tie"] += 1
    return {"n_pairs": len(qids), "rows": {c: dict(r) for c, r in rows.items()}, "wins": dict(wins),
            "generators": dict(Counter(gens[q]["model"] for q in qids)),
            "judges": dict(Counter(judg[q]["judge"] for q in qids))}, per_q


def breakdown(ds: dict, per_q_ret: dict, per_q_ans: dict) -> dict:
    groups = defaultdict(list)
    for q in ds["questions"]:
        if not q["answerable"]:
            continue
        groups[f"type:{q['answer_types'][0]}"].append(q["qid"])
        groups["evidence:table_or_figure" if q["table_or_figure_evidence"] else "evidence:text"].append(q["qid"])
    out = {}
    for gname, qids in groups.items():
        entry = {"n": len(qids)}
        for cfg in (BASELINE, "docintel"):
            hits = [per_q_ret[qid][cfg]["hit@5"] for qid in qids if cfg in per_q_ret.get(qid, {})]
            entry[f"{cfg}_hit@5"] = float(np.mean(hits)) if hits else None
            corr = [SCORE[per_q_ans[qid][cfg]["grade"]["correctness"]] for qid in qids if qid in per_q_ans]
            entry[f"{cfg}_correctness"] = float(np.mean(corr)) if corr else None
            entry["n_judged"] = len(corr)
        out[gname] = entry
    return out


def ingestion_stats() -> dict:
    out = {}
    for name in CHUNKSETS:
        st = _read(WORK / f"chunks_{name}.json", {"chunks": [], "timing": {}})
        secs = sum(t["seconds"] for t in st["timing"].values())
        pages = sum(t["pages"] for t in st["timing"].values())
        lens = [len(c["text"]) for c in st["chunks"]]
        out[name] = {"papers": len(st["timing"]), "pages": pages, "chunks": len(lens),
                     "sec_per_page": secs / pages if pages else None,
                     "median_chunk_chars": float(np.median(lens)) if lens else None}
    return out


def fmt_pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    ds = load_dataset()
    ret, per_q_ret = retrieval_metrics(ds)
    ans, per_q_ans = answer_metrics(ds)

    retrieval_summary = {}
    for cname, m in ret.items():
        vals = m["values"]
        entry = {"label": m["label"], "metrics": {k: ci(v) for k, v in vals.items()}}
        if cname != BASELINE and BASELINE in ret:
            # Pair on the questions both configs answered (a resumed run may be partial).
            base_ix = {qid: i for i, qid in enumerate(ret[BASELINE]["qids"])}
            pairs = [(i, base_ix[qid]) for i, qid in enumerate(m["qids"]) if qid in base_ix]
            base = ret[BASELINE]["values"]
            entry["vs_baseline"] = {
                k: paired([vals[k][i] for i, _ in pairs], [base[k][j] for _, j in pairs])
                for k in ("hit@1", "hit@5", "mrr@5", "recall@5", "ndcg@5")}
        retrieval_summary[cname] = entry

    answer_summary = {"n_pairs": ans["n_pairs"], "wins": ans["wins"], "generators": ans["generators"],
                      "judges": ans["judges"], "configs": {}}
    for cfg in GEN_CONFIGS:
        r = ans["rows"][cfg]
        answer_summary["configs"][cfg] = {k: ci(v) for k, v in r.items()}
    # Where does the answer gap come from? Split by whether standard RAG's top-5
    # contained the annotated evidence.
    split = {"hit": [], "miss": []}
    for qid, pq in per_q_ans.items():
        q = next(x for x in ds["questions"] if x["qid"] == qid)
        if not q["answerable"] or BASELINE not in per_q_ret.get(qid, {}):
            continue
        key = "hit" if per_q_ret[qid][BASELINE]["hit@5"] else "miss"
        split[key].append((SCORE[pq[BASELINE]["grade"]["correctness"]],
                           SCORE[pq["docintel"]["grade"]["correctness"]]))
    answer_summary["by_baseline_retrieval"] = {
        k: {"n": len(v), "naive": float(np.mean([x[0] for x in v])) if v else None,
            "docintel": float(np.mean([x[1] for x in v])) if v else None}
        for k, v in split.items()}
    if ans["n_pairs"]:
        a, b = ans["rows"]["docintel"], ans["rows"][BASELINE]
        answer_summary["vs_baseline"] = {k: paired(a[k], b[k]) for k in
                                         ("correctness", "accuracy", "token_f1", "faithful")
                                         if len(a.get(k, [])) == len(b.get(k, [])) and a.get(k)}

    latency = _read(WORK / "latency.json", {})
    lat_summary = {}
    if latency.get("samples"):
        for k in latency["samples"][0]:
            lat_summary[k] = float(np.median([s[k] for s in latency["samples"]]))
    for cfg, ms in latency.get("generation_ms", {}).items():
        if ms:
            lat_summary[f"generation_ms:{cfg}"] = float(np.median(ms))

    qmap = {q["qid"]: q for q in ds["questions"]}
    papers = {p["paper_id"]: p for p in ds["papers"]}
    retrieval = _read(WORK / "retrieval.json", {})
    per_question = []
    for q in ds["questions"]:
        item = {"qid": q["qid"], "question": q["question"], "paper_id": q["paper_id"],
                "title": papers[q["paper_id"]]["title"], "answerable": q["answerable"],
                "answers": q["answers"][:3], "answer_type": q["answer_types"][0],
                "evidence_pages": q["evidence_pages"],
                "evidence": (q["evidence_sets"][0][0][:600] if q["evidence_sets"] else ""),
                "retrieval": per_q_ret.get(q["qid"], {}),
                "generation": per_q_ans.get(q["qid"])}
        for cfg in GEN_CONFIGS:
            top = retrieval.get(cfg, {}).get(q["qid"], [])
            item.setdefault("contexts", {})[cfg] = [
                {"page": c["page_start"], "section": c.get("section", ""), "text": c["text"][:420]}
                for c in top]
        per_question.append(item)

    summary = {
        "dataset": {
            "name": "QASPER (test split)", "papers": len(ds["papers"]),
            "pages": sum(p["pages"] for p in ds["papers"]),
            "questions": len(ds["questions"]),
            "answerable": sum(q["answerable"] for q in ds["questions"]),
            "unanswerable": sum(not q["answerable"] for q in ds["questions"]),
            "answer_types": dict(Counter(q["answer_types"][0] for q in ds["questions"] if q["answerable"])),
            "table_or_figure_questions": sum(q.get("table_or_figure_evidence", False) for q in ds["questions"]),
        },
        "setup": {"embedding_model": DENSE_MODEL, "final_k": FINAL_K, "baseline": BASELINE,
                  "configs": {k: v["label"] for k, v in CONFIGS.items()}},
        "retrieval": retrieval_summary,
        "parse_fidelity": parse_fidelity(ds),
        "answers": answer_summary,
        "breakdown": breakdown(ds, per_q_ret, per_q_ans),
        "ingestion": ingestion_stats(),
        "latency_ms": lat_summary,
        "per_question": per_question,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary)
    print(f"wrote {RESULTS / 'summary.json'} and RESULTS.md")


def write_markdown(s: dict) -> None:
    d = s["dataset"]
    L = [f"# DocIntel benchmark results\n",
         f"**Dataset:** {d['name']}: {d['papers']} arXiv papers ({d['pages']} pages), "
         f"{d['questions']} questions ({d['answerable']} answerable with human-annotated evidence, "
         f"{d['unanswerable']} unanswerable).  ",
         f"**Embedding (all systems):** `{s['setup']['embedding_model']}`, top-{s['setup']['final_k']} context.\n",
         "## Retrieval (does the answer-bearing passage reach the LLM?)\n",
         "| System | Hit@1 | Hit@5 | MRR@5 | Evidence recall@5 | nDCG@5 | Avg context chars |",
         "|---|---|---|---|---|---|---|"]
    for cname, e in s["retrieval"].items():
        m = e["metrics"]
        L.append(f"| {e['label']} | {fmt_pct(m['hit@1']['mean'])} | {fmt_pct(m['hit@5']['mean'])} | "
                 f"{m['mrr@5']['mean']:.3f} | {fmt_pct(m['recall@5']['mean'])} | {m['ndcg@5']['mean']:.3f} | "
                 f"{m['context_chars']['mean']:.0f} |")
    if "docintel" in s["retrieval"] and "vs_baseline" in s["retrieval"]["docintel"]:
        v = s["retrieval"]["docintel"]["vs_baseline"]
        L.append("\nDocIntel vs standard RAG (paired bootstrap, 95% CI):\n")
        for k, x in v.items():
            L.append(f"- {k}: {x['diff'] * 100:+.1f} pts [{x['lo'] * 100:+.1f}, {x['hi'] * 100:+.1f}], p={x['p']:.4f}")
    n_ev = next(iter(s["parse_fidelity"].values()))["chunk_intact"]["n"]
    L.append(f"\n## Parsing & chunking fidelity ({n_ev} gold evidence paragraphs)\n")
    L.append("| Chunkset | Recovered anywhere in parsed text | Intact inside one chunk |\n|---|---|---|")
    for name, x in s["parse_fidelity"].items():
        L.append(f"| {name} | {fmt_pct(x['text_recovered']['mean'])} | {fmt_pct(x['chunk_intact']['mean'])} |")
    a = s["answers"]
    if a["n_pairs"]:
        L += [f"\n## End-to-end answers ({a['n_pairs']} paired questions, blinded LLM judge)\n",
              f"Generators: {a['generators']} · Judges: {a['judges']}\n",
              "| System | Judge correctness | Strict accuracy | Answer F1 | Faithful | Citation precision | Correct abstention (unanswerable) | False abstention |",
              "|---|---|---|---|---|---|---|---|"]
        for cfg, m in a["configs"].items():
            def g(k):
                return fmt_pct(m[k]["mean"]) if k in m else "—"
            L.append(f"| {'DocIntel' if cfg == 'docintel' else 'Standard RAG'} | {g('correctness')} | {g('accuracy')} | {g('token_f1')} | "
                     f"{g('faithful')} | {g('citation_precision')} | {g('correct_abstention')} | {g('false_abstention')} |")
        L.append(f"\nHead-to-head (answerable): {a['wins']}")
        if a.get("vs_baseline", {}).get("correctness"):
            v = a["vs_baseline"]["correctness"]
            L.append(f"\nJudge-correctness difference: {v['diff'] * 100:+.1f} pts "
                     f"[{v['lo'] * 100:+.1f}, {v['hi'] * 100:+.1f}], p={v['p']:.3f} (paired bootstrap).")
        br = a.get("by_baseline_retrieval", {})
        if br.get("miss", {}).get("n"):
            L.append(f"\nWhen standard RAG's top-5 **missed** the evidence (n={br['miss']['n']}): "
                     f"standard {fmt_pct(br['miss']['naive'])} vs DocIntel {fmt_pct(br['miss']['docintel'])}. "
                     f"When it **found** it (n={br['hit']['n']}): {fmt_pct(br['hit']['naive'])} vs "
                     f"{fmt_pct(br['hit']['docintel'])}.")
    L.append("\n## Ingestion\n")
    for name, x in s["ingestion"].items():
        spp = f"{x['sec_per_page']:.2f}s/page" if x["sec_per_page"] else "—"
        L.append(f"- {name}: {x['chunks']} chunks, median {x['median_chunk_chars']:.0f} chars, {spp}")
    if s["latency_ms"]:
        L.append("\n## Latency (median ms)\n")
        for k, v in s["latency_ms"].items():
            L.append(f"- {k}: {v:.0f}")
    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
