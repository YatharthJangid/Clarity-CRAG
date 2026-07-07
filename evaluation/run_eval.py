"""Golden-dataset evaluation: Naive RAG vs Corrective RAG (CRAG).

For every question in golden_dataset.json this script answers it two ways:

1. Naive RAG   - raw top-k retrieved chunks piped straight into the LLM,
                 no relevance grading, no filtering, no web fallback.
2. CRAG        - the full app.crag graph (retrieve -> grade -> refine/web
                 fallback -> generate), i.e. exactly what the API serves.

Both runs are scored with the same ragas metrics (faithfulness, answer
relevancy, context precision/recall, answer correctness) so the numbers are
directly comparable. Results are written to evaluation/results/.

Usage: python evaluation/run_eval.py   (run from repo root, needs real API
keys in backend/.env - see backend/.env.example)
"""
import json
import sys
import uuid
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))

from app.crag.graph import build_graph  # noqa: E402
from app.crag.llm import llm, primary_llm  # noqa: E402
from app.crag.prompts import answer_prompt  # noqa: E402
from app.crag.vectorstore import embeddings, get_retriever  # noqa: E402

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Accuracy (Answer Correctness)",
}


def naive_rag_answer(question: str) -> tuple[str, list[str]]:
    docs = get_retriever().invoke(question)
    contexts = [d.page_content for d in docs]
    context_str = "\n\n".join(contexts)
    out = (answer_prompt | llm).invoke({"question": question, "context": context_str})
    return out.content, (contexts or [""])


BLANK_STATE = {
    "docs": [],
    "good_docs": [],
    "verdict": "",
    "reason": "",
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
    "web_query": "",
    "web_docs": [],
    "answer": "",
}


def crag_answer(question: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    result = build_graph().invoke({"question": question, **BLANK_STATE}, config=config)
    contexts = result.get("kept_strips") or [result.get("refined_context", "")]
    return result["answer"], (contexts or [""])


def run_pipeline(golden, answer_fn, label):
    rows = []
    for i, item in enumerate(golden, 1):
        print(f"  [{label}] {i}/{len(golden)}: {item['question'][:60]}...")
        answer, contexts = answer_fn(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )
    return rows


def score(rows, name):
    dataset = EvaluationDataset.from_list(rows)
    # ragas wants a plain BaseChatModel, not the RunnableWithFallbacks that `llm`
    # is - so it gets the raw Gemini model directly rather than the fallback chain.
    result = evaluate(dataset=dataset, metrics=METRICS, llm=primary_llm, embeddings=embeddings)
    df = result.to_pandas()
    df.to_csv(RESULTS_DIR / f"per_question_{name}.csv", index=False)
    return df


def build_comparison_table(naive_df, crag_df) -> pd.DataFrame:
    rows = []
    for metric_key, label in METRIC_LABELS.items():
        naive_mean = naive_df[metric_key].mean()
        crag_mean = crag_df[metric_key].mean()
        improvement_pct = ((crag_mean - naive_mean) / naive_mean * 100) if naive_mean else float("nan")
        rows.append(
            {
                "Metric": label,
                "Naive RAG": round(naive_mean, 3),
                "Corrective RAG (CRAG)": round(crag_mean, 3),
                "Improvement": f"{improvement_pct:+.1f}%",
            }
        )
    return pd.DataFrame(rows)


def save_chart(comparison: pd.DataFrame):
    metrics = comparison["Metric"]
    naive_vals = comparison["Naive RAG"]
    crag_vals = comparison["Corrective RAG (CRAG)"]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.bar([i - width / 2 for i in x], naive_vals, width, label="Naive RAG", color="#94a3b8")
    ax.bar([i + width / 2 for i in x], crag_vals, width, label="Corrective RAG (CRAG)", color="#6366f1")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Naive RAG vs Corrective RAG - ragas metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, (nv, cv) in enumerate(zip(naive_vals, crag_vals)):
        ax.text(i - width / 2, nv + 0.02, f"{nv:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, cv + 0.02, f"{cv:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics_chart.png", dpi=180)
    print(f"Saved chart to {RESULTS_DIR / 'metrics_chart.png'}")


def main():
    golden = json.loads((Path(__file__).resolve().parent / "golden_dataset.json").read_text())

    print(f"Running Naive RAG over {len(golden)} questions...")
    naive_rows = run_pipeline(golden, naive_rag_answer, "naive")

    print(f"Running CRAG over {len(golden)} questions...")
    crag_rows = run_pipeline(golden, crag_answer, "crag")

    print("Scoring Naive RAG with ragas...")
    naive_df = score(naive_rows, "naive")

    print("Scoring CRAG with ragas...")
    crag_df = score(crag_rows, "crag")

    comparison = build_comparison_table(naive_df, crag_df)
    comparison.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    (RESULTS_DIR / "comparison_table.md").write_text(comparison.to_markdown(index=False))

    save_chart(comparison)

    print("\n" + comparison.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
