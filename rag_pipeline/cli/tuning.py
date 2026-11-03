"""Coarse-to-fine search for weighted hybrid retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from indexing.embedder import create_embedder


# Each coarse grid includes low, medium, and high values. The fine pass
# searches the neighborhood within one grid step of the best coarse trial.
_GRID = {
    "k1": [0.6, 1.2, 1.8, 2.4],
    "b": [0.25, 0.50, 0.75, 1.00],
    "fusion_weight": [0.0, 0.05, 0.15, 0.35, 0.60],
}


def tune_bm25(benchmark, args) -> None:
    import optuna

    config = benchmark.config
    if not config.bm25.enabled or config.bm25.fusion_method != "weighted":
        raise ValueError("Tuning requires BM25 with weighted fusion")

    questions = benchmark._load_questions(args.questions, args.max_questions)
    if not questions:
        raise ValueError("Tuning requires at least one labeled question")
    relevant_ids = [{str(law_id) for law_id in q["relevant_laws"]} for q in questions]

    store, bm25, _ = benchmark._index(args.id_map, args.rechunk)
    if bm25 is None or not bm25.is_built() or not bm25.has_benchmark_metadata():
        raise ValueError("Tuning requires a BM25 index with benchmark IDs")
    embedder = create_embedder(config)
    queries, embeddings = benchmark._prepare_queries(questions, embedder)
    retriever = benchmark._retriever(embedder, store, bm25)

    def evaluate(params):
        bm25.set_scoring(params["k1"], params["b"], config.bm25.top_k)
        retriever.bm25_weight = params["fusion_weight"]
        hits = 0
        for query, embedding, relevant in zip(queries, embeddings, relevant_ids, strict=True):
            results = retriever.retrieve(query, top_k=10, query_embedding=embedding)
            ranked_ids = {result.metadata.get("benchmark_id", "") for result in results}
            hits += bool(ranked_ids & relevant)
        return hits / len(questions)

    def coarse_objective(trial):
        return evaluate({
            name: trial.suggest_categorical(name, values)
            for name, values in _GRID.items()
        })

    coarse = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.GridSampler(_GRID, seed=42)
    )
    coarse.optimize(coarse_objective)

    bounds = {}
    for name, values in _GRID.items():
        index = values.index(coarse.best_params[name])
        bounds[name] = (values[max(0, index - 1)], values[min(len(values) - 1, index + 1)])

    def fine_objective(trial):
        return evaluate({
            name: trial.suggest_float(name, lower, upper)
            for name, (lower, upper) in bounds.items()
        })

    fine = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    fine.enqueue_trial(coarse.best_params)
    fine.optimize(fine_objective, n_trials=args.fine_trials)
    winner = fine if fine.best_value > coarse.best_value else coarse

    output = Path("benchmarks/results/bm25_tuning.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "objective": "recall_at_10",
        "total_questions": len(questions),
        "reranker_enabled": False,
        "coarse": {"trials": len(coarse.trials), "best_value": coarse.best_value},
        "fine": {"trials": len(fine.trials), "best_value": fine.best_value},
        "best_value": winner.best_value,
        "bm25": winner.best_params,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved tuning results to {output}")
