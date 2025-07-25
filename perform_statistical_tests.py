#!/usr/bin/env python3
"""
Perform full statistical workflow and save all results (no plots):
  1) Shapiro–Wilk normality per metric×model
  2) Levene’s variance homogeneity per metric across all models
  3) Pairwise Mann–Whitney U per metric for all model pairs
  4) Kruskal–Wallis + all-pairs Dunn post-hoc (raw p-value and Bonferroni-adjusted p_adj)
Saves four CSVs: normality_shapiro.csv, variance_levene.csv,
mannwhitney_u.csv, and stats_allpairs_<first_metric>_to_<last_metric>.csv
"""
import pandas as pd
import scipy.stats as st
import scikit_posthocs as sp

# ────────────────────────────────────────────────────────────────────────
MERGED_CSV = "./merged_results.csv"
ALL_METRICS = [
    "rouge1", "rouge2", "rougel", "bleu",
    "faithfulness", "answer_correctness",
    "context_recall", "context_precision",
    "answer_relevancy", "Accuracy"
]

# ────────────────────────────────────────────────────────────────────────
def load_long_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Accuracy" not in df.columns and "answer_correctness" in df.columns:
        df["Accuracy"] = (df["answer_correctness"] > 0.5).astype(float)
    long = df.melt(
        id_vars=["model"],
        value_vars=[m for m in ALL_METRICS if m in df.columns],
        var_name="metric", value_name="score"
    ).dropna(subset=["score"])
    long["model"] = long["model"].astype(str)
    long["metric"] = long["metric"].astype(str)
    return long

# ────────────────────────────────────────────────────────────────────────
def run_shapiro(long_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for metric in ALL_METRICS:
        sub_m = long_df[long_df.metric == metric]
        for model in sub_m.model.unique():
            vals = sub_m[sub_m.model == model].score
            if len(vals) >= 3:
                W, p = st.shapiro(vals)
            else:
                W, p = None, None
            records.append({
                "metric": metric,
                "model": model,
                "shapiro_W": W,
                "shapiro_p": p
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv("normality_shapiro.csv", index=False)
    return df

# ────────────────────────────────────────────────────────────────────────
def run_levene(long_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for metric in ALL_METRICS:
        sub_m = long_df[long_df.metric == metric]
        groups = [grp.score.values for _, grp in sub_m.groupby("model")]
        if len(groups) > 1:
            stat, p = st.levene(*groups)
        else:
            stat, p = None, None
        records.append({"metric": metric, "levene_stat": stat, "levene_p": p})
    df = pd.DataFrame.from_records(records)
    df.to_csv("variance_levene.csv", index=False)
    return df

# ────────────────────────────────────────────────────────────────────────
def run_mannwhitney(long_df: pd.DataFrame) -> pd.DataFrame:
    from itertools import combinations
    records = []
    for metric in ALL_METRICS:
        sub_m = long_df[long_df.metric == metric]
        models = sub_m.model.unique()
        for a, b in combinations(models, 2):
            vals_a = sub_m[sub_m.model == a].score
            vals_b = sub_m[sub_m.model == b].score
            if len(vals_a) >= 1 and len(vals_b) >= 1:
                U, p = st.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            else:
                U, p = None, None
            records.append({
                "metric": metric,
                "model_a": a,
                "model_b": b,
                "U_stat": U,
                "mannwhitney_p": p
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv("mannwhitney_u.csv", index=False)
    return df

# ────────────────────────────────────────────────────────────────────────
def compute_all_dunn(long_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []
    for m in metrics:
        sub = long_df[long_df.metric == m]
        H, kw_p = st.kruskal(*[grp.score.values for _, grp in sub.groupby("model")])
        raw_dunn = pd.DataFrame()
        adj_dunn = pd.DataFrame()
        if kw_p < 0.05:
            raw_dunn = sp.posthoc_dunn(sub, val_col="score", group_col="model", p_adjust=None)
            adj_dunn = sp.posthoc_dunn(sub, val_col="score", group_col="model", p_adjust="bonferroni")
        idx = list(raw_dunn.index) if not raw_dunn.empty else list(sub.model.unique())
        for i, a in enumerate(idx):
            for b in idx[i+1:]:
                p_value = float(raw_dunn.loc[a, b]) if not raw_dunn.empty else None
                p_adj = float(adj_dunn.loc[a, b]) if not adj_dunn.empty else None
                records.append({
                    "metric": m,
                    "model_a": a,
                    "model_b": b,
                    "p_value": p_value,
                    "p_adj": p_adj,
                    "kw_p": kw_p
                })
    df = pd.DataFrame.from_records(records)
    df.to_csv(f"stats_allpairs_{metrics[0]}_to_{metrics[-1]}.csv", index=False)
    return df

# ────────────────────────────────────────────────────────────────────────
def main():
    long_df = load_long_df(MERGED_CSV)
    run_shapiro(long_df)
    run_levene(long_df)
    run_mannwhitney(long_df)
    compute_all_dunn(long_df, ALL_METRICS)
    print("✅ Generated: normality_shapiro.csv, variance_levene.csv, mannwhitney_u.csv, stats_allpairs_" +
          f"{ALL_METRICS[0]}_to_{ALL_METRICS[-1]}.csv")

if __name__ == "__main__":
    main()
