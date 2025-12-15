import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid", context="talk")

"""
@brief Load datasets used for poster figure generation.

This module reads four JSON files containing human and Perspective API
scores for two models (LLaMA and Qwen). The loaded variables are
`llama_h`, `qwen_h`, `llama_p`, and `qwen_p`.
"""

with open("../counterspeech/outputs/llama32_scored_dataset_human.json") as f:
    llama_h = json.load(f)
with open("../counterspeech/outputs/qwen25_scored_dataset_human.json") as f:
    qwen_h = json.load(f)

with open("../counterspeech/outputs/llama32_perspective_scores_final.json") as f:
    llama_p = json.load(f)
with open("../counterspeech/outputs/qwen25_perspective_scores_final.json") as f:
    qwen_p = json.load(f)

print("Loaded all four datasets.")

"""
@brief Convert JSON lists to pandas DataFrames and align rows by index.

`human_df` converts human-annotated JSON entries into a DataFrame with
explicit human score columns. `persp_df` converts Perspective API score
entries into a DataFrame with corresponding perspective score columns.
"""

def human_df(data, model):
    rows = []
    for idx, d in enumerate(data):
        s = d["scores"]
        rows.append({
            "idx": idx,
            "model": model,
            "original_comment": d.get("comment"),
            "counterspeech": d.get("counterspeech_english"),
            "lang": d.get("lang"),
            "toxicity_human": s["TOXICITY"],
            "insult_human": s["INSULT"],
            "profanity_human": s["PROFANITY"],
            "sevtox_human": s["SEVERE_TOXICITY"],
            "identatk_human": s["IDENTITY_ATTACK"]
        })
    return pd.DataFrame(rows)

def persp_df(data, model):
    rows = []
    for idx, d in enumerate(data):
        s = d["scores"]
        rows.append({
            "idx": idx,
            "model": model,
            "toxicity_persp": s["TOXICITY"],
            "insult_persp": s["INSULT"],
            "profanity_persp": s["PROFANITY"],
            "sevtox_persp": s["SEVERE_TOXICITY"],
            "identatk_persp": s["IDENTITY_ATTACK"]
        })
    return pd.DataFrame(rows)


df_llama_h = human_df(llama_h, "llama")
df_qwen_h  = human_df(qwen_h,  "qwen")

df_llama_p = persp_df(llama_p, "llama")
df_qwen_p  = persp_df(qwen_p,  "qwen")


"""@brief Merge human and Perspective DataFrames by index.

Merge `df_llama_h` with `df_llama_p` and `df_qwen_h` with
`df_qwen_p` on the `idx` and `model` columns so that rows align for
paired comparisons.
"""

df_llama = df_llama_h.merge(df_llama_p, on=["idx","model"])
df_qwen  = df_qwen_h.merge(df_qwen_p,  on=["idx","model"])

print("Merged llama rows:", len(df_llama))
print("Merged qwen rows:", len(df_qwen))

"""
@brief Normalize human scores from a 1–5 scale to 0–1.

The original human annotations are on a 1–5 scale; this step divides
those columns by 5. Perspective API scores are already in the 0–1 range.
"""

human_cols = ["toxicity_human","insult_human","profanity_human","sevtox_human","identatk_human"]
for col in human_cols:
    df_llama[col + "_norm"] = df_llama[col] / 5.0
    df_qwen[col + "_norm"]  = df_qwen[col]  / 5.0

"""
@brief Build an index-aligned paired DataFrame merging LLaMA and Qwen rows.

This constructs `df_paired` by truncating to the smaller dataset length
and merging on the `idx` column to allow paired comparisons.
"""

max_len = min(len(df_llama), len(df_qwen))
df_paired = df_llama.iloc[:max_len].merge(
    df_qwen.iloc[:max_len],
    left_on="idx",
    right_on="idx",
    suffixes=("_llama","_qwen")
)

print("Paired dataset size:", len(df_paired))

"""
@brief Compute perspective differences between models.

Creates `delta_persp_toxicity` (LLaMA − Qwen) and the mean perspective
toxicity for Bland–Altman style plots.
"""

import numpy as np

df_paired["delta_persp_toxicity"] = (
    df_paired["toxicity_persp_llama"] - df_paired["toxicity_persp_qwen"]
)

df_paired["mean_persp_toxicity"] = (
    df_paired["toxicity_persp_llama"] + df_paired["toxicity_persp_qwen"]
) / 2


"""
@brief Histogram and KDE of perspective toxicity differences.
"""

plt.figure(figsize=(12,6))
sns.histplot(
    df_paired["delta_persp_toxicity"],
    kde=True,
    bins=40,
    color="steelblue"
)
plt.axvline(0, color='red', linestyle='--', label='No Bias Line')
plt.title("Difference in Perspective API Toxicity (LLaMA − Qwen)")
plt.xlabel("Perspective Toxicity Difference")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig_perspective_diff_hist.png", dpi=300)
plt.close()


"""
@brief Scatter plot of perspective difference by comment index.
"""

plt.figure(figsize=(14,6))
plt.scatter(
    df_paired["idx"],
    df_paired["delta_persp_toxicity"],
    alpha=0.5,
    s=20,
    color="purple"
)
plt.axhline(0, color='red', linestyle='--')
plt.title("Perspective Toxicity Difference by Comment Index")
plt.xlabel("Comment Index")
plt.ylabel("LLaMA − Qwen Toxicity Difference")
plt.tight_layout()
plt.savefig("figures/fig_perspective_diff_scatter.png", dpi=300)
plt.close()


"""
@brief Bland–Altman plot for perspective toxicity comparison.
"""

plt.figure(figsize=(12,6))
plt.scatter(
    df_paired["mean_persp_toxicity"],
    df_paired["delta_persp_toxicity"],
    alpha=0.5,
    color="teal"
)

mean_diff = df_paired["delta_persp_toxicity"].mean()
std_diff = df_paired["delta_persp_toxicity"].std()

plt.axhline(mean_diff, color='red', linestyle='-', label=f"Mean Diff = {mean_diff:.3f}")
plt.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--', label="+1.96 SD")
plt.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--', label="-1.96 SD")

plt.xlabel("Mean Perspective Toxicity (LLaMA & Qwen)")
plt.ylabel("Difference (LLaMA − Qwen)")
plt.title("Bland–Altman Plot — Perspective Toxicity Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig_bland_altman_perspective.png", dpi=300)
plt.close()


"""
@brief Generate difference histograms for all five toxicity dimensions.
"""

dimensions = [
    ("toxicity_persp_llama","toxicity_persp_qwen","toxicity"),
    ("insult_persp_llama","insult_persp_qwen","insult"),
    ("profanity_persp_llama","profanity_persp_qwen","profanity"),
    ("sevtox_persp_llama","sevtox_persp_qwen","severe_toxicity"),
    ("identatk_persp_llama","identatk_persp_qwen","identity_attack")
]

for llama_col, qwen_col, name in dimensions:
    df_paired[f"delta_{name}"] = df_paired[llama_col] - df_paired[qwen_col]

    plt.figure(figsize=(12,6))
    sns.histplot(df_paired[f"delta_{name}"], kde=True, bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Difference in Perspective {name.replace('_',' ').title()} (LLaMA − Qwen)")
    plt.xlabel("Difference")
    plt.tight_layout()
    plt.savefig(f"figures/fig_perspective_diff_{name}.png", dpi=300)
    plt.close()

print("🎉 Perspective difference figures generated successfully!")

"""
@brief Ensure output folder exists for generated figures.
"""

os.makedirs("../counterspeech/outputs/figures", exist_ok=True)


"""
@brief Figure 1 — Distributional human toxicity comparison.
"""

plt.figure(figsize=(12,6))
sns.kdeplot(data=df_llama, x="toxicity_human", fill=True, label="LLaMA")
sns.kdeplot(data=df_qwen,  x="toxicity_human", fill=True, label="Qwen")
plt.title("Human Toxicity Distribution — LLaMA vs Qwen")
plt.legend()
plt.savefig("figures/fig1_distribution_human.png", dpi=300)
plt.close()


"""
@brief Figure 2 — Cultural bias box plot across full datasets.
"""

plt.figure(figsize=(12,6))
sns.boxplot(data=pd.concat([df_llama,df_qwen]), x="lang", y="toxicity_human", hue="model")
plt.title("Cultural Bias: Toxicity by Language")
plt.savefig("figures/fig2_cultural_bias.png", dpi=300)
plt.close()


"""
@brief Figure 3 — Paired scatter of LLaMA vs Qwen human toxicity.
"""

plt.figure(figsize=(8,8))
plt.scatter(
    df_paired["toxicity_human_norm_llama"],
    df_paired["toxicity_human_norm_qwen"],
    alpha=0.4
)
plt.xlabel("LLaMA (Human Toxicity, Norm)")
plt.ylabel("Qwen (Human Toxicity, Norm)")
plt.title("Paired Comparison — Human Toxicity")
plt.savefig("figures/fig3_paired_scatter.png", dpi=300)
plt.close()


"""
@brief Figure 4 — Perspective minus human toxicity (paired KDEs).
"""

df_paired["diff_llama"] = df_paired["toxicity_persp_llama"] - df_paired["toxicity_human_norm_llama"]
df_paired["diff_qwen"]  = df_paired["toxicity_persp_qwen"]  - df_paired["toxicity_human_norm_qwen"]

plt.figure(figsize=(12,6))
sns.kdeplot(df_paired["diff_llama"], fill=True, label="LLaMA")
sns.kdeplot(df_paired["diff_qwen"],  fill=True, label="Qwen")
plt.title("Perspective API Bias (Perspective − Human)")
plt.legend()
plt.savefig("figures/fig4_perspective_bias.png", dpi=300)
plt.close()


"""
@brief Figure 5 — Correlation heatmaps of human toxicity dimensions.
"""

plt.figure(figsize=(10,8))
sns.heatmap(df_llama[human_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("LLaMA — Human Toxicity Dimension Correlation")
plt.savefig("figures/fig5_llama_heatmap.png", dpi=300)
plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(df_qwen[human_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Qwen — Human Toxicity Dimension Correlation")
plt.savefig("figures/fig5_qwen_heatmap.png", dpi=300)
plt.close()


"""
@brief Figure 6 — Violin plots for toxicity dimensions by model.
"""

df_dim = pd.concat([df_llama.assign(model="LLaMA"), df_qwen.assign(model="Qwen")])

melt = df_dim.melt(
    id_vars=["model"],
    value_vars=[c + "_norm" for c in human_cols],
    var_name="dimension",
    value_name="score"
)

plt.figure(figsize=(14,6))
sns.violinplot(data=melt, x="dimension", y="score", hue="model", split=True)
plt.xticks(rotation=45)
plt.title("Human Toxicity Dimensions by Model")
plt.savefig("figures/fig6_dimensions.png", dpi=300)
plt.close()


print("🎉 All poster-ready figures saved in ./figures/")
