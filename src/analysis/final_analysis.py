import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# =========================================================
# 1. LOAD ALL FOUR FILES
# =========================================================
files = {
    "LLaMA_HUMAN": "llama32_scored_dataset_human.json",
    "Qwen_HUMAN":  "qwen25_scored_dataset_human.json",
    "LLaMA_PERSP": "llama32_perspective_scores_final.json",
    "Qwen_PERSP":  "qwen25_perspective_scores_final.json"
}

data = {}
for key, path in files.items():
    with open(path, "r", encoding="utf-8") as f:
        data[key] = json.load(f)

# =========================================================
# 2. CONVERT JSON → DATAFRAME
# =========================================================
tox_cols = ["TOXICITY", "SEVERE_TOXICITY", "INSULT", "PROFANITY", "IDENTITY_ATTACK"]

def to_df(entries, model, score_type):
    rows = []
    for idx, entry in enumerate(entries):
        scores = entry["scores"]
        row = {col: scores[col] for col in tox_cols}
        row["id"] = idx
        row["model"] = model
        row["score_type"] = score_type
        rows.append(row)
    return pd.DataFrame(rows)

df_llama_h = to_df(data["LLaMA_HUMAN"], "LLaMA", "HUMAN")
df_qwen_h  = to_df(data["Qwen_HUMAN"],  "Qwen",  "HUMAN")
df_llama_p = to_df(data["LLaMA_PERSP"], "LLaMA", "PERSPECTIVE")
df_qwen_p  = to_df(data["Qwen_PERSP"],  "Qwen",  "PERSPECTIVE")

df_human = pd.concat([df_llama_h, df_qwen_h])
df_persp = pd.concat([df_llama_p, df_qwen_p])
df_all = pd.concat([df_human, df_persp])

# =========================================================
# 3. NORMALIZE HUMAN SCORES TO 0–1 RANGE
# =========================================================
# original scores are 1–5 → normalize to 0–1
def normalize_df(df):
    df_norm = df.copy()
    for col in tox_cols:
        df_norm[col] = (df[col] - 1) / 4
    return df_norm

df_llama_h_norm = normalize_df(df_llama_h)
df_qwen_h_norm  = normalize_df(df_qwen_h)
df_human_norm = normalize_df(df_human)

# =========================================================
# 4. PLOTTING HELPERS
# =========================================================
def savefig(title):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_").lower() + ".png", dpi=300)
    plt.show()

# =========================================================
# 5. ADVANCED COMPARISONS
# =========================================================

# ---------------------------------------------------------
# A. LLaMA(HUMAN_norm) vs Qwen(HUMAN_norm)
# ---------------------------------------------------------
plt.figure(figsize=(7,6))
sns.scatterplot(
    x=df_llama_h_norm["TOXICITY"],
    y=df_qwen_h_norm["TOXICITY"],
    alpha=0.6
)
plt.plot([0,1], [0,1], "--", color="black")
plt.xlabel("LLaMA Human (normalized)")
plt.ylabel("Qwen Human (normalized)")
savefig("figs/Scatter_LLaMA_vs_Qwen_HUMAN_Normalised")

# ---------------------------------------------------------
# B. LLaMA(PERSP) vs Qwen(PERSP)
# ---------------------------------------------------------
plt.figure(figsize=(7,6))
sns.scatterplot(
    x=df_llama_p["TOXICITY"],
    y=df_qwen_p["TOXICITY"],
    alpha=0.6
)
plt.plot([0,1],[0,1], "--", color="black")
plt.xlabel("LLaMA Perspective")
plt.ylabel("Qwen Perspective")
savefig("figs/Scatter_LLaMA_vs_Qwen_Perspective")

# ---------------------------------------------------------
# C. HUMAN_norm vs PERSPECTIVE (per model)
# ---------------------------------------------------------
def compare_human_persp(df_h, df_p, model):
    merged = df_h.merge(df_p, on="id", suffixes=("_H", "_P"))
    plt.figure(figsize=(7,6))
    sns.scatterplot(
        x=merged["TOXICITY_H"],
        y=merged["TOXICITY_P"],
        alpha=0.6
    )
    plt.plot([0,1],[0,1],"--",color="black")
    plt.xlabel("Human (normalized)")
    plt.ylabel("Perspective API")
    savefig(f"figs/{model}_Human_vs_Perspective")

compare_human_persp(df_llama_h_norm, df_llama_p, "LLaMA")
compare_human_persp(df_qwen_h_norm,  df_qwen_p,  "Qwen")

# ---------------------------------------------------------
# D. HUMAN-PERSP DIFFERENCE HISTOGRAMS
# ---------------------------------------------------------
def diff_hist(df_h, df_p, model):
    merged = df_h.merge(df_p, on="id", suffixes=("_H", "_P"))
    diff = merged["TOXICITY_H"] - merged["TOXICITY_P"]
    plt.figure(figsize=(8,4))
    sns.histplot(diff, kde=True, color="purple")
    plt.xlabel("Human_norm - Perspective Score")
    savefig(f"figs/{model}_HumanMinusPerspective")

diff_hist(df_llama_h_norm, df_llama_p, "LLaMA")
diff_hist(df_qwen_h_norm,  df_qwen_p,  "Qwen")

# ---------------------------------------------------------
# E. CALIBRATION CURVES (Perspective vs Human_norm)
# ---------------------------------------------------------
def calibration(df_h, df_p, model):
    merged = df_h.merge(df_p, on="id", suffixes=("_H","_P"))
    bins = np.linspace(0, 1, 6)
    merged["bucket"] = pd.cut(merged["TOXICITY_P"], bins)

    calib = merged.groupby("bucket")[["TOXICITY_P","TOXICITY_H"]].mean()
    calib.plot(figsize=(8,4), marker="o")
    plt.ylabel("Score")
    savefig(f"figs/{model}_CalibrationCurve")

calibration(df_llama_h_norm, df_llama_p, "LLaMA")
calibration(df_qwen_h_norm,  df_qwen_p,  "Qwen")

# ---------------------------------------------------------
# F. BLAND–ALTMAN PLOTS
# ---------------------------------------------------------
def bland_altman(df_h, df_p, model):
    merged = df_h.merge(df_p, on="id", suffixes=("_H","_P"))
    mean = (merged["TOXICITY_H"] + merged["TOXICITY_P"]) / 2
    diff = merged["TOXICITY_H"] - merged["TOXICITY_P"]

    plt.figure(figsize=(7,6))
    plt.scatter(mean, diff, alpha=0.6)
    plt.axhline(diff.mean(), color="red")
    plt.xlabel("Mean Score (Human_norm + Perspective)/2")
    plt.ylabel("Difference (Human_norm - Perspective)")
    savefig(f"figs/{model}_BlandAltman")

bland_altman(df_llama_h_norm, df_llama_p, "LLaMA")
bland_altman(df_qwen_h_norm,  df_qwen_p,  "Qwen")

# ---------------------------------------------------------
# G. DISAGREEMENT HEATMAP
# ---------------------------------------------------------
def disagreement_heatmap(df_h, df_p, model):
    merged = df_h.merge(df_p, on="id", suffixes=("_H","_P"))
    diff = pd.DataFrame({
        "TOX": merged["TOXICITY_H"] - merged["TOXICITY_P"],
        "INS": merged["INSULT_H"] - merged["INSULT_P"],
        "PROF": merged["PROFANITY_H"] - merged["PROFANITY_P"],
        "ID": merged["IDENTITY_ATTACK_H"] - merged["IDENTITY_ATTACK_P"]
    })
    plt.figure(figsize=(6,5))
    sns.heatmap(diff.corr(), annot=True, cmap="coolwarm")
    savefig(f"figs/{model}_HumanPerspective_DisagreementHeatmap")

disagreement_heatmap(df_llama_h_norm, df_llama_p, "LLaMA")
disagreement_heatmap(df_qwen_h_norm,  df_qwen_p,  "Qwen")

# ---------------------------------------------------------
# H. MODEL BIAS DRIFT ANALYSIS
# ---------------------------------------------------------
bias_df = pd.DataFrame({
    "model": ["LLaMA","Qwen"],
    "Human_avg": [df_llama_h_norm["TOXICITY"].mean(), df_qwen_h_norm["TOXICITY"].mean()],
    "Persp_avg": [df_llama_p["TOXICITY"].mean(), df_qwen_p["TOXICITY"].mean()]
})
bias_df["PerspectivePenalty"] = bias_df["Persp_avg"] - bias_df["Human_avg"]

bias_df.plot(x="model", y=["Human_avg","Persp_avg","PerspectivePenalty"], kind="bar")
savefig("figs/Perspective_Bias_Drift")
