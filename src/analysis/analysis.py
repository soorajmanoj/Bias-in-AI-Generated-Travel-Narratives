import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from wordcloud import WordCloud

# STYLE
sns.set_palette("coolwarm")
sns.set_style("white")
plt.rcParams["axes.grid"] = False

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
input_file = "../counterspeech/outputs/llama32_perspective_scores_final.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for item in data:
    scores = item.get("perspective_scores", {})
    rows.append({
        "comment": item.get("comment", ""),
        "language": item.get("lang", "").lower(),   # <---- IMPORTANT FIX
        "toxicity": scores.get("TOXICITY"),
        "severe_toxicity": scores.get("SEVERE_TOXICITY"),
        "insult": scores.get("INSULT"),
        "profanity": scores.get("PROFANITY"),
        "identity_attack": scores.get("IDENTITY_ATTACK"),
    })

df = pd.DataFrame(rows)

# Rename rom_hindi → hinglish
df["language"] = df["language"].replace({"rom_hindi": "hinglish"})

# Keep ONLY hinglish and english
df = df[df["language"].isin(["hinglish", "english"])]

print("\n🧾 Comment counts:")
print(df["language"].value_counts())


# -----------------------------------------------------------
# 1. Mean Toxicity with 95% CI
# -----------------------------------------------------------
def mean_ci(series):
    mean = series.mean()
    ci = stats.t.interval(0.95, len(series)-1, loc=mean, scale=stats.sem(series)) \
        if len(series) > 1 else (mean, mean)
    return mean, *ci

summary = []
for lang, group in df.groupby("language"):
    mean, ci_low, ci_high = mean_ci(group["toxicity"])
    summary.append({"language": lang, "mean": mean, "ci_low": ci_low, "ci_high": ci_high})

summary_df = pd.DataFrame(summary)

plt.figure(figsize=(6, 4))
sns.barplot(data=summary_df, x="language", y="mean")
plt.errorbar(
    x=range(len(summary_df)),
    y=summary_df["mean"],
    yerr=[summary_df["mean"] - summary_df["ci_low"],
          summary_df["ci_high"] - summary_df["mean"]],
    fmt="none",
    ecolor="black",
    capsize=4
)
plt.title("Mean Toxicity — Hinglish vs English (95% CI)")
plt.xlabel("Language")
plt.ylabel("Mean Toxicity")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 2. KDE Plot
# -----------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.kdeplot(data=df, x="toxicity", hue="language", fill=True, alpha=0.3)
plt.title("Toxicity Distribution — Hinglish vs English")
plt.xlabel("Toxicity Score")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 3. Histogram
# -----------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="toxicity", hue="language", kde=False, alpha=0.4)
plt.title("Histogram of Toxicity — Hinglish vs English")
plt.xlabel("Toxicity Score")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 4. Boxplot
# -----------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="language", y="toxicity")
plt.title("Toxicity Spread — Hinglish vs English")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 5. Heatmap of All Toxicity Dimensions
# -----------------------------------------------------------
pivot = df.groupby("language")[["toxicity","severe_toxicity","insult","profanity","identity_attack"]].mean()

plt.figure(figsize=(7, 4))
sns.heatmap(pivot, annot=True, cmap="coolwarm")
plt.title("Avg Toxicity Dimensions — Hinglish vs English")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 6. Word Clouds
# -----------------------------------------------------------
for lang in ["hinglish", "english"]:
    subset = df[df["language"] == lang]
    text = " ".join(subset["comment"].tolist())

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {lang.capitalize()} Counterspeech")
    plt.show()
