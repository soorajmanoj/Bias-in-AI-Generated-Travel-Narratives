import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from umap import UMAP


# ------------- CONFIG -------------
INPUT_FILE = "../../data/processed/clean/open_coding_final.csv"
TOPIC_OUT_CSV = "../../results/topic_keywords.csv"
TOPIC_MODEL = "bertopic"  # options: "bertopic" or "lda"
NUM_TOPICS = 8  # used for LDA
# ----------------------------------


# ------------- Step 1: Load data -------------
df = pd.read_csv(INPUT_FILE)

if "label_rater1" in df.columns and "label_rater2" in df.columns:
    kappa = cohen_kappa_score(df["label_rater1"], df["label_rater2"])
    print(f"\n🧪 Cohen's Kappa: {kappa:.3f}")
    if kappa < 0.70:
        print("⚠️ Inter-coder agreement < 0.70. Consider refining codebook.\n")
else:
    print("ℹ️ Rater columns not found. Skipping Kappa calculation.")

# Filter to rows that have usable comments
comments = df["comment"].dropna().astype(str).tolist()

# ------------- Step 2: Topic Modeling -------------
if TOPIC_MODEL == "bertopic":
    print("🔍 Running BERTopic...")
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(comments)

    # Save topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(TOPIC_OUT_CSV, index=False)
    print(f"✅ BERTopic output saved to: {TOPIC_OUT_CSV}")

    # Visualize
    # Lower neighbors for small dataset
    topic_model.umap_model = UMAP(n_neighbors=5, n_components=2, metric="cosine", random_state=42)
    topic_model.visualize_topics().show()

elif TOPIC_MODEL == "lda":
    print("🔍 Running LDA...")

    vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    X = vectorizer.fit_transform(comments)

    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    lda.fit(X)

    vocab = vectorizer.get_feature_names_out()

    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vocab[i] for i in topic.argsort()[:-11:-1]]
        print(f"\n🧠 Topic {topic_idx}: {' | '.join(top_words)}")
        topic_words.append({"topic": topic_idx, "keywords": ", ".join(top_words)})

    # Save to CSV
    pd.DataFrame(topic_words).to_csv(TOPIC_OUT_CSV, index=False)
    print(f"✅ LDA topic output saved to: {TOPIC_OUT_CSV}")

else:
    raise ValueError("Invalid TOPIC_MODEL. Choose 'bertopic' or 'lda'.")

# ------------- Done -------------
print("\n🚀 M2 analysis complete.")
