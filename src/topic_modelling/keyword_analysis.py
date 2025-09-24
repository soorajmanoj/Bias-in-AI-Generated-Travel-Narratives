import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.lda_model
from contextlib import redirect_stdout

# Load the JSON file.
# Make sure 'cleaned_data_noLLM.json' is in the same directory as your script.
file_path = '../../data/clean/cleaned_data_noLLM.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit()

# Extract and flatten all English comments into a single list.
# We also filter out any potential empty strings.
all_english_comments = [
    comment
    for comment_list in df['english']
    if isinstance(comment_list, list)
    for comment in comment_list
    if comment and isinstance(comment, str)
]

# Verify the number of comments extracted
print(f"Successfully extracted {len(all_english_comments)} English comments.")
print("A few examples:")
print(all_english_comments[:3])


# Since the data is already cleaned, we can proceed directly to vectorization.
# max_df=0.9 means "ignore words that appear in more than 90% of the documents"
# min_df=2 means "ignore words that appear in less than 2 documents"
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

# Create the Document-Term Matrix
X = vectorizer.fit_transform(all_english_comments)

# Get the vocabulary (the words that form our features)
feature_names = vectorizer.get_feature_names_out()

print(f"\nCreated a Document-Term Matrix with shape: {X.shape}")
print(f"Vocabulary size: {len(feature_names)}")


# Set the number of topics you want to find
num_topics = 10

# Create and fit the LDA model
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

print(f"\nTraining of LDA model with {num_topics} topics is complete.")

def display_topics(model, feature_names, num_top_words, output_file):
    output_file.write(f"\n--- Top {num_top_words} words for each topic ---\n")
    print(f"\n--- Top {num_top_words} words for each topic ---")
    for topic_idx, topic in enumerate(model.components_):
        # Sort the words in the topic by their weight
        top_words_indices = topic.argsort()[:-num_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
        output_file.write(f"Topic {topic_idx + 1}: {' '.join(top_words)}\n")

output_filename = "../../results/keyword_analysis/lda.txt"

# Open the file in write mode ('w')
with open(output_filename, 'w') as f:
    # Print the first line to the file
    f.write(f"Training of LDA model with {num_topics} topics is complete.\n\n")
    # Call the function, passing the file object 'f'
    display_topics(lda, feature_names, 10, f)

print(f"✅ Successfully saved LDA topics to '{output_filename}'")


print("\nPreparing visualization...")

# Prepare the visualization data
# Note: Set mds='tsne' for a t-SNE plot, which can be better for visualization
vis_data = pyLDAvis.lda_model.prepare(lda, X, vectorizer, mds='tsne')

# Save the visualization to an HTML file
pyLDAvis.save_html(vis_data, '../../results/keyword_analysis/topic_modeling_visualization.html')

print("\nSaved interactive visualization to 'topic_modeling_visualization.html'.")
print("Open this file in your web browser to explore the topics.")



# Your function remains unchanged
def get_top_ngrams(corpus, ngram_range, n=10):
    # Use CountVectorizer to easily get n-gram counts
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Assume 'all_english_comments' is already defined
# For example:
# all_english_comments = ["this is a good sentence", "this is another sentence"]

# --- Solution ---
output_filename = "../../results/keyword_analysis/top_ngrams.txt"

# Open the target file in write mode ('w')
with open(output_filename, 'w') as f:
    # Redirect any 'print' command inside this block to the file 'f'
    with redirect_stdout(f):
        print("--- Top 10 Bigrams (2-word phrases) ---")
        top_bigrams = get_top_ngrams(all_english_comments, ngram_range=(2, 2))
        # To make the output a bit cleaner, let's format it.
        # This converts the list of tuples to a DataFrame for nice printing.
        print(pd.DataFrame(top_bigrams, columns=['Ngram', 'Frequency']))

        print("\n--- Top 10 Trigrams (3-word phrases) ---")
        top_trigrams = get_top_ngrams(all_english_comments, ngram_range=(3, 3))
        print(pd.DataFrame(top_trigrams, columns=['Ngram', 'Frequency']))

        print("\n--- Top 10 Quadgrams (4-word phrases) ---")
        top_quadgrams = get_top_ngrams(all_english_comments, ngram_range=(4, 4))
        print(pd.DataFrame(top_quadgrams, columns=['Ngram', 'Frequency']))

# This confirmation message will print to your console, not the file
print(f"✅ Successfully saved n-gram results to '{output_filename}'")