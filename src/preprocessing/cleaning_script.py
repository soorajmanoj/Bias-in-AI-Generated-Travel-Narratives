import re
import json
import os
try:
    from spellchecker import SpellChecker
except ImportError:
    print(" Pyspellchecker not found. Spelling correction will be skipped.")
    print("Please run 'pip install pyspellchecker'")
    SpellChecker = None
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print(" Langdetect not found. Language detection will be skipped.")
    print("Please run 'pip install langdetect'")
    detect = None


STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",

    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])


CUSTOM_VOCABULARY = {
    'kro', 'esme', 'tumhara', 'mahine', 'ka', 'kitna', 'hai', 'bhai', 'kya',
    'nahi', 'mein', 'tera', 'mera', 'hum', 'ho', 'gaya', 'tha',
    'vlog', 'vlogger', 'bro', 'bruh', 'btw', 'cuz', 'dey', 'u', 'ur', 'ajj','ji',
    'paisa','bidesh','videsh','itna','Ye','sab','baachi','kaha','kahan','tumse',
    'bahut','bhaiya','kon','dekh','dekhke','jabardast','garibi','sala','acha',
    'bura','desh','khareed','nehi','nahi','aapka','aap','kaun','lgti','bhabi','apne','bhabi','bhabhi'
}



def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

def remove_stopwords(text):
    """Removes stop words and punctuation from a string."""
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)

def correct_spelling(text, spell_checker_instance):
    """Corrects spelling for each word in a sentence."""
    if not spell_checker_instance:
        return text
    
    words = re.findall(r'\b\w+\b', text.lower())
    corrected_words = [spell_checker_instance.correction(word) for word in words]
    final_words = [corr if corr is not None else orig for corr, orig in zip(corrected_words, words)]
    return " ".join(final_words)


input_filepath = os.path.join('..', '..', 'data', 'raw', 'youtube_data.json')
output_filepath = os.path.join('..', '..', 'data', 'clean', 'cleaned_data_noLLM_full.json')

try:
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f" Error: The file {input_filepath} was not found.")
    exit()

if SpellChecker:
    spell = SpellChecker()
    spell.word_frequency.load_words(CUSTOM_VOCABULARY)
    print(" Spelling corrector initialized with custom vocabulary.")
else:
    spell = None

all_cleaned_data = []
print("🚀 Starting data cleaning process...")

for video_object in data:
    if not isinstance(video_object, dict) or "video_id" not in video_object:
        continue

    cleaned_data = {
        "video_id": video_object["video_id"],
        "rom_hindi": [],
        "english": [],
        "other": []
    }

    for comment in video_object.get("comments", []):
        if not isinstance(comment, str) or not comment.strip():
            continue
        original_words = set(word.lower() for word in comment.split())
        if CUSTOM_VOCABULARY.intersection(original_words):
            processed_comment = comment.lower()
            cleaned_comment = remove_emojis(processed_comment)
            cleaned_comment = remove_stopwords(cleaned_comment)
            cleaned_data["rom_hindi"].append(" ".join(cleaned_comment.split()))
            continue


        lang = 'other'
        if detect:
            try:
                lang = detect(comment)
            except LangDetectException:
                lang = 'other'

        if lang == 'en':
            processed_comment = correct_spelling(comment, spell)
        else:

            processed_comment = comment.lower()
        

        cleaned_comment = remove_emojis(processed_comment)
        cleaned_comment = remove_stopwords(cleaned_comment)
        cleaned_comment = " ".join(cleaned_comment.split())
        

        if lang == 'en':
            cleaned_data["english"].append(cleaned_comment)
        else:
            cleaned_data["other"].append(cleaned_comment)

    all_cleaned_data.append(cleaned_data)


output_dir = os.path.dirname(output_filepath)
os.makedirs(output_dir, exist_ok=True)

with open(output_filepath, 'w', encoding='utf-8') as f:
    json.dump(all_cleaned_data, f, indent=4, ensure_ascii=False)

print(f" Successfully processed {len(all_cleaned_data)} video(s) and saved the result to {output_filepath}")