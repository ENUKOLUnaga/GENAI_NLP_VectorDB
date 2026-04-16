import pandas as pd

#data loading
df = pd.read_csv("google_reviews_corpus.csv")
reviews = df["review"].dropna().tolist()

print(reviews[:5])

#downloading required packages
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')

#Text Preprocessing

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

processed_reviews = []

for text in reviews:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    processed_reviews.append(tokens)

print(processed_reviews[:5])

#converting word to vector
from gensim.models import Word2Vec

w2v_model = Word2Vec(
    sentences=processed_reviews,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)

#cosine similarity
print("good vs excellent:", w2v_model.wv.similarity("good", "excellent"))
print("good vs bad:", w2v_model.wv.similarity("good", "bad"))

#Named Entity Recognition
sample = reviews[0]

tokens = word_tokenize(sample)
tags = pos_tag(tokens)
ner_tree = ne_chunk(tags)
print(ner_tree)

#FastText Embeddings
from gensim.models import FastText
ft_model = FastText(
    sentences=processed_reviews,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

#comparing word2vec and FastText
print("Word2Vec:", w2v_model.wv.most_similar("good"))
print("FastText:", ft_model.wv.most_similar("good"))

#storing responses in json
import json

results = {
    "similarity_good_excellent": float(w2v_model.wv.similarity("good", "excellent")),
    "similarity_good_bad": float(w2v_model.wv.similarity("good", "bad"))
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)