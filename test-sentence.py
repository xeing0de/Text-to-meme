from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
    "Я сегодня к Диману иду, я не могу так рано точно, одиннадцать написал, так как уже точно в общаге должен буду быть"
]
print(sentences[0])

embeddings = model.encode(sentences)

print(embeddings.shape)

kw_model = KeyBERT(model=model)
keywords = kw_model.extract_keywords(
    sentences[0],
    keyphrase_ngram_range=(1, 3),
    stop_words=None,
    top_n=3,
)

print(keywords)
