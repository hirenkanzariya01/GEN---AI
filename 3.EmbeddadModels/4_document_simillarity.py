from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

os.environ["HF_HOME"] = "D:/huggingface_embeddings_cashe"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "virat kohli is an indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is an format indian caption framous for his clam demeanor and finishing skill",
    "Sachin tendukar is also known as the god of cricket holds many betting records",
    "Rohit sharma is also known for his elegent bettings and record breaking double centuries.",
    "Jasprit bumrah is an indian fast bowler known for his yorkerts.",
]

query = "what is rohit sharma "

doc_embedings = embedding.embed_documents(documents)
query_embedings = embedding.embed_query(query)

scores = cosine_similarity([query_embedings], doc_embedings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(documents[index])
print("similary score is :", score)
