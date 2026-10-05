from langchain_huggingface import HuggingFaceEmbeddings

embeding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = [
    "Gandhinagar is the capetal of Gujarat.",
    "Delhi is the capital of india ",
    "gujatat has largest seasport"
]

vector = embeding.embed_documents(text)

print(str(vector))
