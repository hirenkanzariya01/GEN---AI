from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embedings.embed_query("Delhi is the capetal of india")
print(str(result))
