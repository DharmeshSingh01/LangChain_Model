from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Dhoni'

doc_embed = embeddings.embed_documents(documents)
query_embed = embeddings.embed_query(query)
scores = cosine_similarity(np.array([query_embed]), np.array(doc_embed))[0]
print(scores)
# print(sorted(list(enumerate(scores)), key=lambda x: x[1])[-1])
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: '{query}'")
print("-" * 40)
print(f"Best match: {documents[index]}")
print(f"Similarity Score: {scores[index]:.4f}")
