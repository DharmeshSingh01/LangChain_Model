from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Create LangChain documents for IPL players

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model='gemini-embedding-2-preview'),
    persist_directory='my_cr_db',
    collection_name='sample'
)
docs = [doc1, doc2, doc3, doc4, doc5]

existing_data = vectorstore.get()
existing_ids = existing_data['ids'] if existing_data and 'ids' in existing_data else [
]
if len(existing_ids) > 0:
    print(
        f"Database already contains {len(existing_ids)} documents. Skipping embedding model completely to save quota!")
else:
    vectorstore.add_documents(docs)

print(vectorstore.get(include=['embeddings', 'documents', 'metadatas']))

query = "Who among these are a bowler?"
basic_results = vectorstore.similarity_search(query, k=2)

print("--- Basic Search Results ---")
for res in basic_results:
    print(res.page_content)

print('#'*100)
print("--- Search Results with same score ---")
query = "Who among these are a bowler?"

score_result = vectorstore.similarity_search_with_score(query, k=2)

for document, score in score_result:
    print(f"Score: {score} | Content: {document.page_content}")
