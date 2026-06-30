from langchain_community.retrievers import WikipediaRetriever
retriver = WikipediaRetriever(top_k_results=2, lang="en")  # type:ignore
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

result = retriver.invoke(query)


for i, doc in enumerate(result):
    print(f"\n***********Result---{i+1}********")
    print(doc)
