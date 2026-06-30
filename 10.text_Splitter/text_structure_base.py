from langchain_text_splitters import RecursiveCharacterTextSplitter


text = '''my name is dharmesh Singh
I live in Noida
'''
splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0
)

result = splitter.split_text(text)

print(result)
