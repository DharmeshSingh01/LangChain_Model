from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

document_loader = TextLoader('cricket.txt', encoding='utf-8')

doc = document_loader.load()

prompt = PromptTemplate(
    template='Summarize following poem {poem}',
    input_variables=['poem']
)

parse = StrOutputParser()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
chain = prompt | model | parse

print(chain.invoke({'poem': doc[0].page_content}))
