from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template='generate detail report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='generete 5 pointer summary from the following text {text}',
    input_variables=['text']
)
parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'curuption in india'})

print(result)
