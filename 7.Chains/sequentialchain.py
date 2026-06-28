from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template='generate detail report on {topic}',
    input_variables=['topic']  # type:ignore
)
prompt2 = PromptTemplate(
    template='generete 5 pointer summary from the following text {text}',
    input_variables=['text']  # type:ignore
)
parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chain = RunnableSequence(prompt | model | parser |
                         prompt2 | model | parser)  # type:ignore

result = chain.invoke({'topic': 'curuption in india'})

print(result)
