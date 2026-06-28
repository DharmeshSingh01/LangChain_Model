from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

propmt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='explain the following joke \n {text}',
    input_variables=['text']

)

chain = RunnableSequence(propmt | model | parser |
                         prompt2 | model | parser)  # type:ignore

result = chain.invoke({'topic': 'C#'})

print(result)
