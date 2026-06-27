from langchain_core.runnables import (RunnableParallel, RunnableSequence)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

propmt1 = PromptTemplate(
    template='generete a tweet about {topic}',
    input_variables=['topic']

)
prompt2 = PromptTemplate(
    template='generete linkedin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(propmt1 | model | parser),
        'linkedin': RunnableSequence(prompt2 | model | parser)
    }
)

result = parallel_chain.invoke({'topic': 'AI'})

print(result)
