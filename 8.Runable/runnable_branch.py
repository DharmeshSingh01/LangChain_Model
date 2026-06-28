from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template=' generete detail report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summarize following text \n {text}',
    input_variables=['text']  # type:ignore
)


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

generete_chain = RunnableSequence(prompt1 | model | parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2 | model | parser)),
    (RunnablePassthrough())
)

final_chain = RunnableSequence(generete_chain | branch_chain)

result = final_chain.invoke({'topic': 'DNA fingar print'})

print(result)
