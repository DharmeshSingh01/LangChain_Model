from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


class Feedback(BaseModel):
    sentiments: Literal["Positive", "Negative"] = Field(
        description="Give the sentiment of the feedback")


parser2 = PydanticOutputParser(pydantic_object=Feedback)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier = prompt1 | model | parser2
result = classifier.invoke({'feedback': 'this is very bad product'})

prompt_pos = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt_neg = PromptTemplate(
    template='Write a appropriate response to this negative deedback \n {feedback}',
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x: x.sentiments == 'Positive', prompt_pos | model | parser),
    (lambda x: x.sentiments == 'Negative', prompt_neg | model | parser),
    RunnableLambda(lambda x: 'could not find sentiments')  # type: ignore
)

chain = classifier | branch_chain  # type:ignore

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()
