from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']  # type:ignore
)


def word_count(text):
    return len(text.split())


joke_chain = RunnableSequence(prompt | model | parser)

prallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_counts': RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_chain, prallel_chain)

result = final_chain.invoke({'topic': 'football'})

final_result = '''{}\n word count --> {} '''.format(
    result['joke'], result['word_counts'])

print(final_result)
