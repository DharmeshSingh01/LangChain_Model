from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


chattemplete = ChatPromptTemplate([
    ('system', 'you are a helpful {domain} expert'),
    ('user', 'Explain in simple term, what is {topic}')
])
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash',)
prompt = chattemplete.invoke({'domain': 'cricket', 'topic': 'leg spin'})

result = model.invoke(prompt)
print(result.text)
print(prompt)
