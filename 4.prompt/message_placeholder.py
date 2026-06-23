from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'you are a customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{query}')
])
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)
print('#'*40)

prompt = chat_template.invoke(
    {'chat_history': chat_history, 'query': 'where is my refund?'})

print(prompt)
