import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
st.set_page_config(page_title="AI Chat App", layout="centered")

st.title("🤖 AI Chat App")
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash',)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage(
        content='You are a helpful assistant.'))

# Display chat messages
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.markdown(f"**You:** {msg['content']}")
#     else:
#         st.markdown(f"**AI:** {msg['content']}")
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    elif isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**AI:** {msg.content}")

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    # st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(HumanMessage(content=user_input))

    result = model.invoke(st.session_state.messages)
    ai_response = f"{result.text}"

    # Add AI response
    # st.session_state.messages.append(
    #     {"role": "assistant", "content": ai_response})
    st.session_state.messages.append(AIMessage(content=ai_response))

    st.rerun()

    st.write(st.session_state.messages)
