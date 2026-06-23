from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash')
st.header('Reserch tools')
user_imput = st.text_input('Enter your propmt')
if st.button('summarize'):
    result = model.invoke(user_imput)
    st.write(result.text)
