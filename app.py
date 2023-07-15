import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = apikey

st.write("Hello")
st.title("Youtube Generator")
prompt = st.text_input("Input")

llm = OpenAI(temperature=0.5)

if prompt:
    resp = llm(prompt)
    st.write(resp)
