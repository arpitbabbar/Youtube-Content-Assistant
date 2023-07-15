import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.write("Hello")
st.title("Youtube Generator")
prompt = st.text_input("Input")

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a new youtube video title for my new video about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title'],
    template='write me a new youtube video script for my new video about {title}'
)

llm = OpenAI(temperature=0.5)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], verbose=True, input_variables=['topic'], output_variables=['title','script'])

if prompt:
    resp = sequential_chain({'topic': prompt})
    st.write(resp['title'])
    st.write(resp['script'])
