import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

st.write("Hello")
st.title("Youtube Generator")
prompt = st.text_input("Input")

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a new youtube video title for my new video about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a new youtube video script for my new video about {title} while leveraging this wikipedia research {wikipedia_research}'
)

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


llm = OpenAI(temperature=0.5)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

#sequential_chain = SequentialChain(chains=[title_chain, script_chain], verbose=True, input_variables=['topic'], output_variables=['title','script'])

wiki = WikipediaAPIWrapper()

if prompt:
    #resp = sequential_chain({'topic': prompt})
    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia History'):
        st.info(wiki_research)