import streamlit as st
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import YoutubeLoader



def prompt_summary(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=["it", "en"],)

    loader.load()

    # Define prompt
    prompt_template = """This is a transcript of a Youtube video.
    Please, write a summary of the themes of th video using the same language of the transcript.

    TRANSCRIPT:
    "{text}"

    DETAILED SUMMARY:"""
    prompt_template = """This is a transcript of a Youtube video.
    Please identify the themes that may be unique to this video and write a detailed and comprehensive summary of the video.
    Use the ITALIAN language to write the summary.
    
    TRANSCRIPT:
    "{text}"

    DETAILED SUMMARY AND HIGHLIGHTS:"""


    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
#    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm = ChatAnthropic(model=MODEL, temperature=0)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    docs = loader.load()
#    print(docs)
    resp = stuff_chain.invoke(docs)
    return resp['output_text']


def standard_summary(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["it", "en"],)

    loader.load()

    #MODEL = "claude-3-opus-20240229"
    #MODEL = "claude-3-sonnet-20240229"
    MODEL = "claude-3-haiku-20240307"

    docs = loader.load()
    print(docs)

    llm = ChatAnthropic(model=MODEL, temperature=0)

    chain = load_summarize_chain(llm, chain_type="stuff")

    print("----")

    summ =  chain.invoke(docs)
    #print(summ)

    #print(summ.keys())
    print(summ["output_text"])

#URL = "https://www.youtube.com/watch?v=sjL6Gl6ZIqs"
#URL = "https://youtu.be/jvqFAi7vkBc?si=wXbtLWm33jVCuiNN"
#URL = "https://www.youtube.com/watch?v=2zu9g3kzM-Q"
#URL = "https://www.youtube.com/watch?v=BvoJQwkrLmg"
#URL = "https://www.youtube.com/watch?v=hO7Xe2Qtb1Y" # Russian mall
#URL = "https://www.youtube.com/watch?v=EIThol_A8sM" # Eremita
#URL = "https://www.youtube.com/watch?v=zlvfCm4NKkI"
#URL = "https://www.youtube.com/watch?v=FD0QwUkAHkQ"
#URL = "https://www.youtube.com/watch?v=3-nolbBZvw4"
#URL = "https://www.youtube.com/watch?v=8mtada9uc0Q"
#URL = "https://www.youtube.com/watch?v=fWfC8jJKtxc"
#URL = "https://www.youtube.com/watch?v=a836GRBS4iQ"
#URL = "https://www.youtube.com/watch?v=jxwBlp8HfKU"
#URL = "https://www.youtube.com/watch?v=Cog4km4gQ00"
#URL = "https://www.youtube.com/watch?v=43vYRZe9aus" # Adrian
#URL = "https://www.youtube.com/watch?v=XGQdIdzxBf0"
#URL = "https://www.youtube.com/watch?v=jiiQUTQTNtk"


#MODEL = "claude-3-opus-20240229"
#MODEL = "claude-3-sonnet-20240229"
MODEL = "claude-3-haiku-20240307"

#standard_summary(URL)
#prompt_summary(URL)

st.set_page_config(
    page_title="Youtube summary",
    page_icon="👋",
)
st.write("# Summarize youtube video")

with st.form("my-form", clear_on_submit=True):
    yt_url = st.text_input('Youtube video URL')
    submit = st.form_submit_button('Start processing')

if submit and yt_url is not None:
    st.write(prompt_summary(yt_url))
    #standard_summary(yt_url)