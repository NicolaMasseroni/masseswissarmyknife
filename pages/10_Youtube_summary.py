import streamlit as st
#from langchain.chains.combine_documents.stuff import StuffDocumentsChain
#from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
#from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

def prompt_summary(url):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language=["it", "en", "fr", "de"]
    )

#    loader.load()
    docs = loader.load()

    # Define prompt
    prompt_template = """This is a transcript of a Youtube video.
    Please, write a summary of the themes of th video using the same language of the transcript.

    TRANSCRIPT:
    "{context}"

    DETAILED SUMMARY:"""

    prompt_template = """This is a transcript of a Youtube video.
    Please identify the themes that may be unique to this video and write a detailed and comprehensive summary of the video.
    Use the ITALIAN language to write the summary.

    TRANSCRIPT:
    "{context}"

    DETAILED SUMMARY AND HIGHLIGHTS:"""

    prompt_template = """Below is a transcript of a YouTube video. Please analyze the content and identify any themes or topics 
    that may be unique to this specific video. Based on your analysis, write a detailed and comprehensive summary in Italian, 
    capturing the key points and nuances of the discussion.

    TRANSCRIPT:
    "{context}"

    DETAILED SUMMARY AND HIGHLIGHTS:"""


    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
    #llm = ChatAnthropic(model=MODEL, temperature=0)
    chain = create_stuff_documents_chain(llm, prompt)

#    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # stuff_chain = StuffDocumentsChain(
    #     llm_chain=llm_chain, document_variable_name="text"
    # )
    #    print(docs)
#    resp = stuff_chain.invoke(docs)
    resp = chain.invoke({"context": docs})
#    print(resp)
    return resp


# def standard_summary(url):
#     loader = YoutubeLoader.from_youtube_url(
#         url, add_video_info=True, language=["it", "en", "fr", "de"]
#     )

#     loader.load()

#     # MODEL = "claude-3-opus-20240229"
#     # MODEL = "claude-3-sonnet-20240229"
#     MODEL = "claude-3-haiku-20240307"

#     docs = loader.load()
#     print(docs)

#     llm = ChatAnthropic(model=MODEL, temperature=0)

#     chain = load_summarize_chain(llm, chain_type="stuff")

#     print("----")

#     summ = chain.invoke(docs)
#     # print(summ)

#     # print(summ.keys())
#     print(summ["output_text"])




# MODEL = "claude-3-opus-20240229"
# MODEL = "claude-3-sonnet-20240229"
MODEL = "claude-3-haiku-20240307"


st.set_page_config(
    page_title="Youtube summary",
    page_icon="🤖",
)
st.write("# Summarize youtube video")

with st.form("my-form", clear_on_submit=True):
    yt_url = st.text_input("Youtube video URL")
    submit = st.form_submit_button("Start processing")

if submit and yt_url is not None:
    with st.spinner("Summarizing..."):
        st.write(prompt_summary(yt_url))
