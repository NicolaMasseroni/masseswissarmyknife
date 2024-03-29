import streamlit as st
from sentence_splitter import SentenceLLMTextSplitter
from langchain.vectorstores import Weaviate
import weaviate
import os
from langchain.docstore.document import Document
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

WEAVIATE_URL = "http://localhost:8080"

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
      'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]
    }
)
#client.schema.delete_all()
schema = {
    "classes": [
        {
            "class": "Paragraph",
            "description": "A written paragraph",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
}

# dbschema = client.schema.get()
classname = "Paragraph"
# #aclient.schema.delete_class(classname)
# if not any(d['class'] == classname for d in dbschema["classes"]):
# #    print("Creating class")
#     client.schema.create(schema)

st.set_page_config(page_title="Sentence splitter",page_icon=':shark:')

st.write("# Sentence splitter")

with st.form("my-form", clear_on_submit=False):
    language = st.selectbox('Language:', ('IT', 'EN'))
    punctuation = st.checkbox("Improve punctuation", value=True)

    uploaded_file = st.file_uploader("Choose a file to summarize")
    to_vector = st.checkbox("Upload to vectorstore", value=True)

    submit = st.form_submit_button('Start processing')


if submit and uploaded_file is not None:
    source_txt = uploaded_file.read().decode("utf-8")
    sentence_splitter = SentenceLLMTextSplitter(
        chunk_size=1000, 
        language=language.lower(), 
        separator="\n", 
        verbose=True, 
        improve_punctuation=punctuation)
    with st.spinner('Splitting sentences...'):
        texts = sentence_splitter.split_text(source_txt)
    chunk_tot = len(texts)
    docs = []
    for text in texts:
        doc = Document(page_content=text)
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["doc_type"] = "Transcript"
        doc.metadata["chunk"] = texts.index(text)
        doc.metadata["chunk_tot"] = chunk_tot
        docs.append(doc)
        st.write("Chunk {}/{}:\n{}".format(texts.index(text), chunk_tot, text))
    # Join texts
    splitted = "\n\n".join(texts)
#     if st.button("Upload to Weaviate"):
#         st.balloons()
    if to_vector:
        with st.spinner("Uploading to Weaviate..."):
            retriever = WeaviateHybridSearchRetriever(client=client, index_name=classname, text_key="content", attributes=[], create_schema_if_missing=False)
            retriever.add_documents(docs)
    ts_filename = "splitted-{}".format(uploaded_file.name)
    st.download_button(
      label="Download splitted file",
      data=splitted,
      file_name=ts_filename,
      mime='text/plain')

#         st.write("Upload complete!")
#         if st.button("Clear Weaviate"):
#             st.write("Clearing Weaviate...")    

    # with st.form("wv", clear_on_submit=False):
    #     submitwv = st.form_submit_button('Upload to WV')
