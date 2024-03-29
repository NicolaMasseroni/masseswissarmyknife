import streamlit as st
from hierarchical_summarizer import HierarchicalSummarizer
import json

with st.form("my-form", clear_on_submit=True):
  language = st.selectbox('Language:', ('IT', 'EN'))
  target_chars = st.number_input('Summary lenght (in chars)', value=3000, min_value=2500, max_value=8000, step=500)
  uploaded_file = st.file_uploader("Choose a file to summarize")
  submit = st.form_submit_button('Start processing')


if submit and uploaded_file is not None:
  source_txt = uploaded_file.read().decode("utf-8")
#  st.write("Summarizing...")
#  st.write("# Source text:\n{}".format(source_txt))
  summ = HierarchicalSummarizer(verbose=True, language=language.lower(), target_chars=target_chars)
  with st.spinner('Summarizing...'):
    summary_struct, token_usage = summ.summarize(source_txt)
  summary_txt = " ".join([s["text"] for s in summary_struct])
  st.write("# Summary:\n{}".format(summary_txt))
  st.write("Total Tokens usage: {}".format(token_usage))  
  st.download_button(
      label="Download summary tree (json format)",
      data=json.dumps(summary_struct),
      file_name="summary.json",
      mime='text/plain')
  st.download_button(
      label="Download summary text",
      data=summary_txt,
      file_name="summary.txt",
      mime='text/plain')
# source_filename = 'data/transcript-230630-2024.mp3.txt'

# with open(source_filename, 'r') as file:
#     source_txt = file.read()

# summ = HierarchicalSummarizer(verbose=True)

# #ultimo_punto = source_filename.rfind(".")
# #base_filename = source_filename[:ultimo_punto]
# summary_struct, token_usage = summ.summarize(source_txt)

# summary = " ".join([s["text"] for s in summary_struct])

# print("FINALE:\n{}".format(summary))
# print("Total Tokens usage: {}".format(token_usage))