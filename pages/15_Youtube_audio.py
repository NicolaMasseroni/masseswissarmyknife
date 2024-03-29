from yt_dlp import YoutubeDL
import streamlit as st
from pytube import YouTube
import re

def convert_to_valid_filename(s):
    # Rimuovi spazi e caratteri speciali tranne l'underscore
    s = re.sub(r'[^\w\s]', '', s)
    # Sostituisci gli spazi con underscores
    s = s.replace(' ', '_')
    return s.lower()


st.set_page_config(
    page_title="Youtube audio download",
    page_icon="👋",
)
st.write("# Download audio from Youtube")

with st.form("my-form", clear_on_submit=True):
    yt_url = st.text_input('Youtube video URL')
    submit = st.form_submit_button('Start processing')

if submit and yt_url is not None:

    yt = YouTube(yt_url)
    basename = convert_to_valid_filename(yt.title)

    st.write("Title: ", yt.title)
    st.write("File name: ", basename)
    URLS = [yt_url]

    params = {'format': 'bestaudio/best',
              'outtmpl': {'default': '{}.%(ext)s'.format(basename)},
              'postprocessors': [{'key': 'FFmpegExtractAudio',
                                  'nopostoverwrites': False,
                                  'preferredcodec': 'best',
                                  'preferredquality': '5'}]}
    with st.spinner('Downloading audio...'):
        with YoutubeDL(params) as ydl:
            ydl.download(URLS)
    filename = '{}.opus'.format(basename)
    with open(filename, "rb") as file:
        st.download_button(
            label="Download audio",
            data=file,
            file_name=filename,
            mime='application/octet-stream')