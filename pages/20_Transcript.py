import streamlit as st
from pydub import AudioSegment
import openai
import math

chunk_duration = 15 * 60 * 1000

def transcribe_audio(audio, file_name):
  chunk_num = math.ceil(len(audio) / chunk_duration)
  st.write("Getting transcription...")
  bar = st.progress(0)

  ts_chunks = []
  for i, chunk in enumerate(audio[::chunk_duration]):
    chunk_filename = "chunk-{}.mp3".format(i)
#    time.sleep(1)
    with open(chunk_filename, "wb") as f:
      chunk.export(f, format="mp3")
    bar.progress((i*2 + 1) / (chunk_num * 2))

#    time.sleep(2)

    audio_file = open(chunk_filename, "rb")
    ts_chunk = openai.Audio.transcribe("whisper-1", audio_file, language=language)
    ts_chunks.append(ts_chunk['text'])
    bar.progress((i + 1) / chunk_num)
  st.write("Transcription complete!")

  ts_full = " ".join(ts_chunks)
  return ts_full

st.set_page_config(page_title="Audio transcript",page_icon=':shark:')
st.write("# Transcribe an audio file")

with st.form("my-form", clear_on_submit=True):
  language = st.selectbox('Language:', ('IT', 'EN'))

  uploaded_file = st.file_uploader("Choose a file to transcribe")
  submit = st.form_submit_button('Start processing')

if submit and uploaded_file is not None:
    if uploaded_file.name.endswith('mp3'):
      st.write('Caricato file mp3')
      with st.spinner('Reading file...'):
        audio = AudioSegment.from_mp3(uploaded_file)
      file_type = 'mp3'
    elif uploaded_file.name.endswith('opus'):
      st.write('Caricato file opus')
      with st.spinner('Reading file...'):
        audio = AudioSegment.from_file(uploaded_file, "ogg")
      file_type = 'opus'
    elif uploaded_file.name.endswith('aac'):
      st.write('Caricato file AAC')
      with st.spinner('Reading file...'):
        audio = AudioSegment.from_file(uploaded_file, "aac")
      file_type = 'aac'

    durata = len(audio) // 1000
    st.write(f'Audio lenght: {durata} seconds')
    transcript = transcribe_audio(audio, uploaded_file.name)
    ts_filename = "{}-transcript.txt".format(uploaded_file.name)
    st.download_button(
      label="Download transcription file",
      data=transcript,
      file_name=ts_filename,
      mime='text/plain')
