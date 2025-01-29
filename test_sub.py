from youtube_transcript_api import YouTubeTranscriptApi
import sys
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
#from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def prompt_summary(transcript):

    subs_string = ""

    for entry in transcript:
        start_time = int(entry['start'])
        text = entry['text']
        subs_string += f'[{start_time//60}:{start_time%60:02d}] {text}\n'
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
    chain = prompt | llm

#    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # stuff_chain = StuffDocumentsChain(
    #     llm_chain=llm_chain, document_variable_name="text"
    # )
    #    print(docs)
#    resp = stuff_chain.invoke(docs)
    resp = chain.invoke({"context": subs_string})
#    print(resp)
    return resp.content


def get_subtitles(video_id, language='it', save=False):
    """
    Scarica i sottotitoli di un video YouTube
    
    Args:
        video_id (str): ID del video YouTube (la parte finale dell'URL dopo v=)
        language (str): Codice lingua dei sottotitoli (default: 'it' per italiano)
    
    Returns:
        list: Lista di dizionari contenenti i sottotitoli con timestamp
    """
    try:
        # Ottiene la trascrizione
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        
        if save:
            # Salva i sottotitoli in un file di testo
            with open(f'sottotitoli_{video_id}.txt', 'w', encoding='utf-8') as f:
                for entry in transcript:
                    start_time = int(entry['start'])
                    text = entry['text']
                    f.write(f'[{start_time//60}:{start_time%60:02d}] {text}\n')
                    
            print(f'Sottotitoli salvati in sottotitoli_{video_id}.txt')
        return transcript
        
    except Exception as e:
        print(f'Errore nel download dei sottotitoli: {str(e)}')
        return None



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Utilizzo: python script.py <video_id>')
        sys.exit(1)
        
    video_id = sys.argv[1]
#    get_subtitles(video_id, save=True)
#    subs = get_subtitles(video_id)
    subs = get_subtitles(video_id, language='it')
    print(subs)
    summary = prompt_summary(subs)
    print(summary)