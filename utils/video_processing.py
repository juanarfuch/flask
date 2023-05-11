import os
import logging
from dotenv import find_dotenv, load_dotenv
from utils.youtubePers import YoutubeLoading
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
openaiapikey = os.environ.get("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

def load_transcript(url, add_video_info=True):
    try:
        loader = YoutubeLoading.from_youtube_url(url, add_video_info=add_video_info)
        transcript = loader.load()
    except Exception as e:
        logger.error(f"An error occurred while loading the transcript: {e}")
        return None
    return transcript

def split_transcript(transcript, chunk_size=1000, chunk_overlap=0):
    if not transcript:
        logger.error("No transcript to split.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(transcript)
    return docs