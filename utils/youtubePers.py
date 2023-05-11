"""Loader that loads YouTube transcript."""
from __future__ import annotations

from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class YoutubeLoading(BaseLoader):
    """Loader that loads Youtube transcripts."""

    def __init__(
        self,
        video_id: str,
        add_video_info: bool = False,
        languages: List[str] = None,
        continue_on_failure: bool = False,
    ):
        print("Using updated version of YoutubeLoader")
        """Initialize with YouTube video ID."""
        self.video_id = video_id
        self.languages = languages if languages is not None else ["en", "es","en-GB",]
        self.add_video_info = add_video_info
        self.continue_on_failure = continue_on_failure

    @classmethod
    def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoading:
        """Given youtube URL, load video."""
        video_id = youtube_url.split("youtube.com/watch?v=")[-1]
        return cls(video_id, **kwargs)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import (
                NoTranscriptFound,
                TranscriptsDisabled,
                YouTubeTranscriptApi,
            )
        except ImportError:
            raise ImportError(
                "Could not import youtube_transcript_api python package. "
                "Please install it with `pip install youtube-transcript-api`."
            )

        metadata = {"source": self.video_id}

        if self.add_video_info:
            # Get more video meta info
            # Such as title, description, thumbnail url, publish_date
            video_info = self._get_video_info()
            metadata.update(video_info)

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            ##Creamos un diccionario que contine todos los languages codes que se encuentran en transcript
            available_transcripts = {t.language_code: t for t in transcript_list}
            ##Inicializamos la variable en none
            transcript = None
            ###Iteramos sobre los languages codes que definimos en self.languages
            
            for language in self.languages:
                ##Y buscamos en el diccionario que contiene todos los languages codes la llave que se corresponde con el lenguaje que estamos iterando
                transcript = available_transcripts.get(language)
                if transcript:
                    break
                else:
                    # Check for generated transcript
                    generated_key = f"{language}_auto"
                    ###Chequeamos buscando por transcripciones autogeneradas
                    transcript = available_transcripts.get(generated_key)
                    if transcript:
                        break
            if not transcript:
                raise NoTranscriptFound(f"No transcript found for video {self.video_id} in any of the requested languages: {self.languages}")
        except TranscriptsDisabled:
            return []

        transcript_pieces = transcript.fetch()

        transcript = " ".join([t["text"].strip(" ") for t in transcript_pieces])

        return [Document(page_content=transcript, metadata=metadata)]

    def _get_video_info(self) -> dict:
        """Get important video information.
        Components are:
            - title
            - description
            - thumbnail url,
            - publish_date
            - channel_author
            - and more.
        """
        try:
            from pytube import YouTube

        except ImportError:
            raise ImportError(
                "Could not import pytube python package. "
                "Please it install it with `pip install pytube`."
            )
        yt = YouTube(f"https://www.youtube.com/watch?v={self.video_id}")
        video_info = {
            "title": yt.title,
            "description": yt.description,
            "view_count": yt.views,
            "thumbnail_url": yt.thumbnail_url,
            "publish_date": yt.publish_date,
            "length": yt.length,
            "author": yt.author,
        }
        return video_info