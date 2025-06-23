import json
from youtube_search import YoutubeSearch
from prompts.academic.academic_sum_tool import academic_summarizer_prompt
from prompts.academic.academic_video_sum_tool import academic_video_sum_prompt
from utils.gemini import GeminiLLM
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# --- Academic Summarizer Tool ---
class AcademicNoteInput(BaseModel):
    text: str = Field(description="Lecture or academic notes to summarize")

class AcademicNoteSummarizerTool(BaseTool):
    name: str = "summarize_notes"
    description: str = "Summarizes lecture or study notes."
    args_schema: Type[BaseModel] = AcademicNoteInput

    def _run(self, text: str) -> str:
        gemini_llm = GeminiLLM()
        chain = academic_summarizer_prompt | gemini_llm.get_llm_with_parser()
        return chain.invoke({"text": text})


# --- YouTube Video Recommender Tool with LLM Summarization ---
class YouTubeVideoInput(BaseModel):
    query: str = Field(description="Search query for YouTube videos")


class YouTubeVideoRecommenderTool(BaseTool):
    name: str = "recommend_video"
    description: str = "Fetches and summarizes top 5 YouTube videos for a query."
    args_schema: Type[BaseModel] = YouTubeVideoInput

    def _run(self, query: str) -> str:
        try:
            results = YoutubeSearch(query, max_results=5).to_dict()
            video_data = [
                {
                    "title": r.get('title', 'No Title'),
                    "url": f"https://www.youtube.com{r.get('url_suffix', '')}"
                }
                for r in results
            ]

            # Use LLM to summarize the results
            gemini_llm = GeminiLLM()
            chain = academic_video_sum_prompt | gemini_llm.get_llm_with_parser()

            llm_input = {
                "query": query,
                "videos": video_data
            }

            return chain.invoke(llm_input)

        except Exception as e:
            return f"Error fetching or summarizing YouTube videos: {str(e)}"
