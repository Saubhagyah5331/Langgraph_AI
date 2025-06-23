import requests
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Type
from utils.gemini import GeminiLLM
from prompts.news.news_sum_prompt import news_tool_prompt
from api_key_enpoints.world_newsapi_client import get_newapi_key


class NewsArticleInput(BaseModel):
    topic: str = Field(description="Topic to search news articles for")

class NewsArticleSummarizerTool(BaseTool):
    name: str = "fetch_and_summarize_news"
    description: str = "Fetches and summarizes the latest news article related to the given topic."
    args_schema: Type[BaseModel] = NewsArticleInput

    def _run(self, topic: str) -> str:
        try:
            # Setup
            news_api_key = get_newapi_key()
            gemini_llm = GeminiLLM()
            chain = news_tool_prompt | gemini_llm.get_llm_with_parser()

            # API endpoints
            base_search_url = "https://api.worldnewsapi.com/search-news"
            base_extract_url = "https://api.worldnewsapi.com/extract-news"
            headers = {'x-api-key': news_api_key}

            # Step 1: Search article
            search_url = f"{base_search_url}?text={topic}&number=1&language=en"
            response = requests.get(search_url, headers=headers)

            if response.status_code != 200:
                return f"Error searching news: {response.status_code} - {response.text}"
            
            search_data = response.json()
            if 'news' not in search_data or not search_data['news']:
                return "No news articles found for the given topic."
            
            article = search_data['news'][0]
            article_title = article.get('title', 'No Title')
            article_url = article.get('url')

            if not article_url:
                return "Error: No article URL found."

            # Step 2: Extract content
            extract_url = f"{base_extract_url}?url={article_url}"
            extract_response = requests.get(extract_url, headers=headers)

            if extract_response.status_code != 200:
                return f"Error extracting article content: {extract_response.status_code} - {extract_response.text}"

            content_data = extract_response.json()
            article_content = content_data.get('text', '')
            if not article_content:
                return "Error: No detailed content found in the article."

            # Step 3: Summarize using Gemini
            summary = chain.invoke({"content": article_content})

            return f"**Title:** {article_title}\n\n**Summary:** {summary}"
        
        except Exception as e:
            return f"Error processing news article: {str(e)}"
