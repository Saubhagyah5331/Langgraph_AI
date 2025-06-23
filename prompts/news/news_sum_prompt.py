from langchain_core.prompts import PromptTemplate

news_tool_prompt = PromptTemplate.from_template(
    """
You are a professional news summarizer.

Your task is to summarize the following news article while preserving **all key facts**, **critical information**, and **context**.

Instructions:
- Do not omit any major developments or figures.
- Maintain the article's original meaning and tone.
- Make the summary concise, clear, and suitable for a general audience.

--- News Article Content ---
{content}
----------------------------

Provide the summarized article below:
"""
)
