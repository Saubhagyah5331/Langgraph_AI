from langchain_core.prompts import PromptTemplate

recommend_prompt = PromptTemplate.from_template(
    """
You are a smart shopping assistant. Based on the user's query "{query}", review the following product list:

{products}

Summarize and recommend the top products, mentioning:
- Key features
- Price
- Which product stands out and why

Respond in clear, user-friendly language.
"""
)
