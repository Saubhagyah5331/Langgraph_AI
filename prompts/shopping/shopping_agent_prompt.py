from langchain.prompts import PromptTemplate

shopping_agent_prompt = PromptTemplate.from_template("""
You are a shopping task classifier agent.

Your job is to read the user's shopping-related query and classify it into one of the following categories:

1. recommend_product - If the user is asking for product suggestions, ideas, or recommendations.
2. compare_products - If the user wants a comparison between two specific products. The format might include "vs" or comparative language.

Only respond with one of the above tool names.

Query:
{query}

Tool name:
""")
