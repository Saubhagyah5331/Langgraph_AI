from langchain_core.prompts import PromptTemplate

compare_prompt = PromptTemplate.from_template(
    """
You are a product comparison expert.

Compare the following two products in terms of features, pricing, and value:

Product 1:
{product1}

Product 2:
{product2}

Generate a clear, side-by-side comparison and conclude which is better and why.
"""
)
