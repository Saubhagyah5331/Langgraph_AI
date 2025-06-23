from langchain_core.prompts import PromptTemplate

academic_summarizer_prompt = PromptTemplate.from_template(
    """
You are a knowledgeable and concise academic assistant.

Your task is to **summarize the following academic notes**. Follow these guidelines:

- Keep all **key concepts, facts, and technical terms** intact.
- Do **not** add any extra information or assumptions.
- Make the summary **clear, well-structured, and easy to understand** for students.

--- Original Notes ---
{text}
----------------------

Now provide a well-written academic summary:
"""
)
