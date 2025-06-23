from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

classification_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a helpful routing assistant. Your job is to classify user queries into one of the following agent categories:

- academic: for educational, lecture notes, or video summary related questions.
- news: for current events, news summaries, or trending topics.
- shopping: for product recommendations or comparisons.
- general: for all other general-purpose queries.

Only respond with one of: academic, news, shopping, general.
"""),
    MessagesPlaceholder(variable_name="messages")
])
