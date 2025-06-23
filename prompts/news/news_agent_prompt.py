from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

news_agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a news task classifier agent.

Your task is to classify the user's query into one of the following tools:

1. fetch_and_summarize_news - If the user is asking for the latest news or wants a summary about a current topic.

Only respond with the correct tool name.
"""),
    MessagesPlaceholder(variable_name="messages")
])
