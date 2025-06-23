from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

system_message = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are an academic assistant specialized in helping students with:
1. Summarizing lecture notes
2. Recommending YouTube videos and summarizing them

Use the tools provided to perform the actions. When a user inputs a query, decide whether it relates to note summarization or YouTube video recommendations and call the appropriate tool.
"""),
    MessagesPlaceholder(variable_name="messages")
])
