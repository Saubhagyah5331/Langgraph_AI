from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    query: str
    response: str
    next: str
    chat_history: List[BaseMessage]
    feedback_received: Optional[bool]  # Track if initial feedback was received
    clarification_received: Optional[bool]  # Track if clarification was received