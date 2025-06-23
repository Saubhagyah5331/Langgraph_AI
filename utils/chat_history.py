from langchain_core.messages import HumanMessage, AIMessage
from utils.state_types import AgentState


class ChatHistoryManager:
    """Class-based chat history manager."""
    
    @staticmethod
    def update_history(state: AgentState, agent_response: str) -> AgentState:
        """Update chat history with user query and agent response."""
        state["chat_history"].append(HumanMessage(content=state["query"]))
        state["chat_history"].append(AIMessage(content=agent_response))
        return state