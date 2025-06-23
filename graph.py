from langgraph.graph import StateGraph, START, END
from utils.state_types import AgentState
from langgraph.checkpoint.mongodb import MongoDBSaver
from utils.router import AgentRouter
from agents.general_agent import GeneralAgent
from agents.news_agent import NewsAgent
from agents.shopping_agent import ShoppingAgent
from agents.academic_agent import AcademicAgent
from langgraph.types import interrupt, Command
from loguru import logger


class MultiAgentGraph:
    """Class-based multi-agent graph with human-in-the-loop functionality."""

    def __init__(self, saver: MongoDBSaver):
        self.saver = saver
        self.router = AgentRouter()
        self.graph = None
        self.agents = {
            "general": GeneralAgent(),
            "news": NewsAgent(),
            "shopping": ShoppingAgent(),
            "academic": AcademicAgent()
        }

    def build_graph(self):
        """Build and compile the LangGraph."""
        logger.info("Initializing the multi-agent LangGraph with Human-in-the-Loop...")

        graph = StateGraph(AgentState)

        # Add core nodes
        graph.add_node("router", self._router_node)
        graph.add_node("human_feedback", self._human_feedback_node)

        # Add individual agent nodes
        for name, agent in self.agents.items():
            graph.add_node(f"{name}_agent", self._make_agent_runner(agent))

        self._add_edges(graph)
        logger.success("✅ LangGraph with Human-in-the-Loop built successfully.")
        self.graph = graph.compile(checkpointer=self.saver)
        return self.graph

    def _make_agent_runner(self, agent_instance):
        """Wraps agent.run into a LangGraph-compatible function."""
        def run_agent(state: AgentState) -> AgentState:
            query = state["query"]
            response = agent_instance.run(query)
            state["response"] = response
            return state
        return run_agent

    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that classifies queries to appropriate agents."""
        logger.info(f"⭐ Routing query: {state['query']}")
        next_agent = self.router.classify_agent(state["query"])
        logger.info(f"📦 Classified to agent: {next_agent}")
        state["next"] = next_agent
        return state

    def _human_feedback_node(self, state: AgentState):
        """Human-in-the-loop feedback node using interrupts."""
        logger.info("🔄 Waiting for human feedback...")

        if not state.get("feedback_received", False):
            feedback_data = interrupt({
                "type": "satisfaction_check",
                "response": state.get("response", ""),
                "query": state.get("query", ""),
                "message": "Was the response helpful? (yes/no)"
            })

            if isinstance(feedback_data, str):
                feedback_lower = feedback_data.strip().lower()
                if feedback_lower == "yes":
                    logger.info("✅ Feedback accepted by user.")
                    state["next"] = "end"
                    self._reset_feedback_flags(state)
                    return state
                elif feedback_lower == "no":
                    logger.info("❌ Feedback rejected. Asking for clarification.")
                    state["feedback_received"] = True
                else:
                    logger.info("⚠️ Unclear feedback, ending conversation.")
                    state["next"] = "end"
                    self._reset_feedback_flags(state)
                    return state

        if state.get("feedback_received", False) and not state.get("clarification_received", False):
            clarification_data = interrupt({
                "type": "clarification_request",
                "message": "Please provide what you were expecting or how I can improve the response:"
            })

            if isinstance(clarification_data, str) and clarification_data.strip():
                logger.info(f"📝 Received clarification: {clarification_data}")
                state["query"] += f". Clarification: {clarification_data.strip()}"
                state["clarification_received"] = True
                self._reset_feedback_flags(state)
                state["next"] = "router"
                logger.info(f"🔄 Updated query: {state['query']}")
            else:
                logger.info("⚠️ No clarification provided, ending conversation.")
                state["next"] = "end"
                self._reset_feedback_flags(state)

        return state

    def _reset_feedback_flags(self, state: AgentState):
        """Reset feedback flags for next iteration."""
        state["feedback_received"] = False
        state["clarification_received"] = False

    def _add_edges(self, graph):
        """Add all edges to the graph."""
        graph.add_edge(START, "router")

        graph.add_conditional_edges(
            "router",
            lambda state: f"{state['next']}_agent",
            {f"{name}_agent": f"{name}_agent" for name in self.agents}
        )

        for name in self.agents:
            graph.add_edge(f"{name}_agent", "human_feedback")

        graph.add_conditional_edges(
            "human_feedback",
            self._route_from_feedback,
            {
                "router": "router",
                "end": END
            }
        )

    def _route_from_feedback(self, state: AgentState) -> str:
        """Route from human feedback to next destination."""
        next_destination = state.get("next", "end")
        logger.info(f"🔀 Routing from human_feedback to: {next_destination}")
        return next_destination
