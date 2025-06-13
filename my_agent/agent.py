from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, tool_node
from my_agent.utils.state import AgentState
from langgraph.prebuilt import tools_condition


# Define the config
class GraphConfig(TypedDict):
#    model_name: Literal["anthropic", "openai", "openrouter"]
    model_name: Literal["openrouter"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # internal helper that checks for tool_calls
    {
        "action": "action",
        "tools": "action",  # compatibility with older return value
        "__end__": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
