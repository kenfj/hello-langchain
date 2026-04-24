from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def validate_user(user_id: int, addresses: list[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (list[str]): Previous addresses as a list of strings.
    """
    return user_id > 0 and len(addresses) > 0


tools = [validate_user]
llm = ChatOllama(model="gemma4:latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# gemma4 は tool 結果を受けても自発的に要約しないため、明示的に指示
SYSTEM_MESSAGE = (
    "system",
    (
        "You are a helpful assistant. "
        "After receiving tool results, summarize the outcome for the user."
    ),
)


def chat(state: MessagesState) -> MessagesState:
    # LLM is stateless: pass full history each time so it can decide next step
    ai_msg = llm_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])
    return {"messages": [ai_msg]}


graph = StateGraph(MessagesState)

graph.add_node("chat", chat)
# ToolNode executes tools from AIMessage.tool_calls, returns ToolMessage
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat")
# tools_condition checks AIMessage after chat node:
#   tool_calls exist -> "tools" node, otherwise -> END
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

app = graph.compile(checkpointer=MemorySaver())


def run_step6_langgraph_agent() -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "user123"}}

    initial_state = {
        "messages": [
            (
                "human",
                "Could you validate user 123? They previously lived at "
                "123 Fake St in Boston MA and 234 Pretend Boulevard in "
                "Houston TX.",
            )
        ]
    }

    result = app.invoke(initial_state, config)  # pyright: ignore[reportArgumentType]

    # AIMessage has either content (text) or tool_calls (tool invocation)
    # ToolMessage has content with tool execution result (e.g. "true")
    for msg in result["messages"]:
        print(f"{type(msg).__name__}: {msg.content or msg.tool_calls}")


def main() -> None:
    print("Step 6: LangGraph Agent (Tool execution loop)")
    app.get_graph().print_ascii()

    run_step6_langgraph_agent()


if __name__ == "__main__":
    main()
