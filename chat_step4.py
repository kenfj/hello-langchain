from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

llm = ChatOllama(model="gemma4:latest", temperature=0)

SYSTEM_MESSAGE = ("system", "あなたは優秀なアシスタントです。")


def chat(state: MessagesState) -> MessagesState:
    ai_msg = llm.invoke([SYSTEM_MESSAGE, *state["messages"]])
    # return partial state: reducer appends to existing messages
    return {"messages": [ai_msg]}


graph = StateGraph(MessagesState)

graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=MemorySaver())


def run_step4_langgraph_with_memory() -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "user123"}}

    # arg is the initial state of the graph
    result = app.invoke({"messages": [("human", "私の名前は太郎です")]}, config)  # pyright: ignore[reportArgumentType]
    print(result["messages"][-1].content)

    # MemorySaver restores prior messages by thread_id
    result = app.invoke({"messages": [("human", "私の名前は？")]}, config)  # pyright: ignore[reportArgumentType]
    print(result["messages"][-1].content)


def main() -> None:
    print("Step 4: LangGraph 基本 + MemorySaver")

    run_step4_langgraph_with_memory()


if __name__ == "__main__":
    main()
