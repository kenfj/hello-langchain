from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
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

DB_PATH = "step4_memory.db"


def run_step4_langgraph_with_sqlite() -> None:
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": "user123"}}

        result = app.invoke({"messages": [("human", "私の名前は太郎です")]}, config)  # pyright: ignore[reportArgumentType]
        print(result["messages"][-1].content)

        # SqliteSaver がファイルに状態を保存するため、thread_id で履歴を復元
        result = app.invoke({"messages": [("human", "私の名前は？")]}, config)  # pyright: ignore[reportArgumentType]
        print(result["messages"][-1].content)


def main() -> None:
    print("Step 4: LangGraph 基本 + SqliteSaver")

    run_step4_langgraph_with_sqlite()


if __name__ == "__main__":
    main()
