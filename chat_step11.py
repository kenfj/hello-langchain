from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

llm = ChatOllama(model="gemma4:latest", temperature=0)


class HILState(TypedDict):
    user_input: str
    draft: str
    approved: bool
    response: str


def generate(state: HILState) -> HILState:
    human_msg = state["user_input"]
    if state.get("draft"):
        human_msg += f"\n\n(前回の回答「{state['draft']}」は却下されました。別の表現で)"
    messages = [
        ("system", "ユーザーの質問に一文で回答してください。"),
        ("human", human_msg),
    ]
    result = llm.invoke(messages)
    return {"draft": str(result.content)}  # pyright: ignore[reportReturnType]


def review(state: HILState) -> HILState:
    # interrupt() でグラフの実行を一時停止し、人間の入力を待つ
    # 再開時に Command(resume=値) で渡された値が interrupt() の戻り値になる
    decision = interrupt(
        f"以下の回答を承認しますか？\n\n{state['draft']}\n\n(yes/no): "
    )
    approved = decision.lower().strip() in ("yes", "y")
    return {"approved": approved}  # pyright: ignore[reportReturnType]


def approve(state: HILState) -> HILState:
    return {"response": state["draft"]}  # pyright: ignore[reportReturnType]


# ノードではなく add_conditional_edges に渡すルーティング関数
def after_review(state: HILState) -> str:
    if state["approved"]:
        return "approve"
    return "generate"


graph = StateGraph(HILState)

graph.add_node("generate", generate)
graph.add_node("review", review)
graph.add_node("approve", approve)

graph.add_edge(START, "generate")
graph.add_edge("generate", "review")
graph.add_conditional_edges(
    "review",
    after_review,
    {"approve": "approve", "generate": "generate"},
)
graph.add_edge("approve", END)

DB_PATH = "step11_memory.db"


# Human-in-the-loop には checkpointer が必須 (中断状態を保持するため)
def run_step11_human_in_the_loop() -> None:
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        app.get_graph().print_ascii()

        config: RunnableConfig = {"configurable": {"thread_id": "demo"}}

        print("--- 1回目: 生成 → 人間に確認を求めて中断 ---")
        for event in app.stream({"user_input": "日本の首都の魅力を一言で"}, config):  # pyright: ignore[reportArgumentType]
            print(event)

        print("\n--- 人間が却下 → 再生成 → 再度中断 ---")
        for event in app.stream(Command(resume="no"), config):
            print(event)

        print("\n--- 人間が承認 → 完了 ---")
        for event in app.stream(Command(resume="yes"), config):
            print(event)

        final_state = app.get_state(config)
        print(f"\n最終回答: {final_state.values['response']}")


def main() -> None:
    print("Step 11: Human-in-the-loop（中断・再開）")

    run_step11_human_in_the_loop()


if __name__ == "__main__":
    main()
