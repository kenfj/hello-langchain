from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from ollama import ResponseError

FALLBACK_MODEL = "gemma4:latest"
MAX_RETRIES = 2
FALLBACK_RESPONSE = "現在応答できません。しばらくしてから再度お試しください。"


class RetryState(TypedDict):
    user_input: str
    model: str
    response: str
    retry_count: int
    success: bool


def call_llm(state: RetryState) -> RetryState:
    try:
        llm = ChatOllama(model=state["model"], temperature=0)
        messages = [
            ("system", "ユーザーの質問に簡潔に回答してください。"),
            ("human", state["user_input"]),
        ]
        result = llm.invoke(messages)
        return {  # pyright: ignore[reportReturnType]
            "response": str(result.content),
            "success": True,
        }
    except ResponseError:
        return {  # pyright: ignore[reportReturnType]
            "response": "",
            "success": False,
        }


# リトライ時にフォールバックモデルに切り替える
def retry(state: RetryState) -> RetryState:
    return {  # pyright: ignore[reportReturnType]
        "retry_count": state["retry_count"] + 1,
        "model": FALLBACK_MODEL,
    }


def fallback(state: RetryState) -> RetryState:  # noqa: ARG001
    return {"response": FALLBACK_RESPONSE}  # pyright: ignore[reportReturnType]


# ノードではなく add_conditional_edges に渡すルーティング関数
def check_result(state: RetryState) -> str:
    if state["success"]:
        return "done"
    if state["retry_count"] >= MAX_RETRIES:
        return "fallback"
    return "retry"


graph = StateGraph(RetryState)

graph.add_node("call_llm", call_llm)
graph.add_node("retry", retry)
graph.add_node("fallback", fallback)

graph.add_edge(START, "call_llm")
graph.add_conditional_edges(
    "call_llm",
    check_result,
    {"done": END, "retry": "retry", "fallback": "fallback"},
)
graph.add_edge("retry", "call_llm")
graph.add_edge("fallback", END)

app = graph.compile()


def run_step10_with_results(
    label: str, initial_model: str, *, retry_count: int = 0
) -> None:
    input_data = {
        "user_input": "日本の首都は？",
        "model": initial_model,
        "retry_count": retry_count,
        "success": False,
    }
    result = app.invoke(input_data)  # pyright: ignore[reportArgumentType]
    print(f"[{label}]")
    print(f"  結果: {result['response']}")
    print(f"  最終モデル: {result['model']}")
    print(f"  リトライ回数: {result['retry_count']}\n")


def main() -> None:
    print("Step 10: エラーハンドリング（リトライ・フォールバック）")
    app.get_graph().print_ascii()

    run_step10_with_results("正常系", FALLBACK_MODEL)
    run_step10_with_results("モデル切替で復旧", "nonexistent-model")
    # 最初を2回目として開始してフォールバックに遷移させる
    run_step10_with_results(
        "全滅 → フォールバック応答", "nonexistent-model", retry_count=MAX_RETRIES
    )


if __name__ == "__main__":
    main()
