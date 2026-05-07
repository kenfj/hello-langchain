from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

llm = ChatOllama(model="gemma4:latest", temperature=0)


Category = Literal["technical", "chat", "complaint"]


class Classification(BaseModel):
    category: Category


class RouterState(TypedDict):
    user_input: str
    category: Category
    response: str


classifier = llm.with_structured_output(Classification)


def classify(state: RouterState) -> RouterState:
    messages = [
        (
            "system",
            "Classify the user message: "
            "technical=技術的な質問, chat=雑談, complaint=苦情や不満",
        ),
        ("human", state["user_input"]),
    ]
    result = classifier.invoke(messages)
    return {"category": result.category}  # pyright: ignore[reportReturnType, reportAttributeAccessIssue]


def handle_technical(state: RouterState) -> RouterState:
    messages = [
        ("system", "あなたは技術サポートです。簡潔に回答してください。"),
        ("human", state["user_input"]),
    ]
    result = llm.invoke(messages)
    return {"response": str(result.content)}  # pyright: ignore[reportReturnType]


def handle_chat(state: RouterState) -> RouterState:
    messages = [
        ("system", "あなたはフレンドリーな話し相手です。短く気軽に会話してください。"),
        ("human", state["user_input"]),
    ]
    result = llm.invoke(messages)
    return {"response": str(result.content)}  # pyright: ignore[reportReturnType]


def handle_complaint(state: RouterState) -> RouterState:
    messages = [
        (
            "system",
            "あなたはクレーム対応の担当者です。簡潔に謝罪し、解決策を提示してください。",
        ),
        ("human", state["user_input"]),
    ]
    result = llm.invoke(messages)
    return {"response": str(result.content)}  # pyright: ignore[reportReturnType]


# ノードではなく add_conditional_edges に渡すルーティング関数
def route(state: RouterState) -> str:
    return state["category"]


graph = StateGraph(RouterState)

graph.add_node("classify", classify)
graph.add_node("handle_technical", handle_technical)
graph.add_node("handle_chat", handle_chat)
graph.add_node("handle_complaint", handle_complaint)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route,
    {
        "technical": "handle_technical",
        "chat": "handle_chat",
        "complaint": "handle_complaint",
    },
)
graph.add_edge("handle_technical", END)
graph.add_edge("handle_chat", END)
graph.add_edge("handle_complaint", END)

app = graph.compile()


def run_step8_langgraph_routing() -> None:
    inputs = [
        "Pythonでリストの重複を削除する方法は？",
        "今日はいい天気ですね！",
        "注文した商品が届かないんですが、どうなっているんですか？",
    ]

    for user_input in inputs:
        result = app.invoke({"user_input": user_input})  # pyright: ignore[reportArgumentType]
        print(f"入力: {user_input}")
        print(f"分類: {result['category']}")
        print(f"回答: {result['response']}\n")


def main() -> None:
    print("Step 8: 分類→回答（条件分岐）")
    app.get_graph().print_ascii()

    run_step8_langgraph_routing()


if __name__ == "__main__":
    main()
