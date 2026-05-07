from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

llm = ChatOllama(model="gemma4:latest", temperature=0)


# カスタム State: MessagesState ではなく独自フィールド (履歴なしの1回限りの処理)
class PipelineState(TypedDict):
    user_input: str
    translated_text: str
    reviewed_text: str


def translate(state: PipelineState) -> PipelineState:
    messages = [
        ("system", "Translate the text to Japanese. Return only the translation."),
        ("human", state["user_input"]),
    ]
    result = llm.invoke(messages)
    return {"translated_text": str(result.content)}  # pyright: ignore[reportReturnType]


def review(state: PipelineState) -> PipelineState:
    messages = [
        (
            "system",
            "原文と翻訳を比較し、誤訳や不自然な表現があれば修正してください。"
            "修正後の文のみ返してください。",
        ),
        ("human", f"原文: {state['user_input']}\n翻訳: {state['translated_text']}"),
    ]
    result = llm.invoke(messages)
    return {"reviewed_text": str(result.content)}  # pyright: ignore[reportReturnType]


graph = StateGraph(PipelineState)

# 条件なしで直列に繋ぐパイプライン
graph.add_node("translate", translate)
graph.add_node("review", review)
graph.add_edge(START, "translate")
graph.add_edge("translate", "review")
graph.add_edge("review", END)

app = graph.compile()


def run_step7_langgraph_multistage_pipeline() -> None:
    user_message = (
        "She has a green thumb, her garden always looks like a million bucks."
    )

    result = app.invoke({"user_input": user_message})  # pyright: ignore[reportArgumentType]

    # 各ノードで partial state を返し、結果はマージされて、全てのフィールドが返る
    print(result)


def main() -> None:
    print("Step 7: 翻訳→校正（多段パイプライン）")
    app.get_graph().print_ascii()

    run_step7_langgraph_multistage_pipeline()


if __name__ == "__main__":
    main()
