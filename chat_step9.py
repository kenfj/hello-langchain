from typing import TypedDict

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

llm = ChatOllama(model="gemma4:latest", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# コーヒーの豆知識をベクトルストアに格納して検索する
documents = [
    "エスプレッソのカフェイン量はドリップより少ない。高圧で短時間抽出するため濃厚だが量が少ないためである。",
    "コーヒー豆は実はサクランボに似た果実の種子で、コーヒーチェリーと呼ばれる。",
    "浅煎りは酸味が強くフルーティー、深煎りは苦味が強くコクがある。",
    "コーヒーの最適な抽出温度は90〜96℃で、沸騰直後のお湯は苦味が出やすい。",
    "デカフェはカフェインを97%以上除去したもので、風味はほぼ通常と変わらない。",
]

# ドキュメントをベクトルストアに格納
vector_store = InMemoryVectorStore.from_texts(documents, embeddings)


class RAGState(TypedDict):
    user_input: str
    retrieved_docs: str
    response: str


def retrieve(state: RAGState) -> RAGState:
    results = vector_store.similarity_search(state["user_input"], k=2)
    docs_text = "\n".join(doc.page_content for doc in results)
    return {"retrieved_docs": docs_text}  # pyright: ignore[reportReturnType]


def summarize(state: RAGState) -> RAGState:
    messages = [
        (
            "system",
            "以下の参考情報を基に、ユーザーの質問に日本語で簡潔に回答してください。"
            "参考情報から回答を導けない場合のみ「情報が見つかりません」と答えてください。\n\n"
            f"参考情報:\n{state['retrieved_docs']}",
        ),
        ("human", state["user_input"]),
    ]
    result = llm.invoke(messages)
    return {"response": str(result.content)}  # pyright: ignore[reportReturnType]


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("summarize", summarize)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()


def run_step9_langgraph_rag() -> None:
    queries = [
        "エスプレッソのカフェインは多いの？",
        "浅煎りと深煎りの違いは？",
        "コーヒーの保存方法は？",
    ]

    for query in queries:
        result = app.invoke({"user_input": query})  # pyright: ignore[reportArgumentType]
        print(f"質問: {query}")
        print(f"検索結果: {result['retrieved_docs']}")
        print(f"回答: {result['response']}\n")


def main() -> None:
    print("Step 9: 検索→要約（RAG）")
    app.get_graph().print_ascii()

    run_step9_langgraph_rag()


if __name__ == "__main__":
    main()
