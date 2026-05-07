import asyncio
from dataclasses import dataclass

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# コーヒーの豆知識をベクトルストアに格納して検索する
documents = [
    "エスプレッソのカフェイン量はドリップより少ない。高圧で短時間抽出するため濃厚だが量が少ないためである。",
    "コーヒー豆は実はサクランボに似た果実の種子で、コーヒーチェリーと呼ばれる。",
    "浅煎りは酸味が強くフルーティー、深煎りは苦味が強くコクがある。",
    "コーヒーの最適な抽出温度は90〜96℃で、沸騰直後のお湯は苦味が出やすい。",
    "デカフェはカフェインを97%以上除去したもので、風味はほぼ通常と変わらない。",
]

vector_store = InMemoryVectorStore.from_texts(documents, embeddings)

summarizer = Agent(
    "ollama:gemma4:latest",
    instructions=(
        "以下の参考情報を基に、ユーザーの質問に日本語で簡潔に回答してください。"
        "参考情報から回答を導けない場合のみ「情報が見つかりません」と答えてください。"
    ),
    model_settings=ModelSettings(temperature=0),
)


@dataclass
class RAGState:
    user_input: str = ""
    retrieved_docs: str = ""
    response: str = ""


@dataclass
class RetrieveNode(BaseNode[RAGState, None, str]):
    async def run(self, ctx: GraphRunContext[RAGState]) -> SummarizeNode:
        results = vector_store.similarity_search(ctx.state.user_input, k=2)
        ctx.state.retrieved_docs = "\n".join(doc.page_content for doc in results)
        return SummarizeNode()


@dataclass
class SummarizeNode(BaseNode[RAGState, None, str]):
    async def run(self, ctx: GraphRunContext[RAGState]) -> End[str]:
        prompt = (
            f"参考情報:\n{ctx.state.retrieved_docs}\n\n質問: {ctx.state.user_input}"
        )
        result = await summarizer.run(prompt)
        ctx.state.response = result.output
        return End(result.output)


graph = Graph(nodes=(RetrieveNode, SummarizeNode))


async def run_step9_rag() -> None:
    queries = [
        "エスプレッソのカフェインは多いの？",
        "浅煎りと深煎りの違いは？",
        "コーヒーの保存方法は？",
    ]

    for query in queries:
        state = RAGState(user_input=query)
        result = await graph.run(RetrieveNode(), state=state)
        print(f"質問: {query}")
        print(f"検索結果: {state.retrieved_docs}")
        print(f"回答: {result.output}\n")


def main() -> None:
    print("Step 9 (Pydantic AI): 検索→要約（RAG）")
    print(graph.mermaid_code())

    asyncio.run(run_step9_rag())


if __name__ == "__main__":
    main()
