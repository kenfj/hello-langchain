import asyncio
from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

agent = Agent(
    "ollama:gemma4:latest",
    instructions="あなたは優秀なアシスタントです。",
    model_settings=ModelSettings(temperature=0),
)


# LangGraph の MessagesState に相当 (pydantic-graph は汎用のため自前で定義)
# データは Node ではなく State に持たせる (Node は処理の単位)
@dataclass
class ChatState:
    user_input: str = ""
    message_history: list[ModelMessage] = field(default_factory=list)


# LangGraph のノードは関数、pydantic-graph はクラス (BaseNode 継承)
# BaseModel ではなく dataclass ベース -- ノードにバリデーションは不要なため
# BaseNode の型引数: 共有状態, 外部依存, 終了時の結果型
@dataclass
class ChatNode(BaseNode[ChatState, None, str]):
    # ctx.state / ctx.deps で BaseNode の第1/第2型引数にアクセス
    # 返り値がエッジを定義: End[str] → 終了, 別ノード → 次のノードへ遷移
    async def run(self, ctx: GraphRunContext[ChatState]) -> End[str]:
        result = await agent.run(
            ctx.state.user_input,
            message_history=ctx.state.message_history,
        )
        # state を直接更新 (LangGraph の partial state + reducer に相当)
        ctx.state.message_history = result.all_messages()
        return End(result.output)


# nodes にクラスを渡してグラフ構造を定義 (add_node/add_edge は不要)
graph = Graph(nodes=(ChatNode,))


async def run_step4_graph_with_memory() -> None:
    state = ChatState(user_input="私の名前は太郎です")
    result = await graph.run(ChatNode(), state=state)
    print(result.output)

    # 履歴を引き継いで新しい state を作成 (LangGraph の MemorySaver + thread_id に相当)
    state = ChatState(
        user_input="私の名前は？",
        message_history=state.message_history,
    )
    result = await graph.run(ChatNode(), state=state)
    print(result.output)


def main() -> None:
    print("Step 4 (Pydantic AI): pydantic-graph 基本")

    asyncio.run(run_step4_graph_with_memory())


if __name__ == "__main__":
    main()
