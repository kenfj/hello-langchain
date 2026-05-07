import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

translator = Agent(
    "ollama:gemma4:latest",
    instructions="Translate the text to Japanese. Return only the translation.",
    model_settings=ModelSettings(temperature=0),
)

reviewer = Agent(
    "ollama:gemma4:latest",
    instructions="原文と翻訳を比較し、誤訳や不自然な表現があれば修正してください。修正後の文のみ返してください。",
    model_settings=ModelSettings(temperature=0),
)


@dataclass
class PipelineState:
    user_input: str = ""
    translated_text: str = ""
    reviewed_text: str = ""


@dataclass
class TranslateNode(BaseNode[PipelineState, None, str]):
    # 返り値が ReviewNode → 次のノードへ遷移
    async def run(self, ctx: GraphRunContext[PipelineState]) -> ReviewNode:
        result = await translator.run(ctx.state.user_input)
        ctx.state.translated_text = result.output
        return ReviewNode()


@dataclass
class ReviewNode(BaseNode[PipelineState, None, str]):
    # 返り値が End[str] → 終了
    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[str]:
        prompt = (
            f"原文: {ctx.state.user_input}\n"
            f"翻訳: {ctx.state.translated_text}"
        )
        result = await reviewer.run(prompt)
        # state は mutable なので明示的に記録する
        ctx.state.reviewed_text = result.output
        return End(result.output)


# TranslateNode.run が ReviewNode を返すことでルーティングしている
graph = Graph(nodes=(TranslateNode, ReviewNode))


async def run_step7_multistage_pipeline() -> None:
    initial_state = PipelineState(
        user_input=(
            "She has a green thumb, her garden always looks like a million bucks."
        ),
    )

    result = await graph.run(TranslateNode(), state=initial_state)

    print(f"翻訳: {initial_state.translated_text}")
    print(f"校正: {result.output}")


def main() -> None:
    print("Step 7 (Pydantic AI): 翻訳→校正（多段パイプライン）")
    print(graph.mermaid_code())

    asyncio.run(run_step7_multistage_pipeline())


if __name__ == "__main__":
    main()
