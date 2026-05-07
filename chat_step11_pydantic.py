import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

agent = Agent(
    "ollama:gemma4:latest",
    instructions="ユーザーの質問に一文で回答してください。",
    model_settings=ModelSettings(temperature=0),
)


@dataclass
class HILState:
    user_input: str = ""
    draft: str = ""
    response: str = ""


@dataclass
class GenerateNode(BaseNode[HILState, None, str]):
    async def run(self, ctx: GraphRunContext[HILState]) -> ReviewNode:
        prompt = ctx.state.user_input
        if ctx.state.draft:
            prompt += (
                f"\n\n(前回の回答「{ctx.state.draft}」は却下されました。別の表現で)"
            )
        result = await agent.run(prompt)
        ctx.state.draft = result.output
        return ReviewNode()


@dataclass
class ReviewNode(BaseNode[HILState, None, str]):
    # pydantic-graph には中断/再開の機構がないため input() で人間の入力を待つ
    # 永続化する場合は state を DB に保存して新しい実行として再開する形になる
    # LangGraph は checkpointer で実行途中の状態を保持し、同じ実行を再開できる
    async def run(self, ctx: GraphRunContext[HILState]) -> End[str] | GenerateNode:
        print(f"\n回答案: {ctx.state.draft}")
        decision = await asyncio.to_thread(input, "承認しますか？ (yes/no): ")
        if decision.lower().strip() in ("yes", "y"):
            ctx.state.response = ctx.state.draft
            return End(ctx.state.draft)
        return GenerateNode()


graph = Graph(nodes=(GenerateNode, ReviewNode))


async def run_step11_human_in_the_loop() -> None:
    state = HILState(user_input="日本の首都の魅力を一言で")
    result = await graph.run(GenerateNode(), state=state)
    print(f"\n最終回答: {result.output}")


def main() -> None:
    print("Step 11 (Pydantic AI): Human-in-the-loop（中断・再開）")
    print(graph.mermaid_code())

    asyncio.run(run_step11_human_in_the_loop())


if __name__ == "__main__":
    main()
