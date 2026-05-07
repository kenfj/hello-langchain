import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

FALLBACK_MODEL = "ollama:gemma4:latest"
MAX_RETRIES = 2
FALLBACK_RESPONSE = "現在応答できません。しばらくしてから再度お試しください。"


@dataclass
class RetryState:
    user_input: str = ""
    model: str = FALLBACK_MODEL
    response: str = ""
    retry_count: int = 0
    success: bool = False


@dataclass
class CallLLMNode(BaseNode[RetryState, None, str]):
    async def run(
        self, ctx: GraphRunContext[RetryState]
    ) -> End[str] | RetryNode | FallbackNode:
        try:
            agent = Agent(
                ctx.state.model,
                instructions="ユーザーの質問に簡潔に回答してください。",
                model_settings=ModelSettings(temperature=0),
            )
            result = await agent.run(ctx.state.user_input)
            ctx.state.response = result.output
            ctx.state.success = True
            return End(result.output)
        except ModelHTTPError:
            if ctx.state.retry_count >= MAX_RETRIES:
                return FallbackNode()
            return RetryNode()


@dataclass
class RetryNode(BaseNode[RetryState, None, str]):
    """リトライ時にフォールバックモデルに切り替える"""

    async def run(self, ctx: GraphRunContext[RetryState]) -> CallLLMNode:
        ctx.state.retry_count += 1
        ctx.state.model = FALLBACK_MODEL
        return CallLLMNode()


@dataclass
class FallbackNode(BaseNode[RetryState, None, str]):
    async def run(self, ctx: GraphRunContext[RetryState]) -> End[str]:
        ctx.state.response = FALLBACK_RESPONSE
        return End(FALLBACK_RESPONSE)


graph = Graph(nodes=(CallLLMNode, RetryNode, FallbackNode))


async def run_step10_with_results(
    label: str, initial_model: str, *, retry_count: int = 0
) -> None:
    state = RetryState(
        user_input="日本の首都は？",
        model=initial_model,
        retry_count=retry_count,
    )
    result = await graph.run(CallLLMNode(), state=state)
    print(f"[{label}]")
    print(f"  結果: {result.output}")
    print(f"  最終モデル: {state.model}")
    print(f"  リトライ回数: {state.retry_count}\n")


async def run_all() -> None:
    await run_step10_with_results("正常系", FALLBACK_MODEL)
    await run_step10_with_results("モデル切替で復旧", "ollama:nonexistent-model")
    # 最初を2回目として開始してフォールバックに遷移させる
    await run_step10_with_results(
        "全滅 → フォールバック応答",
        "ollama:nonexistent-model",
        retry_count=MAX_RETRIES,
    )


def main() -> None:
    print("Step 10 (Pydantic AI): エラーハンドリング（リトライ・フォールバック）")
    print(graph.mermaid_code())

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
