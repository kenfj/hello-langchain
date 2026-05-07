import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

MODEL = "ollama:gemma4:latest"
Category = Literal["technical", "chat", "complaint"]


class Classification(BaseModel):
    category: Category


classifier = Agent(
    MODEL,
    instructions=(
        "Classify the user message: "
        "technical=技術的な質問, chat=雑談, complaint=苦情や不満"
    ),
    output_type=NativeOutput(Classification),
    model_settings=ModelSettings(temperature=0),
)

technical_agent = Agent(
    MODEL,
    instructions="あなたは技術サポートです。簡潔に回答してください。",
    model_settings=ModelSettings(temperature=0),
)

chat_agent = Agent(
    MODEL,
    instructions="あなたはフレンドリーな話し相手です。短く気軽に会話してください。",
    model_settings=ModelSettings(temperature=0),
)

complaint_agent = Agent(
    MODEL,
    instructions="あなたはクレーム対応の担当者です。簡潔に謝罪し、解決策を提示してください。",
    model_settings=ModelSettings(temperature=0),
)


@dataclass
class RouterState:
    user_input: str = ""
    category: Category = "chat"
    response: str = ""


@dataclass
class ClassifyNode(BaseNode[RouterState, None, str]):
    async def run(
        self, ctx: GraphRunContext[RouterState]
    ) -> HandleTechnical | HandleChat | HandleComplaint:
        result = await classifier.run(ctx.state.user_input)
        ctx.state.category = result.output.category
        match result.output.category:
            case "technical":
                return HandleTechnical()
            case "chat":
                return HandleChat()
            case "complaint":
                return HandleComplaint()


@dataclass
class HandleTechnical(BaseNode[RouterState, None, str]):
    async def run(self, ctx: GraphRunContext[RouterState]) -> End[str]:
        result = await technical_agent.run(ctx.state.user_input)
        ctx.state.response = result.output
        return End(result.output)


@dataclass
class HandleChat(BaseNode[RouterState, None, str]):
    async def run(self, ctx: GraphRunContext[RouterState]) -> End[str]:
        result = await chat_agent.run(ctx.state.user_input)
        ctx.state.response = result.output
        return End(result.output)


@dataclass
class HandleComplaint(BaseNode[RouterState, None, str]):
    async def run(self, ctx: GraphRunContext[RouterState]) -> End[str]:
        result = await complaint_agent.run(ctx.state.user_input)
        ctx.state.response = result.output
        return End(result.output)


graph = Graph(nodes=(ClassifyNode, HandleTechnical, HandleChat, HandleComplaint))


async def run_step8_routing() -> None:
    inputs = [
        "Pythonでリストの重複を削除する方法は？",
        "今日はいい天気ですね！",
        "注文した商品が届かないんですが、どうなっているんですか？",
    ]

    for user_input in inputs:
        state = RouterState(user_input=user_input)
        result = await graph.run(ClassifyNode(), state=state)
        print(f"入力: {user_input}")
        print(f"分類: {state.category}")
        print(f"回答: {result.output}\n")


def main() -> None:
    print("Step 8 (Pydantic AI): 分類→回答（条件分岐）")
    print(graph.mermaid_code())

    asyncio.run(run_step8_routing())


if __name__ == "__main__":
    main()
