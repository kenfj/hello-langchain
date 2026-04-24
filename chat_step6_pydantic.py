import asyncio
from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


def validate_user(_ctx: RunContext[None], user_id: int, addresses: list[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id: the user ID.
        addresses: Previous addresses as a list of strings.
    """
    return user_id > 0 and len(addresses) > 0


# Pydantic AI の Agent は tool 実行ループを内蔵している
# LangGraph では chat→tools→chat のループをグラフで組むが、Agent が内部で自動処理
# gemma4 は Ollama OpenAI 互換 API の tool calling に非対応のため qwen3.5 を使用
agent = Agent(
    "ollama:qwen3.5:latest",
    tools=[validate_user],
    model_settings=ModelSettings(
        temperature=0,
        extra_body={"reasoning_effort": "none"},
    ),
)


@dataclass
class ChatState:
    user_input: str = ""
    message_history: list[ModelMessage] = field(default_factory=list)


@dataclass
class ChatNode(BaseNode[ChatState, None, str]):
    async def run(self, ctx: GraphRunContext[ChatState]) -> End[str]:
        # Agent.run() が内部で tool 実行ループを回す
        # (LangGraph の chat→tools_condition→tools→chat ループに相当)
        result = await agent.run(
            ctx.state.user_input,
            message_history=ctx.state.message_history,
        )
        ctx.state.message_history = result.all_messages()
        return End(result.output)


# Agent が tool ループを内蔵するため ToolNode は不要、グラフは ChatNode→END だけ
graph = Graph(nodes=(ChatNode,))


async def run_step6_agent() -> None:
    initial_state = ChatState(
        user_input=(
            "Could you validate user 123? They previously lived at "
            "123 Fake St in Boston MA and 234 Pretend Boulevard in "
            "Houston TX."
        ),
    )

    result = await graph.run(ChatNode(), state=initial_state)
    print(result.output)


def main() -> None:
    print("Step 6 (Pydantic AI): Agent (Tool execution loop)")
    print(graph.mermaid_code())

    asyncio.run(run_step6_agent())


if __name__ == "__main__":
    main()
