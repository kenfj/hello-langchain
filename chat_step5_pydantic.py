from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings


# RunContext で Agent の deps にアクセス可能 (今回は未使用)
def validate_user(_ctx: RunContext[None], user_id: int, addresses: list[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id: the user ID.
        addresses: Previous addresses as a list of strings.
    """
    return user_id > 0 and len(addresses) > 0


# gemma4 は Ollama OpenAI 互換 API の tool calling に非対応のため qwen3.5 を使用
agent = Agent(
    "ollama:qwen3.5:latest",
    tools=[validate_user],
    model_settings=ModelSettings(temperature=0),
)


def run_step5_with_tools() -> None:
    result = agent.run_sync(
        "Could you validate user 123? They previously lived at "
        "123 Fake St in Boston MA and 234 Pretend Boulevard in "
        "Houston TX."
    )
    # Pydantic AI は tool 実行まで自動で行う (LangChain の手動実行との対比)
    print(result.output)


def main() -> None:
    print("Step 5 (Pydantic AI): Tool 定義")

    run_step5_with_tools()


if __name__ == "__main__":
    main()
