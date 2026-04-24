from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    "ollama:gemma4:latest",
    instructions="あなたは優秀なアシスタントです。",
    model_settings=ModelSettings(temperature=0),
)


def run_step3_chat_with_history() -> None:
    result1 = agent.run_sync("私の名前は太郎です")
    print(result1.output)

    # all_messages() で履歴を引き継ぐ (store 等の仕組みは不要)
    result2 = agent.run_sync(
        "私の名前は？",
        message_history=result1.all_messages(),
    )
    print(result2.output)


def main() -> None:
    print("Step 3 (Pydantic AI): チャット履歴")

    run_step3_chat_with_history()


if __name__ == "__main__":
    main()
