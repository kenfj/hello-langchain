import asyncio

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    "ollama:gemma4:latest",
    instructions=(
        "You are a translator from English to French."
        " Translate the user sentence."
    ),
    model_settings=ModelSettings(temperature=0),
)


def run_step1_basic() -> None:
    # invoke: get result at once
    result = agent.run_sync("I love programming.")
    print(result.output)


async def run_step1_stream() -> None:
    # stream: async with manages connection lifecycle
    async with agent.run_stream("I love programming.") as response:
        # async for yields tokens incrementally
        async for text in response.stream_text(delta=True):
            print(text, end="", flush=True)


def main() -> None:
    print("Step 1 (Pydantic AI): LLM直接呼び出し")

    run_step1_basic()
    asyncio.run(run_step1_stream())


if __name__ == "__main__":
    main()
