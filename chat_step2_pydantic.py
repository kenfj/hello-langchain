from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.settings import ModelSettings


class Answer(BaseModel):
    answer: str
    reason: str


agent_str = Agent(
    "ollama:gemma4:latest",
    instructions="質問に簡潔に答えてください。",
    model_settings=ModelSettings(temperature=0),
)

# structured output の方式は 3 種類:
#   output_type=Answer              → tool モード (tool_choice:'required')
#   output_type=PromptedOutput(..)  → system メッセージで Schema 指示 + json_object
#   output_type=NativeOutput(..)    → response_format に json_schema を渡す
# Ollama + gemma4 は tool_choice:'required' を無視するため
# NativeOutput を使用
agent_structured = Agent(
    "ollama:gemma4:latest",
    instructions="質問に答え、理由も述べてください。",
    output_type=NativeOutput(Answer),
    model_settings=ModelSettings(temperature=0),
)


def run_step2_str_output() -> None:
    result = agent_str.run_sync("日本の首都は？")
    print(result.output)


def run_step2_structured_output() -> None:
    result = agent_structured.run_sync("日本の首都は？")
    print(result.output)


def main() -> None:
    print("Step 2 (Pydantic AI): プロンプト + 構造化出力")

    run_step2_str_output()
    run_step2_structured_output()


if __name__ == "__main__":
    main()
