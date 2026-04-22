from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:latest", temperature=0)


def run_chat_step1_basic() -> None:
    messages = [
        (
            "system",
            "You are a translator from English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    # invoke: get result at once
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)

    # stream: get tokens incrementally
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)


def main() -> None:
    print("Step 1: LLM直接呼び出し")

    run_chat_step1_basic()


if __name__ == "__main__":
    main()
