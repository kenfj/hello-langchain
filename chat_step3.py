from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:latest", temperature=0)

store: dict = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# NOTE: LangChain-only approach for reference.
# For new projects, use LangGraph with MemorySaver (see Step 4).
def run_step3_chat_with_history() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは優秀なアシスタントです。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config: RunnableConfig = {"configurable": {"session_id": "user123"}}

    result = chain_with_history.invoke({"input": "私の名前は太郎です"}, config=config)
    print(result)

    result = chain_with_history.invoke({"input": "私の名前は？"}, config=config)
    print(result)


def main() -> None:
    print("Step 3: チェーン + 履歴")

    run_step3_chat_with_history()


if __name__ == "__main__":
    main()
