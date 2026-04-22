import uvicorn
from fastapi import FastAPI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langserve import add_routes
from pydantic import BaseModel

app = FastAPI()

llm = ChatOllama(model="gemma4:latest", temperature=0)

store: dict = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


class InputChat(BaseModel):
    input: str


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

add_routes(app, chain_with_history, path="/chat", input_type=InputChat)


def main() -> None:
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)


if __name__ == "__main__":
    main()
