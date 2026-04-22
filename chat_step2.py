from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    reason: str


llm = ChatOllama(model="gemma4:latest", temperature=0)


def run_step2_str_output() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "質問に簡潔に答えてください。"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"input": "日本の首都は？"})
    print(result)


def run_step2_structured_output() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "質問に答え、理由も述べてください。"),
        ("human", "{input}"),
    ])

    # uses tool calling internally to enforce Pydantic schema
    chain = prompt | llm.with_structured_output(Answer)

    result = chain.invoke({"input": "日本の首都は？"})
    print(result)


def main() -> None:
    print("Step 2: プロンプト + チェーン")

    run_step2_str_output()
    run_step2_structured_output()


if __name__ == "__main__":
    main()
