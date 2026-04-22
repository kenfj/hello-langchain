from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def validate_user(user_id: int, addresses: list[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (list[str]): Previous addresses as a list of strings.
    """
    return user_id > 0 and len(addresses) > 0


llm = ChatOllama(model="gemma4:latest", temperature=0)
llm_with_tools = llm.bind_tools([validate_user])


def run_chat_step5_with_tools() -> None:
    result = llm_with_tools.invoke(
        "Could you validate user 123? They previously lived at "
        "123 Fake St in Boston MA and 234 Pretend Boulevard in "
        "Houston TX."
    )

    # LLM returns tool_calls (name, args, id) instead of text content
    for tool_call in result.tool_calls:
        print(tool_call)
        # manually execute the tool (Step 6: LangGraph automates this)
        output = validate_user.invoke(tool_call)
        print(output)


def main() -> None:
    print("Step 5: Tool 定義 + bind_tools")

    run_chat_step5_with_tools()


if __name__ == "__main__":
    main()
