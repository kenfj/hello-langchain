# pyright: reportArgumentType=false
# Pydantic AI のテストは非同期。pytest-asyncio で async テストを実行する。
# Agent.run() を @patch で差し替え、戻り値に output 属性を持つ Mock を返す。
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chat_step8_pydantic import Classification, ClassifyNode, RouterState, graph


@pytest.mark.asyncio
@patch("chat_step8_pydantic.technical_agent")
@patch("chat_step8_pydantic.classifier")
async def test_technical_routing(
    mock_classifier: MagicMock, mock_technical: MagicMock
) -> None:
    mock_classifier.run = AsyncMock(
        return_value=MagicMock(output=Classification(category="technical"))
    )
    mock_technical.run = AsyncMock(return_value=MagicMock(output="list(set(x))"))

    state = RouterState(user_input="重複削除の方法")
    await graph.run(ClassifyNode(), state=state)

    assert state.category == "technical"
    assert state.response == "list(set(x))"


@pytest.mark.asyncio
@patch("chat_step8_pydantic.chat_agent")
@patch("chat_step8_pydantic.classifier")
async def test_chat_routing(mock_classifier: MagicMock, mock_chat: MagicMock) -> None:
    mock_classifier.run = AsyncMock(
        return_value=MagicMock(output=Classification(category="chat"))
    )
    mock_chat.run = AsyncMock(return_value=MagicMock(output="いい天気ですね！"))

    state = RouterState(user_input="こんにちは")
    await graph.run(ClassifyNode(), state=state)

    assert state.category == "chat"
    assert state.response == "いい天気ですね！"


@pytest.mark.asyncio
@patch("chat_step8_pydantic.complaint_agent")
@patch("chat_step8_pydantic.classifier")
async def test_complaint_routing(
    mock_classifier: MagicMock, mock_complaint: MagicMock
) -> None:
    mock_classifier.run = AsyncMock(
        return_value=MagicMock(output=Classification(category="complaint"))
    )
    mock_complaint.run = AsyncMock(
        return_value=MagicMock(output="申し訳ございません。")
    )

    state = RouterState(user_input="商品が届かない")
    await graph.run(ClassifyNode(), state=state)

    assert state.category == "complaint"
    assert state.response == "申し訳ございません。"
