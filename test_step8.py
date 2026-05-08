# pyright: reportArgumentType=false
# unittest.mock.patch: モジュール内オブジェクトを MagicMock で差し替える標準手段。
# 呼び出し検証が不要なら monkeypatch、共通化したければ pytest-mock (mocker) へ。
# 設定を共通化する場合は fixture 化して conftest.py に切り出す。
from unittest.mock import MagicMock, patch

from chat_step8 import app


# @patch はモジュールの実オブジェクトを MagicMock に差し替え、引数に渡す
@patch("chat_step8.classifier")
@patch("chat_step8.llm")
def test_technical_routing(mock_llm: MagicMock, mock_classifier: MagicMock) -> None:
    mock_classifier.invoke.return_value = MagicMock(category="technical")
    mock_llm.invoke.return_value = MagicMock(content="list(set(x))")

    result = app.invoke({"user_input": "重複削除の方法"})

    assert result["category"] == "technical"
    assert result["response"] == "list(set(x))"


@patch("chat_step8.classifier")
@patch("chat_step8.llm")
def test_chat_routing(mock_llm: MagicMock, mock_classifier: MagicMock) -> None:
    mock_classifier.invoke.return_value = MagicMock(category="chat")
    mock_llm.invoke.return_value = MagicMock(content="いい天気ですね！")

    result = app.invoke({"user_input": "こんにちは"})

    assert result["category"] == "chat"
    assert result["response"] == "いい天気ですね！"


@patch("chat_step8.classifier")
@patch("chat_step8.llm")
def test_complaint_routing(mock_llm: MagicMock, mock_classifier: MagicMock) -> None:
    mock_classifier.invoke.return_value = MagicMock(category="complaint")
    mock_llm.invoke.return_value = MagicMock(content="申し訳ございません。")

    result = app.invoke({"user_input": "商品が届かない"})

    assert result["category"] == "complaint"
    assert result["response"] == "申し訳ございません。"
