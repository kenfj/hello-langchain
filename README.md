# Hello World LangChain/LangGraph

## LangChain / LangGraph / LangServe の関係

```
langchain-core    ← 共通基盤（LLM, プロンプト, メッセージ, ツール）
  ↑ 使う              ↑ 使う
LangChain            LangGraph
（チェーン: | パイプ）  （グラフ: StateGraph + ノード + エッジ）
  ↓                    ↓
Runnable             CompiledStateGraph
  ↓
LangServe（add_routes で API 自動生成 ※メンテナンスモード）
```

- LangChain と LangGraph は **オーケストレーション層が異なる**だけで、LLM やツール等の基盤は共通
- LangServe は `Runnable`（LangChain）をゼロコンフィグで API 化できるが、LangGraph のグラフでは Configurable 等が一部非対応
- LangGraph Platform（有料）が LangServe の後継だが、ローカル学習用には LangServe で十分

## Steps

| Step    | ファイル       | 内容                               | ライブラリ |
| ------- | -------------- | ---------------------------------- | ---------- |
| 1       | chat_step1.py  | LLM 直接呼び出し                   | LangChain  |
| 2       | chat_step2.py  | プロンプト + チェーン              | LangChain  |
| 3       | chat_step3.py  | チェーン + 履歴                    | LangChain  |
| 4       | chat_step4.py  | LangGraph 基本 + MemorySaver       | LangGraph  |
| 5       | chat_step5.py  | Tool 定義 + bind_tools             | LangChain  |
| 6       | chat_step6.py  | LangGraph Agent（Tool 実行ループ） | LangGraph  |
| API     | main.py        | LangServe で API 公開（履歴付き）  | LangServe  |

### Step 3 vs 4: 履歴管理の対比

| 項目           | Step 3 (LangChain)                           | Step 4 (LangGraph)              |
| -------------- | --------------------------------------------- | ------------------------------- |
| 履歴管理       | `RunnableWithMessageHistory`                  | `MemorySaver` (checkpointer)    |
| セッション識別 | `session_id`                                  | `thread_id`                     |
| 履歴ストア     | `store` + `get_session_history` を自前実装    | 不要                            |
| プロンプト     | `ChatPromptTemplate` + `MessagesPlaceholder`  | `MessagesState` が履歴を管理    |
| 出力の取得     | `StrOutputParser` で文字列化                  | `state["messages"][-1].content` |

### Step 5 vs 6: Tool 実行の対比

| 項目           | Step 5 (手動)                                 | Step 6 (LangGraph)                        |
| -------------- | --------------------------------------------- | ------------------------------------------ |
| ツール実行     | `for tool_call ... validate_user.invoke()`    | `ToolNode(tools)` が自動実行               |
| 結果の処理     | 自分でハンドリング                            | `tools_condition` で分岐 → LLM に戻す     |
| ループ         | 1回きり                                       | chat→tools→chat を LLM が満足するまで自動で繰り返す |

Step 6 ではノード間で直接値を渡すのではなく、共有の **state（messages）を介して**協調する:
- `chat` ノード: state に AIMessage を書く
- `tools_condition`: state の最後の AIMessage に tool_calls があるか判定
- `ToolNode`: state から tool_calls を読んで実行し、ToolMessage を書く

## Quick Start

```bash
# 各ステップを個別に実行
uv run chat_step1.py

# LangServe API サーバー起動
uv run main.py
# → http://localhost:8000/chat/playground/
```
