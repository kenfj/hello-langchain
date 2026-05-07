# Hello World LangChain/LangGraph/Pydantic AI

LLM エージェントフレームワークを比較するためのサンプルコード集です。

* LLM バックエンドに [Ollama](https://ollama.com/)（ローカル LLM）を使用。
* LangChain/LangGraph では `gemma4:latest`、Pydantic AI の tool calling では `qwen3.5:latest` を使用。

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

### Basic

| Step    | ファイル       | 内容                               | ライブラリ |
| ------- | -------------- | ---------------------------------- | ---------- |
| 1       | chat_step1.py  | LLM 直接呼び出し                   | LangChain  |
| 2       | chat_step2.py  | プロンプト + 構造化出力            | LangChain  |
| 3       | chat_step3.py  | チェーン + 履歴                    | LangChain  |
| 4       | chat_step4.py  | LangGraph 基本 + MemorySaver       | LangGraph  |
| 5       | chat_step5.py  | Tool 定義 + bind_tools             | LangChain  |
| 6       | chat_step6.py  | LangGraph Agent（Tool 実行ループ） | LangGraph  |
| API     | main.py        | LangServe で API 公開（履歴付き）  | LangServe  |

### Advanced (LangChain/PydanticAI)

| Step    | ファイル       | 内容                        | ポイント |
| ------- | -------------- | ------------------------- | ---------- |
| 7       | chat_step7.py  | 翻訳→校正（多段パイプライン） | 複数ノード、カスタム State  |
| 8       | chat_step8.py  | 分類→回答（条件分岐）        | ルーティング、分類結果に応じた処理  |
| 9       | chat_step9.py  | 検索→要約（RAG）            | 外部データソース連携  |
| 10      | chat_step10.py | エラーハンドリング           | リトライ・フォールバック  |
| 11      | chat_step11.py | Human-in-the-loop         | 中断・再開  |

各 Step には `chat_stepN_pydantic.py`（Pydantic AI 版）も用意している。

> **Note:** Advanced の各ステップはグラフ構築パターンの学習を目的としたサンプルであり、multi-agent を推奨するものではない。
>
> - 同一モデルで多段に分業させるとコンテキストが断片化し、単一プロンプトより結果が悪化する場合がある
> - 現在のモデルは system prompt でペルソナを設定しなくても十分な能力があり、分けるだけでは賢くならない
> - Multi-agent が常に優れているわけではなく、有効なのはコンテキスト長の制約やツール権限の分離など構造上分けざるを得ないケースに限られる
> - 1回のプロンプトで済むならそれが最善。グラフ化は管理の複雑さとトレードオフ

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

## LangChain/LangGraph vs Pydantic AI/pydantic-graph

### フレームワークの比較

| 観点 | LangChain | Pydantic AI |
|---|---|---|
| コードのシンプルさ | チェーン・グラフの組み立てが必要 | Agent に集約されシンプル |
| ツール実行 | 手動 (Step 5) またはグラフで自動化 (Step 6) | Agent が自動でループ |
| 履歴管理 | store + RunnableWithMessageHistory / MemorySaver | `message_history` に渡すだけ |
| API 方式 | プロバイダーごとに専用クラス (`ChatOllama` 等) | OpenAI API を事実上の標準とする設計 |

Ollama で使う場合:

- LangChain: ネイティブ API (`/api/chat`) を直接叩く専用クラスで、幅広いモデルで安定
- Pydantic AI: OpenAI 互換 API (`/v1/...`) 経由のため、互換 API の成熟度に左右される。tool calling が安定しているモデル（qwen3.5 等）を推奨。Ollama の互換 API はモデルによって `tool_choice` 無視や `content: null` 拒否などの問題がある

### 構造化出力の実現方式

LangChain は `with_structured_output()` で、内部的にはプロバイダーごとの最適な方式を使う。Pydantic AI は以下の 3 つのモードから選択できる:

| モード | 指定方法 | 仕組み |
|---|---|---|
| tool (デフォルト) | `output_type=Answer` | `tool_choice: 'required'` + ツール定義 |
| prompted | `output_type=PromptedOutput(Answer)` | system メッセージで Schema 指示 + `json_object` |
| native | `output_type=NativeOutput(Answer)` | `response_format` に `json_schema` を渡す |

Ollama で使う場合:

- LangChain (`ChatOllama`): ネイティブ API の `format` パラメータで JSON Schema を渡し、トークン生成レベルで強制。モデル非依存で安定
- Pydantic AI: デフォルトの tool モードは `tool_choice: 'required'` をモデルが尊重するかに依存。gemma4 では動作しない（qwen3.5 では動作確認済み）

### pydantic-graph

pydantic-graph は LangGraph に相当するグラフ実行エンジン。公式ドキュメントでは "Don't use a nail gun unless you need a nail gun" と述べており、単純な用途なら Agent だけで十分。Step 4 では LangGraph との比較のために敢えて使用している。

### Human-in-the-loop の違い

| 観点 | LangGraph | pydantic-graph |
|---|---|---|
| 中断/再開 | `interrupt()` + `Command(resume=...)` | フレームワークレベルの機構なし |
| 状態の永続化 | checkpointer が実行途中の状態を保持 | 自前で state を DB に保存する必要あり |
| アプリのプロセスをまたぐ再開 | 可能（同じ実行を再開） | 新しい実行として再開する形になる |

アプリのプロセスをまたぐ中断/再開（承認フロー等）は両方とも実現可能だが、LangGraph は checkpointer として組み込みで提供している。pydantic-graph では state の永続化/復元を自前で実装する必要がある。

## Quick Start

```bash
# モデルの準備
ollama pull gemma4:latest
ollama pull qwen3.5:latest
ollama pull nomic-embed-text  # エンベディングモデル (Step 9)

# Pydantic AI 用に事前に設定
export OLLAMA_BASE_URL='http://localhost:11434/v1'

# 各ステップを個別に実行
# LangChain/LangGraph
uv run chat_step1.py
# Pydantic AI
uv run chat_step1_pydantic.py

# LangServe API サーバー起動
uv run main.py
# → http://localhost:8000/chat/playground/
```
