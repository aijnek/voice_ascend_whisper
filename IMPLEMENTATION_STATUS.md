# 実装状況レポート

**プロジェクト**: Voice Ascend Whisper - 日本語学習データ収集Webアプリ統合
**日付**: 2025-12-20
**ステータス**: Phase 1完了 - 最小動作版稼働中（約70%進捗）
**Gitコミット**: `5239f9e` (feat: 日本語音声データ収集用Webアプリケーションを追加)

---

## 📋 プロジェクト概要

既存のWhisper finetuningプロジェクトに、日本語音声データ収集用のWebアプリケーションを統合するモノレポ構成の実装。

### 技術スタック
- **Backend**: FastAPI (非同期REST API)
- **Frontend**: HTMX + Pico CSS (サーバーサイドレンダリング)
- **ORM**: SQLModel (Pydantic統合)
- **DB**: SQLite (ローカル開発用)
- **言語**: 日本語（ja）
- **依存管理**: uv

---

## ✅ 完了した作業（Phase 1完了）

### Step 0: MLライブラリリネーム ✓
- [x] `src/voice_ascend_whisper/` → `src/finetune_whisper/` にリネーム完了
- [x] 全スクリプトのインポート文を更新（`scripts/*.py`）
- [x] `configs/data_config.yaml` の言語設定を `hi` → `ja` に変更
- [x] `pyproject.toml` の説明を更新

**Gitコミット**: `41db00d` (refactor: MLライブラリのリネームとWebアプリ依存関係の追加)

### Step 1: プロジェクト構造準備 ✓
- [x] Webアプリディレクトリ構造作成完了
- [x] データディレクトリ作成完了
- [x] `.env.example` 作成
- [x] `.gitignore` 更新（Webアプリデータを除外）

### Step 2: 依存関係追加 ✓
- [x] `pyproject.toml` にWebアプリ依存を追加
  - FastAPI, uvicorn, SQLModel, Jinja2
  - python-multipart, loguru, pandas
  - python-dotenv, pydantic-settings
- [x] `uv sync` 実行済み

### Step 3: データベース層実装 ✓
- [x] `src/webapp/config.py` - アプリケーション設定（get_settings()関数含む）
- [x] `src/webapp/database.py` - SQLiteエンジン、セッション管理
- [x] `src/webapp/models/text.py` - Text, TextCreate, TextUpdate モデル
- [x] `src/webapp/models/recording.py` - Recording, RecordingCreate, RecordingUpdate モデル
- [x] `src/webapp/models/dataset.py` - DatasetExport, DatasetExportCreate, **DatasetExportUpdate** モデル

### Step 4: Common Voice形式ユーティリティ ✓
- [x] `src/finetune_whisper/data/formats.py` 実装
  - `create_common_voice_tsv()`: WebアプリデータをCommon Voice形式TSVに変換
  - `validate_common_voice_format()`: データセット形式検証

### Step 5: サービス層実装（完全@staticmethod化） ✓
**重要**: すべてのサービスクラスを@staticmethodに統一し、設計の一貫性を確保

- [x] `src/webapp/services/text_service.py` - **全メソッド@staticmethod**
  - Text CRUD操作（create, get, update, delete）
  - フィルタリング、統計機能

- [x] `src/webapp/services/audio_service.py` - **全メソッド@staticmethod**
  - Base64 WAV → ファイル保存
  - 自動リサンプリング（16kHz）、モノラル変換
  - duration計算、音声バリデーション
  - **settings: Settings を引数で受け取る**

- [x] `src/webapp/services/recording_service.py` - **全メソッド@staticmethod**
  - Recording CRUD操作
  - AudioServiceと連携（staticmethod呼び出し）
  - **settings: Settings を引数で受け取る**

- [x] `src/webapp/services/export_service.py` - **全メソッド@staticmethod**
  - エクスポート設定作成・管理
  - train/dev/test分割（デフォルト80/10/10）
  - Common Voice形式TSV生成
  - `latest`シンボリックリンク自動更新
  - **settings: Settings を引数で受け取る**

### Step 6: FastAPI routes実装（Phase 1最小版） ✓
- [x] `src/webapp/main.py` - FastAPIアプリケーション初期化
  - lifespan: DB自動初期化
  - 静的ファイルマウント
  - Jinja2テンプレート設定
  - TextServiceとRecordingServiceをDependsで注入

- [x] `src/webapp/routes/__init__.py` - ルーターパッケージ

- [x] `src/webapp/routes/texts.py` - テキストCRUD API
  - GET `/texts/` - テキスト一覧（HTMX対応）
  - POST `/texts/` - テキスト作成（Form() → Pydantic変換）
  - GET `/texts/{id}` - テキスト詳細
  - GET `/texts/{id}/edit` - 編集フォーム
  - PUT `/texts/{id}` - テキスト更新
  - DELETE `/texts/{id}` - テキスト削除

### Step 7: フロントエンド基本実装（Phase 1最小版） ✓
- [x] `src/webapp/templates/base.html` - Pico CSS + HTMXベーステンプレート
- [x] `src/webapp/templates/index.html` - ダッシュボード（統計表示）
- [x] `src/webapp/templates/texts/list.html` - テキスト一覧
- [x] `src/webapp/templates/texts/form.html` - テキスト入力/編集フォーム
- [x] `src/webapp/templates/texts/detail.html` - テキスト詳細

### Step 8: 静的ファイル追加 ✓
- [x] `src/webapp/static/vendor/pico.min.css` - Pico CSS v2
- [x] `src/webapp/static/vendor/htmx.min.js` - HTMX v1.9.10
- [x] `src/webapp/static/css/custom.css` - カスタムスタイル

### 動作確認 ✓
- [x] Webアプリ起動成功: `http://localhost:8000`
- [x] ヘルスチェック正常: `/health`
- [x] ダッシュボード正常表示: `/index`
- [x] テキスト管理機能動作: `/texts/`
- [x] テキスト作成成功（ID=1作成済み）
- [x] データベース正常動作（SQLite: `data/webapp/database/webapp.db`）

---

## 🚧 未完了の作業（Phase 2 & 3）

### Phase 2: 録音機能実装 ⏳
**優先度: 高**

#### 必須ファイル
1. **`src/webapp/routes/recordings.py`** - 録音管理API
   - POST `/recordings/` - 録音アップロード（Base64 WAV）
   - GET `/recordings/` - 録音一覧
   - GET `/recordings/{id}` - 録音詳細
   - DELETE `/recordings/{id}` - 録音削除
   - PUT `/recordings/{id}/validate` - 録音バリデーション

2. **`src/webapp/routes/audio.py`** - 音声ストリーミング
   - GET `/audio/{filename}` - 音声ファイル配信

3. **`src/webapp/static/js/recorder.js`** - Web Audio API録音
   - MediaRecorder使用
   - WAV形式エンコード
   - Base64変換してサーバーへPOST

4. **`src/webapp/templates/recordings/record.html`** - 録音UI
   - テキスト選択
   - 録音開始/停止ボタン
   - 波形表示（オプション）

5. **`src/webapp/templates/recordings/list.html`** - 録音一覧

### Phase 3: エクスポート機能実装 ⏳
**優先度: 中**

#### 必須ファイル
1. **`src/webapp/routes/datasets.py`** - データセットエクスポートAPI
   - POST `/datasets/export` - エクスポート実行
   - GET `/datasets/` - エクスポート履歴
   - GET `/datasets/{id}` - エクスポート詳細
   - DELETE `/datasets/{id}` - エクスポート削除

2. **`src/webapp/templates/datasets/export.html`** - エクスポート設定UI
   - 分割比率設定（train/dev/test）
   - フィルター設定（duration, validated_only）
   - エクスポート実行ボタン

3. **`src/webapp/templates/datasets/list.html`** - エクスポート履歴

### ドキュメント作成 ⏳
**優先度: 低（機能完成後）**

- [ ] `README_WEBAPP.md` - Webアプリの詳細ドキュメント
- [ ] `README.md` 更新 - プロジェクト概要にWebアプリを追加
- [ ] ML pipelineとの統合テスト手順

---

## 🔑 重要な設計ポイント

### サービス層の統一設計（@staticmethod）

**全サービスが@staticmethodで統一されました**:

| サービス | パターン | 依存関係の渡し方 |
|---------|---------|----------------|
| TextService | `@staticmethod` | `session: Session` |
| AudioService | `@staticmethod` | `settings: Settings` |
| RecordingService | `@staticmethod` | `session: Session, settings: Settings` |
| ExportService | `@staticmethod` | `session: Session, settings: Settings` |

**メリット**:
- ✅ 一貫性: すべて同じパターン
- ✅ テスタビリティ: 依存関係を引数で注入
- ✅ シンプル: インスタンス化不要
- ✅ 効率: オブジェクト生成のオーバーヘッドなし

**呼び出し例**:
```python
# routes/texts.py
texts = TextService.get_texts(session=session)

# routes/recordings.py (Phase 2で実装予定)
recording = RecordingService.create_recording(
    session=session,
    recording_data=recording_data,
    base64_audio=base64_audio,
    settings=settings,
)
```

### Pydanticモデルの一貫性

すべてのDBモデルに対してCreate/Updateスキーマを定義:

| モデル | Base | Create | Update |
|--------|------|--------|--------|
| Text | ✅ | ✅ TextCreate | ✅ TextUpdate |
| Recording | ✅ | ✅ RecordingCreate | ✅ RecordingUpdate |
| DatasetExport | ✅ | ✅ DatasetExportCreate | ✅ **DatasetExportUpdate** (追加済み) |

### Form()とPydanticの併用パターン

HTMLフォームとPydanticバリデーションを両立:

```python
@router.post("/texts/")
async def create_text(
    content: str = Form(...),
    description: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    # Form()で受け取ったデータをPydanticモデルに変換
    text_data = TextCreate(content=content, description=description)

    # サービス層でPydanticモデルを使用
    new_text = TextService.create_text(session=session, text_data=text_data)
```

### データベーススキーマ

```python
# Text (テキストエントリ)
- id, content, description, source, language, difficulty, tags
- created_at, updated_at
- relationship: recordings

# Recording (録音データ)
- id, text_id, filename, file_path, file_size, duration
- sample_rate, channels, format
- quality_score, is_validated, notes
- created_at, updated_at
- relationship: text

# DatasetExport (エクスポート履歴)
- id, name, description, export_path
- total_recordings, train_count, dev_count, test_count
- train_ratio, dev_ratio, test_ratio, split_strategy
- min_duration, max_duration, validated_only
- status, error_message
- created_at, completed_at
```

### ディレクトリ構造

```
data/webapp/
├── audio/recordings/           # ユーザー録音ファイル
├── exports/                    # 生成データセット
│   ├── export_name_20251220/
│   │   ├── clips/              # 音声ファイル
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── test.tsv
│   └── latest -> export_name_20251220/
└── database/
    └── webapp.db (32KB, テキスト1件登録済み)
```

---

## 📝 申し送り事項

### 1. 環境設定

**.envファイルの作成**（初回のみ）:
```bash
cp .env.example .env
```

**サーバー起動**:
```bash
cd /Users/aijnek/rnd/projects/voice_ascend_whisper
uv run uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
```

**アクセス**: http://localhost:8000

### 2. サービス層の呼び出し方（重要）

すべてのサービスは@staticmethodなので、**インスタンス化不要**:

```python
from webapp.services.text_service import TextService
from webapp.services.recording_service import RecordingService
from webapp.config import get_settings

# OK - クラスから直接呼び出し
texts = TextService.get_texts(session)

# OK - settingsを渡す
settings = get_settings()
recording = RecordingService.create_recording(
    session, recording_data, base64_audio, settings
)

# NG - インスタンス化は不要
service = TextService()  # これは不要
```

### 3. Pico CSSの使い方

クラス名をほとんど使わず、セマンティックHTMLで美しいUIを実現:

```html
<form hx-post="/texts/" hx-target="#text-list">
  <label>
    テキスト
    <input type="text" name="content" required>
  </label>
  <button type="submit">追加</button>
</form>
```

### 4. HTMX開発のコツ

- `hx-target`: 更新対象のDOM要素を指定
- `hx-swap`: 更新方法を指定（innerHTML, outerHTML等）
- `hx-confirm`: 削除確認ダイアログ
- ページリロードなしでUI更新が可能

### 5. Web Audio API録音（Phase 2で実装）

**重要**: ブラウザから送信する音声データは**Base64エンコードされたWAV形式のみ**:

```javascript
// recorder.js (実装予定)
class AudioRecorder {
  async start() { /* 録音開始 */ }
  async stop() { /* 録音停止、WAV Blob返却 */ }
  async uploadToServer(textId, audioBlob) {
    const base64 = await this.blobToBase64(audioBlob);
    // POST /recordings/ with base64_audio
  }
}
```

サーバー側（AudioService）で自動的に16kHz・モノラルに変換されます。

### 6. データセットエクスポート（Phase 3で実装）

エクスポート後、MLパイプラインで使用:

```yaml
# configs/data_config.yaml
dataset:
  data_dir: ./data/webapp/exports/latest
  language: ja
```

```bash
uv run python scripts/prepare_data.py
uv run python scripts/train.py
```

---

## 🚀 次セッションでの開始方法

### 環境確認
```bash
cd /Users/aijnek/rnd/projects/voice_ascend_whisper
git status  # 最新コミット: 5239f9e
uv sync     # 依存関係確認
```

### 推奨実装順序

**Phase 2: 録音機能（次のステップ）**

1. `src/webapp/routes/recordings.py` - 録音管理API
2. `src/webapp/routes/audio.py` - 音声ストリーミング
3. `src/webapp/static/js/recorder.js` - Web Audio録音
4. `src/webapp/templates/recordings/record.html` - 録音UI
5. `src/webapp/templates/recordings/list.html` - 録音一覧
6. テスト: ブラウザで録音→保存→再生

**Phase 3: エクスポート機能**

1. `src/webapp/routes/datasets.py` - エクスポートAPI
2. `src/webapp/templates/datasets/export.html` - エクスポートUI
3. `src/webapp/templates/datasets/list.html` - エクスポート履歴
4. テスト: エクスポート実行→Common Voice形式確認→MLパイプライン連携

### 参考資料

- 詳細計画: `/Users/aijnek/.claude/plans/wiggly-herding-reef.md`
- Pico CSS: https://picocss.com/
- HTMX: https://htmx.org/
- FastAPI: https://fastapi.tiangolo.com/
- SQLModel: https://sqlmodel.tiangolo.com/
- Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder

---

## 🎯 現在の状態まとめ

### ✅ 完了済み
- MLライブラリリネーム（voice_ascend_whisper → finetune_whisper）
- Webアプリ基盤実装（FastAPI + HTMX + Pico CSS）
- データベース層（SQLModel, SQLite）
- サービス層（完全@staticmethod化）
- テキスト管理機能（CRUD完備）
- 最小動作版が稼働中

### ⏳ 次のステップ
- **Phase 2**: 録音機能実装（Web Audio API + 録音管理）
- **Phase 3**: エクスポート機能実装（Common Voice形式）
- ドキュメント作成

### 📊 進捗率
- **Phase 1（最小動作版）**: 100% ✅
- **Phase 2（録音機能）**: 0%
- **Phase 3（エクスポート機能）**: 0%
- **全体**: 約70%

---

**最終更新**: 2025-12-20 21:00
**Gitコミット**: `5239f9e` (feat: 日本語音声データ収集用Webアプリケーションを追加)
**サーバー状態**: 起動中（http://localhost:8000）
**次回セッション推奨**: Phase 2（録音機能）の実装開始
