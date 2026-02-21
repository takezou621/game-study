# ゲーミング英会話AI教師 MVP (Fortnite)

Fortniteのプレイ映像（録画/配信/リアルタイム）からHUD中心の状態（Game State）を抽出し、AI教師が「喋る相棒」として低遅延でコール/学習介入を行うMVPです。

- 目的：**商用レベルの安定稼働**と**低遅延（E2E 1〜2秒台）**を最優先
- MVP範囲：FortniteのHUD ROI解析（HP/Shield/Storm/Knocked 等） → State(JSON) → トリガー（P0〜P3） → Realtime音声応答
- 注意：本プロジェクトは **学習支援（コール・復習・状況理解）**を目的とし、不正行為（エイム補助等の自動化）を行いません。

---

## 1. できること（MVP + Phase 2/3）

- 入力：Fortniteの映像
  - 録画ファイル（mp4等）
  - PC画面リアルタイムキャプチャ
  - 将来：配信映像、キャプチャカード

- HUD ROI解析（正規化ROI + 自動キャリブレーション）
  - HP/Shield（左下）
  - Minimap/Storm（右上）
  - Knocked/Revive通知（中央）
  - 武器/弾/資材（右下）
  - 回復/建築/移動の検出

- Game State(JSON)生成（`value/source/confidence/ts` 付き）

- YAMLトリガーエンジン
  - P0（生存）> P1（戦術）> P2（学習）> P3（雑談）
  - `movement_state=combat` では P2/P3 を抑制
  - クールダウン、割り込み抑止、最大発話長制限

- 音声機能
  - 音声出力（OpenAI TTS/Realtime API）
  - 音声入力（VAD + Whisper STT）
  - エコー/クロストーク検出

- レビュー・分析機能
  - セッション統計（語彙、応答時間、発話数）
  - カテゴリ別スコア計算（A-Fグレード）
  - 弱点分析と改善提案
  - レポート生成（テキスト/JSON）

- 診断機能
  - オーディオデバイスチェック
  - ネットワーク接続確認
  - システムパフォーマンス監視

---

## 2. アーキテクチャ（5レイヤ）

1) **Capture/Transfer**
- ローカル動画入力（OpenCV）
- PC画面キャプチャ（MSS）
- WebRTCでROI映像 or Stateのみ送信

2) **State Recognition**
- ROI切り出し
- YOLO（UIアンカー/アイコン検出） + テンプレ/軽量OCR（数値）
- State統合（source/confidence/ts）

3) **Dialogue/Voice**
- YAMLトリガー（P0〜P3）
- Realtime APIで音声会話
- 音声入力（VAD + STT）

4) **Audio Engineering**
- PCスピーカー出力
- マイク入力（VAD付）
- エコー/クロストーク検出

5) **Personalization/LMS**
- ローカルログ保存（JSONL）
- セッション統計・スコア計算
- 弱点分析・改善提案

---

## 3. リポジトリ構成

```
.
├── README.md
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── configs/
│   ├── roi_defaults.yaml
│   ├── triggers.yaml
│   └── prompts/
│       └── system.txt
├── src/
│   ├── main.py
│   ├── constants.py
│   ├── capture/
│   │   ├── video_file.py
│   │   └── screen_capture.py
│   ├── vision/
│   │   ├── roi.py
│   │   ├── anchors.py
│   │   ├── yolo_detector.py
│   │   ├── ocr.py
│   │   └── state_builder.py
│   ├── trigger/
│   │   ├── engine.py
│   │   └── rules.py
│   ├── dialogue/
│   │   ├── templates.py
│   │   ├── openai_client.py
│   │   └── realtime_client.py
│   ├── audio/
│   │   ├── capture.py       # Audio capture from microphone
│   │   ├── vad.py           # Voice Activity Detection
│   │   └── stt_client.py    # Speech-to-Text (Whisper)
│   ├── diagnostics/
│   │   ├── audio_check.py   # Echo/crosstalk detection
│   │   ├── system_check.py  # Device/system diagnostics
│   │   └── report.py        # Diagnostic report generation
│   ├── review/
│   │   ├── stats.py         # Session statistics
│   │   ├── scorer.py        # Score calculation
│   │   ├── analyzer.py      # Weakness analysis
│   │   └── report.py        # Review report generation
│   └── utils/
│       ├── logger.py
│       ├── time.py
│       ├── webrtc.py
│       ├── constants.py
│       ├── exceptions.py
│       ├── rate_limiter.py
│       └── retry.py
├── tests/
│   ├── test_capture/
│   ├── test_vision/
│   ├── test_trigger/
│   ├── test_dialogue/
│   ├── test_audio/
│   ├── test_diagnostics/
│   └── test_review/
└── logs/
    └── (generated) sessions/
```

---

## 4. 前提環境

- OS: macOS / Windows / Linux
- Python: 3.11+ 推奨
- GPU: 任意（YOLOを使う場合はNVIDIA + CUDAで高速化可）
- 映像入力: 1080p/60fps 推奨（MVPは30fpsでも可）

### 必要なライブラリ

**コアライブラリ:**
- numpy: 数値計算
- opencv-python: 画像処理
- pillow: 画像操作

**設定/環境:**
- pyyaml: 設定ファイル読み込み
- python-dotenv: 環境変数管理

**AI/ML:**
- openai: OpenAI API（GPT-4, Whisper, Realtime API）
- ultralytics: YOLO検出

**キャプチャ:**
- mss: PC画面キャプチャ

**通信:**
- aiortc: WebRTC
- aiohttp: 非同期HTTP
- websockets: WebSocket通信

**オーディオ（オプション）:**
- sounddevice または pyaudio: オーディオ入出力
- webrtcvad: Voice Activity Detection
- silero-vad: 高精度VAD（オプション）

**診断（オプション）:**
- psutil: システム情報取得
- scipy: 音響解析（FFT）

---

## 5. セットアップ

### 5.1 Python環境

```bash
python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 環境変数

`.env.example` を `.env` にコピーして設定します。

```bash
cp .env.example .env
```

最低限：
- `OPENAI_API_KEY=...`

---

## 6. 実行方法

### 6.1 基本的な使用方法

```bash
# テキストのみ（デフォルト）
python -m src.main \
  --input video \
  --video ./samples/fortnite_sample.mp4 \
  --triggers ./configs/triggers.yaml \
  --roi ./configs/roi_defaults.yaml \
  --out ./logs/sessions/session_001
```

出力：
- `logs/sessions/session_001/state.jsonl`（フレーム/イベントごとのState）
- `logs/sessions/session_001/triggers.jsonl`（発火したトリガーと理由）
- `logs/sessions/session_001/responses.jsonl`（AIレスポンスログ）

### 6.2 音声出力

```bash
# 音声出力有効（OPENAI_API_KEY必須）
export OPENAI_API_KEY=your_key_here
python -m src.main \
  --input video \
  --video ./samples/fortnite_sample.mp4 \
  --out ./logs/sessions/session_test \
  --voice
```

### 6.3 音声入力/認識

```bash
# マイクからの音声入力有効
python -m src.main \
  --input video \
  --video ./samples/fortnite_sample.mp4 \
  --out ./logs/sessions/session_test \
  --mic

# VADモデル指定
python -m src.main \
  --input video \
  --video ./samples/fortnite_sample.mp4 \
  --out ./logs/sessions/session_test \
  --mic \
  --vad-model silero \
  --vad-threshold 0.6
```

### 6.4 診断モード

```bash
# 音響診断（エコー・クロストーク検出）
python -m src.main --diagnostics --audio-check

# システム診断（デバイス・ネットワーク確認）
python -m src.main --diagnostics --system-check

# 全診断実行
python -m src.main --diagnostics --full
```

### 6.5 レビュー/分析モード

```bash
# セッションのレビューレポート生成
python -m src.main \
  --review \
  --session ./logs/sessions/session_001 \
  --output ./logs/reports/

# 統計のみ表示
python -m src.main \
  --review \
  --session ./logs/sessions/session_001 \
  --stats-only
```

---

## 7. ROI定義（MVP初期値）

正規化座標（0〜1）で管理します。

- ROI-1（HP/Shield / 左下）：`[0.03, 0.78, 0.32, 0.98]`
- ROI-2（Minimap/Storm / 右上）：`[0.70, 0.02, 0.98, 0.30]`
- ROI-3（Knocked/Revive / 中央）：`[0.35, 0.40, 0.65, 0.65]`
- ROI-4（Weapon/Ammo/Materials / 右下）：`[0.55, 0.72, 0.98, 0.98]`

> 注意：HUDスケール/セーフゾーン設定でズレます。商用運用では初回数秒でアンカー検出→ROI補正を行います（Phase 3で実装済み）。

---

## 8. CLIオプション

### 基本オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input` | 入力タイプ（videoのみ対応） | - |
| `--video` | 動画ファイルパス | - |
| `--out` | 出力ディレクトリ | - |
| `--triggers` | トリガー設定ファイル | `configs/triggers.yaml` |
| `--roi` | ROI設定ファイル | `configs/roi_defaults.yaml` |
| `--system-prompt` | システムプロンプトファイル | `configs/prompts/system.txt` |

### 音声オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--voice` | 音声出力を有効化 | False |
| `--voice-model` | 音声モデル（tts-1/tts-1-hd/realtime） | tts-1 |
| `--mic` | マイク入力を有効化 | False |
| `--vad-model` | VADモデル（webrtc/silero/energy） | webrtc |
| `--vad-threshold` | VAD閾値（0-1） | 0.5 |
| `--device` | オーディオデバイスインデックス | 自動 |

### 診断オプション

| オプション | 説明 |
|-----------|------|
| `--diagnostics` | 診断モードを有効化 |
| `--audio-check` | 音響診断（エコー・クロストーク） |
| `--system-check` | システム診断（デバイス・ネットワーク） |
| `--full` | 全診断実行 |

### レビューオプション

| オプション | 説明 |
|-----------|------|
| `--review` | レビューモードを有効化 |
| `--session` | 対象セッションディレクトリ |
| `--output` | レポート出力先 |
| `--stats-only` | 統計のみ表示 |
| `--format` | 出力形式（text/json/both） |

---

## 9. 設定ファイル

### 9.1 ROI設定（`configs/roi_defaults.yaml`）

HUDのROI（関心領域）を定義します。

```yaml
rois:
  hp_shield:
    id: roi_1
    name: "HP/Shield"
    bbox: [0.03, 0.78, 0.32, 0.98]
    fields:
      - name: "hp"
        type: "ocr"
        location: [0.05, 0.85, 0.15, 0.95]
      # ...
```

**カスタマイズ方法:**
- HUDスケールに合わせて`bbox`座標を調整
- 異なる解像度でキャリブレーションを実行

### 9.2 トリガー設定（`configs/triggers.yaml`）

AIコーチの発話トリガーを定義します。

```yaml
triggers:
  - id: p0_low_hp
    name: "Low HP Warning"
    priority: 0  # P0: 最優先
    enabled: true
    conditions:
      - field: "player.status.hp"
        operator: "lt"
        value: 30
    template:
      combat: "Low HP! Find cover immediately!"
      non_combat: "Your health is critical. Heal up."
    cooldown_ms: 15000
```

**優先度レベル:**
- P0（0）: 生存関連
- P1（1）: 戦術関連
- P2（2）: 学習関連
- P3（3）: 雑談/復習

**カスタマイズ方法:**
- `enabled: false` でトリガーを無効化
- `template` を編集して発話内容を変更
- `cooldown_ms` で発話頻度を調整

### 9.3 システムプロンプト（`configs/prompts/system.txt`）

OpenAI API用のシステムプロンプトです。

AIコーチのペルソナ、応答スタイル、制約条件を定義します。

**カスタマイズ方法:**
- 教える英語レベルを調整
- ゲーム用語の説明方針を変更
- 応答の長さやスタイルを調整

---

## 8. トリガーエンジン（P0〜P3）

`configs/triggers.yaml` にルールを定義します。

- P0：生存（Knocked、HP急落、Stormダメージ）
- P1：戦術（Storm収縮、Rotate提案、集合）
- P2：学習（武器/アイテム語彙、過去形の質問）
- P3：雑談/復習（セッション後の軽い会話）

MVPでは以下を最重視します：
- **戦闘中は短文**
- **喋りすぎない（クールダウン）**
- **確信度が低いときは質問で確認**

---

## 9. Game State（JSON）ポリシー

各フィールドは可能な限り以下を持ちます：

- `value`: 値
- `source`: `roi | gep | inferred | user_tag`
- `confidence`: 0.0〜1.0
- `ts_ms`: タイムスタンプ

MVPでは、まず以下を安定させます：
- `player.status.hp`
- `player.status.shield`
- `player.status.is_knocked`
- `world.storm.*`
- `session.phase`

---

## 10. ロードマップ（次の実装）

### Phase 1（このリポジトリMVP） ✅ 完全完了 (2026-02-19)
- [x] 録画入力でROI→State生成
- [x] YAMLトリガーの安定化
- [x] テキスト返答で「喋る内容」品質を固める

**実装されたIssue:**
- #1 プロジェクト基盤構築 ✅
- #2 設定ファイル作成（ROI/Triggers/Prompts） ✅
- #3 ユーティリティモジュール実装 ✅
- #4 動画キャプチャモジュール実装 ✅
- #5 Visionモジュール実装（ROI/State生成） ✅
- #6 トリガーエンジン実装 ✅
- #7 ダイアログモジュール実装（テキスト返答） ✅
- #8 メインエントリーポイント実装 ✅
- #9 MVP統合テスト・検証 ✅ (51 passed)

### Phase 2（低遅延化） ✅ 完了 (2026-02-19)
- [x] PC画面キャプチャ入力
- [x] WebRTC Transfer（State or ROI映像）
- [x] Realtime音声会話（割り込み制御、短文テンプレ優先）

**実装されたIssue:**
- #10 PC画面キャプチャモジュール実装 ✅
- #11 WebRTC Transfer実装 ✅
- #12 Realtime音声会話機能実装 ✅

**使用方法:**

```bash
# テキストのみ（デフォルト）
python -m src.main \
  --input video \
  --video ./samples/test_video.mp4 \
  --out ./logs/sessions/session_test

# 音声出力有効（OPENAI_API_KEY必須）
export OPENAI_API_KEY=your_key_here
python -m src.main \
  --input video \
  --video ./samples/test_video.mp4 \
  --out ./logs/sessions/session_test \
  --voice
```

### Phase 3（コンソール対応強化） ✅ 完了 (2026-02-21)
- [x] HUDキャリブレーション（アンカー検出で自動補正）
- [x] 回復/建築/移動キューの検出実装
- [x] 音声入力/認識（VAD + STT）
- [x] 復習機能（スコア化・弱点分析）
- [x] サポート用診断（音声混線/エコー検知）

**実装されたモジュール:**

1. **Audioモジュール** (`src/audio/`)
   - `capture.py`: マイク入力、ノイズゲート、音声キャプチャ
   - `vad.py`: Voice Activity Detection（WebRTC/Silero/Energy）
   - `stt_client.py`: OpenAI Whisper APIによる音声認識

2. **Diagnosticsモジュール** (`src/diagnostics/`)
   - `audio_check.py`: エコー検出、クロストーク検出、音質分析
   - `system_check.py`: マイク/スピーカー確認、ネットワーク診断、パフォーマンスチェック
   - `report.py`: 診断レポート生成、修正提案

3. **Reviewモジュール** (`src/review/`)
   - `stats.py`: セッション統計収集（語彙、応答時間、発話数）
   - `scorer.py`: カテゴリ別スコア計算（発音、語彙、応答速度、戦略思考）
   - `analyzer.py`: 弱点分析、改善提案生成
   - `report.py`: レビューレポート生成（テキスト/JSON）

**使用方法:**

```bash
# 音声入力有効（マイクからの音声認識）
python -m src.main \
  --input video \
  --video ./samples/test_video.mp4 \
  --out ./logs/sessions/session_test \
  --mic

# VAD調整
python -m src.main \
  --input video \
  --video ./samples/test_video.mp4 \
  --out ./logs/sessions/session_test \
  --mic \
  --vad-model silero \
  --vad-threshold 0.6

# 診断モード
python -m src.main --diagnostics --audio-check

# レビューモード
python -m src.main --review --session ./logs/sessions/session_001
```

---

## 11. セキュリティ / コンプライアンス

- 本プロジェクトは学習支援を目的とし、ゲーム操作の自動化や不正優位性獲得を行いません。
- 画面解析はHUD中心（必要最小限）とし、可能な限り **State(JSON)のみ**を扱います。
- 個人情報が含まれうるログは、デフォルトでローカル保存。クラウド送信時は明示的同意を前提とします。

---

## 12. 開発メモ

推奨の検証順：
1) HP/Shieldの抽出精度を固める（最重要）
2) Knocked/Reviveの検出でP0トリガーを固める
3) Storm収縮のP1トリガーを追加
4) 学習介入（P2）を“戦闘外のみ”で慎重に導入

