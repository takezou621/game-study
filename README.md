# ゲーミング英会話AI教師 MVP (Fortnite)

Fortniteのプレイ映像（録画/配信/リアルタイム）からHUD中心の状態（Game State）を抽出し、AI教師が「喋る相棒」として低遅延でコール/学習介入を行うMVPです。

- 目的：**商用レベルの安定稼働**と**低遅延（E2E 1〜2秒台）**を最優先
- MVP範囲：FortniteのHUD ROI解析（HP/Shield/Storm/Knocked 等） → State(JSON) → トリガー（P0〜P3） → Realtime音声応答
- 注意：本プロジェクトは **学習支援（コール・復習・状況理解）**を目的とし、不正行為（エイム補助等の自動化）を行いません。

---

## 1. できること（MVP）

- 入力：Fortniteの映像
  - 録画ファイル（mp4等）
  - 配信映像（将来）
  - リアルタイム画面（将来：PCキャプチャ/キャプチャカード）
- HUD ROI解析（正規化ROI + 初回キャリブレーション予定）
  - HP/Shield（左下）
  - Minimap/Storm（右上）
  - Knocked/Revive通知（中央）
  - 武器/弾/資材（右下）
- Game State(JSON)生成（`value/source/confidence/ts` 付き）
- YAMLトリガーエンジン
  - P0（生存）> P1（戦術）> P2（学習）> P3（雑談）
  - `movement_state=combat` では P2/P3 を抑制
  - クールダウン、割り込み抑止、最大発話長制限
- OpenAI Realtime APIによる音声応答（最初は「テキスト返答」でも動作可）

---

## 2. アーキテクチャ（5レイヤ）

1) **Capture/Transfer**
- MVPはローカル動画入力（OpenCV）で開始
- 将来：WebRTCでROI映像 or Stateのみ送信

2) **State Recognition**
- ROI切り出し
- YOLO（UIアンカー/アイコン検出） + テンプレ/軽量OCR（数値）
- State統合（source/confidence/ts）

3) **Dialogue/Voice**
- YAMLトリガー（P0〜P3）
- Realtime APIで音声会話（MVPはテキストでもOK）

4) **Audio Engineering**
- MVPはPCスピーカー出力（後で仮想ミキサーやダッキングへ拡張）

5) **Personalization/LMS**
- MVPはローカルログ保存（JSONL）
- 将来：Qdrant/Pinecone + 学習エージェント

---

## 3. リポジトリ構成

```
.
├── README.md
├── .env.example
├── requirements.txt
├── configs/
│   ├── roi_defaults.yaml
│   ├── triggers.yaml
│   └── prompts/
│       └── system.txt
├── src/
│   ├── main.py
│   ├── capture/
│   │   ├── video_file.py
│   │   └── (future) screen_capture.py
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
│   │   └── (future) realtime_client.py
│   └── utils/
│       ├── logger.py
│       └── time.py
└── logs/
    └── (generated) sessions/
```

---

## 4. 前提環境

- OS: macOS / Windows / Linux
- Python: 3.11+ 推奨
- GPU: 任意（YOLOを使う場合はNVIDIA + CUDAで高速化可）
- 映像入力: 1080p/60fps 推奨（MVPは30fpsでも可）

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

## 6. 実行方法（MVP）

### 6.1 録画ファイルからState抽出 + トリガー（音声なし）

```bash
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

### 6.2 Realtime音声応答（オプション）

将来対応：`src/dialogue/realtime_client.py` を実装後に利用します。  
MVP段階では、まず **テキスト返答**でトリガー品質を固めることを推奨します。

---

## 7. ROI定義（MVP初期値）

正規化座標（0〜1）で管理します。

- ROI-1（HP/Shield / 左下）：`[0.03, 0.78, 0.32, 0.98]`
- ROI-2（Minimap/Storm / 右上）：`[0.70, 0.02, 0.98, 0.30]`
- ROI-3（Knocked/Revive / 中央）：`[0.35, 0.40, 0.65, 0.65]`
- ROI-4（Weapon/Ammo/Materials / 右下）：`[0.55, 0.72, 0.98, 0.98]`

> 注意：HUDスケール/セーフゾーン設定でズレます。商用運用では初回数秒でアンカー検出→ROI補正を行います（MVPの次のイテレーションで実装）。

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

### Phase 2（低遅延化）
- [x] PC画面キャプチャ入力 ✅ (2026-02-19)
- [x] WebRTC Transfer（State or ROI映像） ✅ (2026-02-19)
- [x] Realtime音声会話（割り込み制御、短文テンプレ優先） ✅ (2026-02-19)

**実装されたIssue:**
- #10 PC画面キャプチャモジュール実装 ✅
- #11 WebRTC Transfer実装 ✅
- #12 Realtime音声会話実装 ✅

**Phase 2 完了！** 🎉

### Phase 3（コンソール対応強化）
- [x] キャプチャカード入力最適化 ✅ (2026-02-19)
- [x] HUDキャリブレーション（アンカー検出で自動補正） ✅ (2026-02-19)
- [x] サポート用診断（音声混線/エコー検知） ✅ (2026-02-19)

**実装されたIssue:**
- #10 PC画面キャプチャモジュール実装 ✅
- #11 Realtime音声会話機能実装 ✅
- #12 Phase 3: HUDキャリブレーションと音声診断機能実装 ✅
- #13 キャプチャカード入力最適化実装 ✅

**Phase 3 完了！** 🎉

**Phase 3 実装内容:**

1. **HUDCalibrator** (`src/vision/calibration.py`)
   - アンカー検出による自動ROIキャリブレーション
   - キャリブレーション品質の統計管理
   - キャリブレーション結果の可視化
   - キャリブレーション設定の保存・読み込み

2. **AudioDiagnostics** (`src/diagnostics/audio_diagnostics.py`)
   - 音声デバイス検出（Linux/macOS/Windows対応）
   - OpenAI APIキー設定チェック
   - 推定遅延計算
   - 一般的な音声問題の診断
   - 診断レポートの出力・保存

3. **CaptureCardCapture** (`src/capture/capture_card.py`)
   - キャプチャカードデバイスの自動検出
   - 低遅延キャプチャ設定（バッファサイズ1、自動露出無効化）
   - デバイス列挙機能
   - 解像度・フレームレートの最適化
   - コンテキストマネージャー対応

**使用方法:**

```bash
# HUDキャリブレーション
python -c "
from vision.calibration import create_hud_calibrator
import cv2

calibrator = create_hud_calibrator('./configs/roi_defaults.yaml')
frame = cv2.imread('fortnite_screenshot.png')
result = calibrator.calibrate_from_frame(frame)
if result.success:
    print('Calibration successful!')
    calibrator.save_calibration('./configs/calibrated_roi.yaml')
else:
    print(f'Calibration failed: {result.error_message}')
"

# 音声診断
python -c "
from diagnostics.audio_diagnostics import create_audio_diagnostics

diag = create_audio_diagnostics()
diag.run_all_diagnostics()
diag.print_report()
diag.save_report('./diagnostics/audio_report.json')
"

# キャプチャカード入力
python -c "
from capture.capture_card import create_capture_card_capture
from vision.roi import ROIExtractor
from trigger.engine import TriggerEngine

# キャプチャカードデバイスを開く
cap = create_capture_card_capture(device_index=0, resolution=(1920, 1080), fps=60)

with cap as capture:
    # 利用可能なデバイス一覧を表示
    devices = capture.list_capture_cards()
    print(f'Found {len(devices)} capture card devices')

    # フレームを読み込み
    frame = capture.read()
    print(f'Captured frame: {frame.shape}')
"
```

**今後の拡張:**
- リアルタイム自動キャリブレーション
- エコー検知の高度化（ループバック検出など）
- キャプチャカード専用の高度な設定（色空間、入力形式など）

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

