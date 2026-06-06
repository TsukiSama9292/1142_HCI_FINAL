# 1142_HCI_FINAL — 動畫瘋彈幕情緒分析

整合 NLP 情緒分類（GoEmotions 28 類）與時間窗聚合，支援批次下載、主角情緒 vs 彈幕情緒之延遲序列分析（LSA）。

## 環境設定

```bash
uv sync                  # 安裝主要依賴
uv sync --extra export   # (可選) 安裝 ONNX 匯出依賴
```

## 使用方式

### 單一影片彈幕情緒分類

```bash
# 基本用法（預設儲存至 logs/<sn>/nlp/）
uv run python3 main.py 14464

# 指定時間範圍
uv run python3 main.py 14464 --start-range 11:52~17:28

# 指定輸出位置
uv run python3 main.py 14464 --start-range 11:52~17:28 -o output.jsonl

# 只下載 JSONL 不做分類
uv run python3 main.py 14464 --start-range 11:52~17:28 --no-classify
```

### 批次處理所有影片

依據 `config/analysis_groups.json` 同時下載 6 部片：

```bash
uv run python3 main.py --from-config config/analysis_groups.json
```

### 前處理 Pipeline

內建四階段前處理（預設啟用，可透過 `--no-preprocess` 略過）：

1. **雜訊過濾**：剔除簽到、廣告、純時間數字等無意義彈幕
2. **NUMBER_SLANG 保護**：`233` / `555` / `666` 在 filter_noise 前比對，避免被數字過濾器誤殺
3. **文字正規化**：縮減重複標點（`????` → `??`）、疊詞（`哈哈哈` → `哈哈`）
4. **規則標記**：`XD` → 有趣、`我婆` → 愛、`臥槽` → 驚訝、`666` → 讚賞 等直接對應（機率 1.0，不經 NLP）
5. **NLP 分類**：僅規則無法覆蓋的語句送 DistilBERT ONNX 模型（多語言 BERT 微調 GoEmotions）

```
# 有前處理（預設）：XD → 有趣（規則，不經 NLP）
# 無前處理：XD → 中性（NLP 誤判為中性）
```

### CLI 參數一覽

| 參數 | 說明 |
|------|------|
| `sn` | 影片 SN 碼（`--from-config` 時可省略） |
| `--from-config <path>` | 批次處理：讀取群組設定檔，處理所有 SN |
| `--start-range MM:SS~MM:SS` | 過濾彈幕時間範圍 |
| `-o, --output <path>` | 輸出路徑（預設: `logs/<sn>/nlp/output.jsonl`） |
| `--threshold <float>` | 情緒分類信心門檻（預設: 0.3） |
| `--segment <sec>` | 時間窗聚合（如 `--segment 8` 輸出每 8 秒分佈） |
| `--no-classify` | 只下載原始 JSONL，不做分類 |
| `--no-preprocess` | 略過前處理 Pipeline，直接 NLP 分類 |

## 輸出格式

指定 `-o output.jsonl` 或省略時（預設 `logs/<sn>/nlp/output.jsonl`）會自動產生三份檔案：

| 檔案 | 內容 | 範例 |
|------|------|------|
| `output.jsonl` | 每條彈幕之情緒分數向量（Key 使用情緒編碼） | `{"emotions": {"@C": 0.9999}}` |
| `output_label.jsonl` | 每條彈幕單一情緒編碼 | `{"emotion": "@C"}` |
| `output_label_ana.jsonl` | 每 8 秒窗情緒編碼比例分佈（從大到小排序） | `{"emotion": {"@C": 0.875, "P": 0.125}}` |

## 情緒編碼表（GoEmotions 28 類）

| 編碼 | 情緒（英） | 情緒（中） | 類別 |
|------|-----------|-----------|------|
| A | Admiration | 讚賞 | 正面 |
| B | Amusement | 有趣 | 正面 |
| C | Approval | 認可 | 正面 |
| D | Caring | 關心 | 正面 |
| E | Desire | 慾望 | 正面 |
| F | Excitement | 興奮 | 正面 |
| G | Gratitude | 感激 | 正面 |
| H | Joy | 喜悅 | 正面 |
| I | Love | 愛 | 正面 |
| J | Optimism | 樂觀 | 正面 |
| K | Pride | 自豪 | 正面 |
| L | Relief | 寬慰 | 正面 |
| M | Anger | 憤怒 | 負面 |
| N | Annoyance | 煩惱 | 負面 |
| O | Disappointment | 失望 | 負面 |
| P | Disapproval | 不認可 | 負面 |
| Q | Disgust | 厭惡 | 負面 |
| R | Embarrassment | 尷尬 | 負面 |
| S | Fear | 恐懼 | 負面 |
| T | Grief | 悲痛 | 負面 |
| U | Nervousness | 緊張 | 負面 |
| V | Remorse | 自責 | 負面 |
| W | Sadness | 悲傷 | 負面 |
| X | Confusion | 困惑 | 模糊 |
| Y | Curiosity | 好奇 | 模糊 |
| Z | Realization | 領悟 | 模糊 |
| @A | Surprise | 驚訝 | 模糊 |
| @B | (None) | 不適用 | — |
| @C | Neutral | 中性 | 中性 |

## 主角情緒 vs 彈幕情緒 LSA 分析

### 主角情緒手動編碼

建立 `logs/<sn>/protagonist.jsonl`，每 8 秒窗一筆，`protagonist_emotion` 使用上方編碼表：

```jsonl
{"segment_start": "0:11:52.00", "segment_end": "0:12:00.00", "protagonist_emotion": "S"}
{"segment_start": "0:12:00.00", "segment_end": "0:12:08.00", "protagonist_emotion": "F"}
```

### 群組設定檔

`config/analysis_groups.json` 定義各敘事策略類型與對應 SN：

```json
{
  "groups": {
    "外掛爽感型": {
      "meta": { "strategy": "power_fantasy", "label": "Power Fantasy" },
      "sns": [
        { "sn": 31622, "title": "我想成為影之強者！ [5]", "short": "影之強者 第5集", "range": "10:00~15:36" },
        { "sn": 11351, "title": "關於我轉生變成史萊姆這檔事 [14]", "short": "史萊姆 第14集", "range": "9:10~14:46" }
      ]
    },
    "心理折磨型": {
      "meta": { "strategy": "psychological", "label": "Psychological Torture" },
      "sns": [
        { "sn": 14464, "title": "Re：從零開始的異世界生活 新編集版 [2B]", "short": "re:0 第2集B", "range": "11:52~17:28" },
        { "sn": 8649,  "title": "來自深淵 [10]", "short": "來自深淵 第10集", "range": "10:47~16:23" }
      ]
    },
    "搞笑解構型": {
      "meta": { "strategy": "comedy", "label": "Comedy Deconstruction" },
      "sns": [
        { "sn": 27444, "title": "與變成了異世界美少女的大叔一起冒險 [1]", "short": "異世界美少女大叔 第1集", "range": "2:17~5:53" },
        { "sn": 7296,  "title": "為美好的世界獻上祝福！ [3]", "short": "為美好世界 第3集", "range": "6:17~11:53" }
      ]
    }
  }
}
```

### 執行 LSA 分析

主角編碼完成後：

```bash
uv run python3 scripts/analyze.py --group-config config/analysis_groups.json -o results/
```

### 分析輸出目錄結構

```
results/
├── all_summaries.json                     # 各組摘要（共鳴比、平均 Cosine）
├── cross_group/                           # 跨組卡方檢定
│   └── 外掛爽感型_vs_心理折磨型.json
├── 全部_(All)/                            # 全部 6 部 pooled 分析
│   ├── summary.json
│   ├── transition_matrix_lag0.csv
│   ├── z_scores_lag0.csv
│   ├── significant_paths_lag0.jsonl
│   └── ...
├── 外掛爽感型/
│   ├── summary.json                       # 共鳴比例、平均 Cosine Similarity
│   ├── transition_matrix_lag0.csv         # 28×28 轉移計數矩陣
│   ├── z_scores_lag0.csv                 # Z-score 顯著性檢定
│   ├── adjusted_residuals_lag0.csv        # 調整後殘差
│   ├── significant_paths_lag0.jsonl       # Z > 1.96 的顯著路徑
│   ├── cosine_similarities.jsonl          # 每窗 Cosine Similarity
│   ├── protagonist_distribution.jsonl     # 主角情緒分佈統計
│   └── danmaku_distribution.jsonl         # 彈幕情緒分佈統計
├── 心理折磨型/
│   └── ...
└── 搞笑解構型/
    └── ...
```

### 分析項目對照

| 分析項目 | 對應產出 |
|----------|---------|
| 所有觀眾 ↔ 主角情緒 | `全部_(All)/` pooled 分析 |
| 同類型內觀眾 ↔ 主角 | 各群組轉移矩陣 + Z-score |
| 不同類型間觀眾 ↔ 主角 | `cross_group/` 卡方檢定 |
| 主角情緒分佈對比 | `*/protagonist_distribution.jsonl` |
| 彈幕情緒分佈對比 | `*/danmaku_distribution.jsonl` |
| Lag 0 / Lag 1 / Lag 2 延遲效應 | `*_lag{0,1,2}.csv` |

## 規則斷言關鍵字（DIRECT_EMOTION_RULES）

| 編碼 | 關鍵字範例 |
|------|-----------|
| A | 太神、跪了、天花板、經費爆炸、666、YYDS |
| B | 笑死、XD、233、www、哈哈、肚子好痛 |
| C | +1、確實、正解、👍、👏、真男人 |
| D | 加油、保重、辛苦了、撐下去 |
| E | 好想要、羨慕、此處有本 |
| F | 燃爆、太帥 |
| G | 謝謝、感謝、好人一生平安 |
| H | 太棒、開心、舒服了、好爽 |
| I | 婆爆、我婆、我老公、我推、老婆、我愛、暈了、好可愛、好萌 |
| J | 期待、坐等、敲碗、一定會 |
| K | 神作、最強、我的超人、天下第一 |
| L | 還好、幸好、鬆一口氣 |
| M | 氣死、可惡、幹、凎、操、肏 |
| N | 煩死、套路、拖戲、無聊 |
| O | 失望、可惜、爛尾、就這？ |
| P | 不合理、智商掉線、好醜 |
| Q | 噁心、太油、三小、殺小、沙小、尛、工三小 |
| R | 尷尬、腳趾、中二、不忍直視 |
| S | 胃痛、好可怕、毛骨悚然、不敢看 |
| T | 便當、虐爆、QAQ、QQ |
| U | 好緊張、怕、拜託不、挫 |
| V | 對不起、抱歉、我錯了 |
| W | 哭爆、哭了、鼻酸、嗚嗚、555 |
| X | 蛤？、不懂、WTF、何意味 |
| Y | 好奇、為什麼、求解 |
| Z | 原來如此、懂了、伏筆、恍然大悟 |
| @A | 真假、臥槽、靠北、夭壽、太扯了 |
| @C | 中性（無規則匹配時 NLP 回退預設） |

## 模型匯出

重新下載 HuggingFace 模型並匯出 ONNX：

```bash
uv sync --extra export
uv run python3 scripts/export_onnx.py
```

## 檔案結構

```
├── main.py                          # CLI 入口
├── pyproject.toml                   # 套件管理
├── config/
│   └── analysis_groups.json         # 群組設定
├── src/hci_analysis/
│   ├── danmaku.py                   # 彈幕下載（ani.gamer.com.tw API）
│   ├── emotions.py                  # 情緒分類模型 + 編碼表
│   ├── preprocess.py                # 四階段前處理 Pipeline
│   └── lsa.py                       # LSA 核心（轉移矩陣、Z-score、Cosine）
├── scripts/
│   ├── export_onnx.py               # 模型匯出
│   └── analyze.py                   # LSA 分析 CLI
├── logs/<sn>/
│   ├── nlp/
│   │   ├── output.jsonl             # 完整情緒分數
│   │   ├── output_label.jsonl       # 單一情緒編碼
│   │   └── output_label_ana.jsonl   # 8 秒窗聚合
│   └── protagonist.jsonl            # (手動) 主角情緒編碼
├── models/
│   └── SchuylerH/bert-multilingual-go-emtions/onnx/
│       ├── model.onnx               # ONNX 模型
│       ├── tokenizer.json
│       └── config.json
└── docs/
    └── 期末計劃書.md                  # 研究計畫書
```
