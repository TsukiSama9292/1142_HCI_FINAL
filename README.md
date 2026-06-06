# 1142_HCI_FINAL — 動畫瘋彈幕情緒分析

## 環境設定

```bash
uv sync                  # 安裝主要依賴
uv sync --extra export   # (可選) 安裝 ONNX 匯出依賴
```

## 使用方式

### 基本彈幕情緒分類

```bash
# 下載 SN=49355 的彈幕並標記情緒
uv run python3 main.py 49355

# 指定時間範圍 + 輸出檔案
uv run python3 main.py 49355 --start-range 12:00~15:00 -o output.jsonl

# 只下載 JSONL 不做分類
uv run python3 main.py 49355 --no-classify
```

### 時間窗聚合 (與手動主角編碼比對)

```bash
# 每 8 秒為一窗，輸出該窗彈幕情緒分佈
uv run python3 main.py 49355 --segment 8 -o segments.jsonl
```

輸出格式：`{"segment_start":"0:12:48.00","segment_end":"0:12:56.00","danmaku_count":6,"emotion_distribution":{"中性":0.8333,"有趣":0.1667}}`

### 模型匯出

重新下載並匯出 ONNX 模型：

```bash
uv sync --extra export
uv run python3 scripts/export_onnx.py
```
