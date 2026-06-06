# aniGamerBS

巴哈姆特動畫瘋彈幕下載工具。透過每集動畫的 SN 碼，直接從動畫瘋 API 抓取彈幕並輸出為 `.ass` 格式字幕檔。

上游專案：[miyouzi/aniGamerPlus](https://github.com/miyouzi/aniGamerPlus) — 在原專案基礎上精簡為純彈幕下載工具，移除影片下載、Web 控制臺、代理、cookie 等其餘功能。

純爬蟲工具，不依賴 ffmpeg，不需要 cookie 或任何額外配置。

## 需求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)（套件管理）

## 使用方式

```bash
uv run python main.py <sn>
uv run python main.py <sn> -o output.ass
```

SN 碼為巴哈姆特動畫瘋每集影片網址中的數字 ID。

### 範例

```bash
uv run python main.py 49355
```

輸出 `danmu_49355.ass`，即為該集的彈幕字幕檔。

## 輸出格式

輸出為 ASS (Advanced SubStation Alpha) 格式，包含：

- 彈幕文字與時間軸
- 三種位置類型：Roll（捲動）、Top（頂部）、Bottom（底部）
- 原始彈幕顏色

## 授權

MIT
