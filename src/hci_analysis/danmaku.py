import json
import sys

import requests


def ts_format(dsec: int) -> str:
    sec = int(dsec / 10)
    cs = dsec % 10
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:d}0"


def parse_time(tstr: str) -> int:
    parts = [int(x) for x in tstr.split(":")]
    if len(parts) == 2:
        return parts[0] * 600 + parts[1] * 10
    if len(parts) == 3:
        return parts[0] * 36000 + parts[1] * 600 + parts[2] * 10
    raise ValueError(f"Invalid time format: {tstr}")


def download_danmaku(
    sn: int,
    output_path: str | None = None,
    start_range: tuple[int, int] | None = None,
) -> list[dict]:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "origin": "https://ani.gamer.com.tw",
        "authority": "ani.gamer.com.tw",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    r = requests.post(
        "https://ani.gamer.com.tw/ajax/danmuGet.php",
        data={"sn": str(sn)},
        headers=headers,
    )
    if r.status_code != 200:
        print(f"彈幕下載失敗: status_code={r.status_code}", file=sys.stderr)
        sys.exit(1)

    items: list[dict] = json.loads(r.text)

    if start_range:
        start_dsec, end_dsec = start_range
        items = [it for it in items if start_dsec <= it["time"] <= end_dsec]

    records = [{"start": ts_format(item["time"]), "text": item["text"]} for item in items]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"JSONL 已儲存: {output_path}")

    return records
