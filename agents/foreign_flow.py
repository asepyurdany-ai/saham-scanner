"""
Foreign Flow Agent — IDX foreign institutional net buy/sell tracker.
Foreign flow = #1 signal on IDX. Smart money = asing.

fetch_foreign_flow(date_str=None) -> dict
get_net_foreign(ticker, flow_data, avg_daily_volume) -> dict
format_foreign_summary(flow_data, watchlist) -> str
save_flow_data(flow_data) -> None
"""

import json
import os
import time
from datetime import datetime

import requests

IDX_API_URL = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"
IDX_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.idx.co.id",
}

FLOW_DATA_PATH = "data/foreign_flow_today.json"


def fetch_foreign_flow(date_str: str = None) -> dict:
    """
    Fetch foreign net buy/sell data from IDX API.

    Args:
        date_str: "YYYY-MM-DD" (default today UTC)

    Returns:
        {ticker: {"foreign_buy": int, "foreign_sell": int, "net_foreign": int}}
        Returns {} on failure after 3 attempts.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    params = {
        "date": date_str,
        "start": 0,
        "length": 100,
        "code": "",
    }

    for attempt in range(3):
        try:
            resp = requests.get(
                IDX_API_URL,
                params=params,
                headers=IDX_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # IDX API returns data under "data" or "Data"
            records = (
                data.get("data", [])
                or data.get("Data", [])
                or []
            )

            result = {}
            for row in records:
                code = str(row.get("StockCode", "") or "").strip()
                if not code:
                    continue
                foreign_buy = int(row.get("ForeignBuy", 0) or 0)
                foreign_sell = int(row.get("ForeignSell", 0) or 0)
                net = foreign_buy - foreign_sell
                result[code] = {
                    "foreign_buy": foreign_buy,
                    "foreign_sell": foreign_sell,
                    "net_foreign": net,
                }

            print(f"[ForeignFlow] Fetched {len(result)} stocks for {date_str}")
            return result

        except Exception as e:
            print(f"[ForeignFlow] Attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)

    print("[ForeignFlow] All retries failed — returning {}")
    return {}


def get_net_foreign(
    ticker: str, flow_data: dict, avg_daily_volume: float = 0
) -> dict:
    """
    Returns foreign flow signal for a single ticker.

    Signal logic:
    - STRONG BUY: net > 0 AND net > avg_daily_volume * 0.05
    - WEAK BUY:   net > 0 but below threshold
    - NEUTRAL:    net ~= 0
    - SELL:       net < 0

    Args:
        ticker: e.g. "BBCA.JK" or "BBCA"
        flow_data: result from fetch_foreign_flow()
        avg_daily_volume: used for STRONG/WEAK classification

    Returns:
        {"net_foreign": float, "signal": str, "strength": str}
    """
    clean = ticker.replace(".JK", "").upper()
    stock = flow_data.get(clean, {})
    net = float(stock.get("net_foreign", 0))

    strong_threshold = avg_daily_volume * 0.05 if avg_daily_volume > 0 else float("inf")

    if net > 0:
        if avg_daily_volume > 0 and net > strong_threshold:
            signal, strength = "BUY", "STRONG"
        else:
            signal, strength = "BUY", "WEAK"
    elif net < 0:
        if avg_daily_volume > 0 and abs(net) > strong_threshold:
            signal, strength = "SELL", "STRONG"
        else:
            signal, strength = "SELL", "WEAK"
    else:
        signal, strength = "NEUTRAL", "NEUTRAL"

    return {"net_foreign": net, "signal": signal, "strength": strength}


def format_foreign_summary(
    flow_data: dict, watchlist: list, threshold: int = 1_000_000
) -> str:
    """
    Summary of foreign flow for watchlist stocks.
    Only shows stocks with |net_foreign| > threshold (default 1 juta lembar).

    Args:
        flow_data: result from fetch_foreign_flow()
        watchlist: list of tickers e.g. ["BBCA.JK", ...]
        threshold: minimum absolute net_foreign to show (in shares, not IDR)

    Returns:
        Formatted string for Telegram
    """
    if not flow_data:
        return "🏦 Foreign flow: data tidak tersedia"

    buy_stocks = []
    sell_stocks = []

    for ticker in watchlist:
        clean = ticker.replace(".JK", "").upper()
        stock = flow_data.get(clean)
        if not stock:
            continue
        net = stock["net_foreign"]
        if abs(net) < threshold:
            continue
        net_juta = net / 1_000_000
        if net > 0:
            buy_stocks.append((clean, net_juta))
        else:
            sell_stocks.append((clean, net_juta))

    lines = ["🏦 <b>FOREIGN FLOW SUMMARY</b>"]

    if buy_stocks:
        lines.append("🟢 Net Buy:")
        for ticker, net in sorted(buy_stocks, key=lambda x: -x[1])[:5]:
            lines.append(f"  {ticker}: +Rp {net:.1f} juta")

    if sell_stocks:
        lines.append("🔴 Net Sell:")
        for ticker, net in sorted(sell_stocks, key=lambda x: x[1])[:5]:
            lines.append(f"  {ticker}: Rp {net:.1f} juta")

    if not buy_stocks and not sell_stocks:
        lines.append("  Tidak ada aktivitas foreign signifikan hari ini")

    return "\n".join(lines)


def save_flow_data(flow_data: dict):
    """Save foreign flow data to data/foreign_flow_today.json."""
    os.makedirs("data", exist_ok=True)
    with open(FLOW_DATA_PATH, "w") as f:
        json.dump(
            {
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "data": flow_data,
            },
            f,
            indent=2,
        )
    print(f"[ForeignFlow] Saved {len(flow_data)} records to {FLOW_DATA_PATH}")
