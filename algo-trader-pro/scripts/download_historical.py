#!/usr/bin/env python3
"""
AlgoTrader Pro - Download Dati Storici
=======================================
Scarica dati OHLCV storici e li salva in formato Parquet.

Uso:
    python scripts/download_historical.py
    python scripts/download_historical.py --symbols BTC ETH SOL --days 365
    python scripts/download_historical.py --provider bybit --timeframe 60
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Aggiungo root al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AlgoTrader Pro — Download Dati Storici           ║
╚══════════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scarica dati OHLCV storici per il training ML e il backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Scarica BTC e ETH, ultimi 2 anni (default)
  python scripts/download_historical.py

  # Solo BTC, ultimi 6 mesi
  python scripts/download_historical.py --symbols BTC --days 180

  # Tutti gli asset, 3 anni, timeframe 4h
  python scripts/download_historical.py --days 1095 --timeframe 240

  # Forza ri-download anche se cache esistente
  python scripts/download_historical.py --force
        """
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTC", "ETH", "SOL", "BNB", "XRP"],
        help="Simboli da scaricare (default: BTC ETH SOL BNB XRP)"
    )
    parser.add_argument(
        "--days", type=int, default=730,
        help="Numero di giorni storici da scaricare (default: 730 = 2 anni)"
    )
    parser.add_argument(
        "--timeframe", type=int, default=60,
        choices=[1, 5, 15, 30, 60, 240, 1440],
        help="Timeframe in minuti (default: 60 = 1h)"
    )
    parser.add_argument(
        "--provider", default=None,
        choices=["bybit", "coingecko"],
        help="Override provider (default: usa DATA_PROVIDER dal .env)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forza ri-download ignorando la cache esistente"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory output (default: database/ohlcv_cache)"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Progress bar semplice (no dipendenze esterne)
# ──────────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, prefix: str = "", width: int = 40) -> str:
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / max(total, 1)
    return f"\r{prefix} [{bar}] {pct:5.1f}% ({current}/{total})"


# Mappa timeframe (minuti) → stringa Bybit per API
TIMEFRAME_MAP = {
    1: "1m", 5: "5m", 15: "15m", 30: "15m",  # 30m non supportato, usa 15m
    60: "1h", 240: "4h", 1440: "1d",
}
# Mappa timeframe (minuti) → chiave store (deve coincidere con engine: 5→"5", 60→"60")
STORE_TIMEFRAME_MAP = {
    1: "1", 5: "5", 15: "15", 30: "15",
    60: "60", 240: "240", 1440: "D",
}


# ──────────────────────────────────────────────────────────────────
# Download singolo simbolo (usa HistoricalDownloader per Bybit)
# ──────────────────────────────────────────────────────────────────

def download_symbol(
    symbol: str,
    timeframe: int,
    days: int,
    provider_name: str,
    ohlcv_store,
    force: bool,
) -> dict:
    """
    Scarica dati OHLCV per un simbolo e li salva in Parquet.
    Ritorna statistiche del download.
    """
    start_time = time.time()
    pair = f"{symbol}USDT"
    tf_str = TIMEFRAME_MAP.get(timeframe, "1h")
    store_tf = STORE_TIMEFRAME_MAP.get(timeframe, "60")

    # Calcola range temporale
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    # Force: elimina cache prima di scaricare
    if force:
        ohlcv_store.delete(pair, store_tf)
        print(f"  📥 {pair} — download completo {days} giorni (force)...")
    else:
        latest_ms = ohlcv_store.get_latest_timestamp(pair, store_tf)
        if latest_ms is not None:
            latest_dt = datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc)
            gap_hours = (end_dt - latest_dt).total_seconds() / 3600
            if gap_hours < (timeframe / 60) * 2:
                print(f"  ✅ {pair} già aggiornato (ultimo: {latest_dt.strftime('%Y-%m-%d %H:%M')})")
                return {"symbol": symbol, "status": "cached", "candles": 0, "elapsed": 0}
            print(f"  🔄 {pair} — aggiorno dal {latest_dt.strftime('%Y-%m-%d %H:%M')} (gap: {gap_hours:.0f}h)")
        else:
            print(f"  📥 {pair} — download completo {days} giorni...")

    try:
        if provider_name == "bybit":
            from src.data.providers.bybit_rest import BybitRESTProvider
            from src.data.historical.downloader import HistoricalDownloader

            provider = BybitRESTProvider()
            downloader = HistoricalDownloader(provider=provider, store=ohlcv_store)
            df = downloader.download_symbol(pair, tf_str, start_date, end_date)

        elif provider_name == "coingecko":
            import asyncio
            from src.data.providers.coingecko_rest import CoinGeckoRestProvider

            async def _fetch():
                prov = CoinGeckoRestProvider()
                await prov.connect()
                try:
                    return await prov.fetch_historical_ohlcv(
                        symbol=pair,
                        timeframe=tf_str,
                        start_date=start_dt,
                        end_date=end_dt,
                    )
                finally:
                    await prov.disconnect()

            df = asyncio.run(_fetch())

        else:
            print(f"  ⚠️  Provider {provider_name} non supportato. Usa --provider bybit o coingecko")
            return {"symbol": symbol, "status": "error", "candles": 0, "elapsed": time.time() - start_time}

        if not df.empty:
            ohlcv_store.save(df, pair, store_tf)

        elapsed = time.time() - start_time
        if df.empty:
            print(f"  ⚠️  Nessun dato scaricato per {pair}")
            return {"symbol": symbol, "status": "empty", "candles": 0, "elapsed": elapsed}

        print(f"  ✅ {pair}: {len(df):,} candele salvate in {elapsed:.1f}s")
        date_from = datetime.fromtimestamp(df["open_time"].min() / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        date_to = datetime.fromtimestamp(df["open_time"].max() / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        return {
            "symbol": symbol,
            "status": "ok",
            "candles": len(df),
            "elapsed": elapsed,
            "date_from": date_from,
            "date_to": date_to,
        }
    except Exception as e:
        print(f"  ❌ Errore {pair}: {e}")
        return {"symbol": symbol, "status": "error", "candles": 0, "elapsed": time.time() - start_time}


# ──────────────────────────────────────────────────────────────────
# Stampa riepilogo finale
# ──────────────────────────────────────────────────────────────────

def print_summary(results: list, total_elapsed: float):
    print("\n" + "═" * 60)
    print("  RIEPILOGO DOWNLOAD")
    print("═" * 60)
    print(f"  {'Simbolo':<8} {'Stato':<10} {'Candele':>10} {'Da':<12} {'A':<12} {'Tempo':>7}")
    print("  " + "─" * 56)

    total_candles = 0
    for r in results:
        status_icon = {"ok": "✅", "cached": "💾", "empty": "⚠️ ", "error": "❌"}.get(r["status"], "?")
        candles = r.get("candles", 0)
        total_candles += candles
        date_from = r.get("date_from", "—")
        date_to = r.get("date_to", "—")
        elapsed = r.get("elapsed", 0)
        print(f"  {status_icon} {r['symbol']:<6} {r['status']:<10} {candles:>10,} {date_from:<12} {date_to:<12} {elapsed:>6.1f}s")

    print("  " + "─" * 56)
    print(f"  {'TOTALE':<8} {'':10} {total_candles:>10,} {'':12} {'':12} {total_elapsed:>6.1f}s")
    print("═" * 60)

    if total_candles > 0:
        print(f"\n✅ Download completato! {total_candles:,} candele totali")
        print(f"   Cache: database/ohlcv_cache/")
        print(f"\n→ Prossimo passo: python scripts/train_models.py\n")
    else:
        print(f"\n⚠️  Nessuna candela scaricata. Verifica la connessione e le API key.\n")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    args = parse_args()

    # Carico config e .env
    print("⚙️  Carico configurazione...\n")
    try:
        from src.utils.config_loader import get_config
        cfg = get_config()
    except Exception as e:
        print(f"  ⚠️  Config non caricata ({e}), uso .env direttamente")

    # Determina provider
    import os
    provider_name = args.provider or os.getenv("DATA_PROVIDER", "coingecko")
    print(f"  Provider: {provider_name.upper()}")
    print(f"  Simboli:  {', '.join(args.symbols)}")
    print(f"  Timeframe: {args.timeframe}m ({args.timeframe//60}h)" if args.timeframe >= 60 else f"  Timeframe: {args.timeframe}m")
    print(f"  Periodo:  {args.days} giorni ({args.days/365:.1f} anni)")
    print(f"  Force:    {'Sì' if args.force else 'No (usa cache)'}\n")

    # Init database e store
    try:
        from src.utils.db import init_db
        init_db()
    except Exception as e:
        print(f"  ⚠️  DB init: {e}")

    from src.data.cache.ohlcv_store import OHLCVStore
    cache_dir = args.output_dir or str(ROOT / "database" / "ohlcv_cache")
    ohlcv_store = OHLCVStore(cache_dir=cache_dir)

    # ── Download loop ────────────────────────────────────────────
    results = []
    total_start = time.time()

    for i, symbol in enumerate(args.symbols, 1):
        print(f"[{i}/{len(args.symbols)}] Scarico {symbol}...")
        result = download_symbol(
            symbol=symbol,
            timeframe=args.timeframe,
            days=args.days,
            provider_name=provider_name,
            ohlcv_store=ohlcv_store,
            force=args.force,
        )
        results.append(result)
        print()  # Riga vuota tra simboli

    total_elapsed = time.time() - total_start
    print_summary(results, total_elapsed)


if __name__ == "__main__":
    main()
