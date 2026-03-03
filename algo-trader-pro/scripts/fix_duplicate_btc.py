#!/usr/bin/env python3
"""
Fix duplicate BTC positions in the database.

C'era un bug che permetteva di aprire 2 posizioni sullo stesso symbol (BTC).
Questo script rimuove il duplicato più recente per ripristinare uno stato coerente.

Esegui con: python scripts/fix_duplicate_btc.py
"""
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "database" / "algotrader.db"


def main():
    if not DB_PATH.exists():
        print(f"DB non trovato: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Trova posizioni aperte per symbol
    rows = conn.execute(
        "SELECT trade_id, symbol, entry_time, notional_value, entry_fee FROM trades WHERE status='open' ORDER BY entry_time"
    ).fetchall()

    by_symbol = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(dict(r))

    # Cerca duplicati (più di 1 posizione per symbol)
    to_cancel = []
    for sym, positions in by_symbol.items():
        if len(positions) > 1:
            # Tieni la prima, marca le altre come 'cancelled' (errore duplicato)
            for p in sorted(positions, key=lambda x: x["entry_time"])[1:]:
                to_cancel.append(p["trade_id"])
                print(f"Marca come cancelled (duplicato): {sym} trade_id={p['trade_id'][:8]}... entry={p['entry_time']}")

    if not to_cancel:
        print("Nessun duplicato trovato.")
        conn.close()
        return

    print(f"\nMarco {len(to_cancel)} trade come 'cancelled' (non verranno ripristinati al restart)...")
    for tid in to_cancel:
        conn.execute(
            "UPDATE trades SET status = 'cancelled', updated_at = datetime('now') WHERE trade_id = ?",
            (tid,),
        )
    conn.commit()
    conn.close()
    print("Fatto. Riavvia il bot per applicare le modifiche.")


if __name__ == "__main__":
    main()
