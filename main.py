"""
Trading Signals PRO — Motor principal
Corre cada 4 horas, analiza 5 mercados, manda alertas por Telegram
Datos: yfinance (gratuito, sin límites, sin API key necesaria)
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from anthropic import Anthropic

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_KEY", "")

CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "14400"))
SL_MULT  = float(os.environ.get("SL_MULT",  "2.0"))
TP_MULT  = float(os.environ.get("TP_MULT",  "3.5"))
RSI_LOW  = float(os.environ.get("RSI_LOW",  "38"))
RSI_HIGH = float(os.environ.get("RSI_HIGH", "62"))

# Mercados: clave interna → símbolo Yahoo Finance
MARKETS = {
    "XAUUSD": {"name": "Oro",      "emoji": "🥇", "only_long": True,  "yf": "GC=F"},
    "EURUSD": {"name": "EUR/USD",  "emoji": "💶", "only_long": False, "yf": "EURUSD=X"},
    "USDCHF": {"name": "USD/CHF",  "emoji": "🇨🇭", "only_long": False, "yf": "CHF=X"},
    "US500":  {"name": "S&P 500",  "emoji": "🏛️", "only_long": True,  "yf": "ES=F"},
    "NDAQ":   {"name": "Nasdaq",   "emoji": "📈", "only_long": True,  "yf": "NQ=F"},
}

# ── CLIENTE ANTHROPIC ─────────────────────────────────────────────────────────
claude = Anthropic(api_key=ANTHROPIC_KEY)


# ── DATOS: YFINANCE ───────────────────────────────────────────────────────────
def fetch_candles(symbol: str, yf_ticker: str) -> pd.DataFrame | None:
    """Descarga velas H4 via yfinance. Gratuito, sin límites."""
    try:
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period="2y", interval="1h", auto_adjust=True)

        if df.empty:
            log.warning(f"{symbol}: yfinance devolvió datos vacíos para {yf_ticker}")
            return None

        df = df.reset_index()
        df = df.rename(columns={"Datetime": "Date", "Date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)

        # Agregar H1 → H4
        df["H4_group"] = df["Date"].dt.floor("4h")
        df_h4 = df.groupby("H4_group").agg(
            Open   = ("Open",   "first"),
            High   = ("High",   "max"),
            Low    = ("Low",    "min"),
            Close  = ("Close",  "last"),
            Volume = ("Volume", "sum"),
        ).reset_index().rename(columns={"H4_group": "Date"})

        df_h4 = df_h4.dropna().reset_index(drop=True)
        log.info(f"{symbol}: {len(df_h4)} velas H4 de yfinance ({yf_ticker})")
        return df_h4

    except Exception as e:
        log.error(f"{symbol}: Error en yfinance — {e}")
        return None


# ── INDICADORES ───────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for span in [21, 50, 200, 12, 26]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    df["EMA200_slope"] = df["EMA200"] - df["EMA200"].shift(6)

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"]  - df["Close"].shift(1))
        )
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    df["MACD"]      = df["EMA12"] - df["EMA26"]
    df["MACD_sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["MACD_cross_up"]   = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)
    df["MACD_cross_down"] = (df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0)

    df["Regime"] = "LATERAL"
    df.loc[(df["EMA21"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"]), "Regime"] = "ALCISTA"
    df.loc[(df["EMA21"] < df["EMA50"]) & (df["EMA50"] < df["EMA200"]), "Regime"] = "BAJISTA"

    return df


# ── RECALIBRACIÓN DINÁMICA ────────────────────────────────────────────────────
def _quick_backtest(df: pd.DataFrame, sl_mult: float, tp_mult: float,
                    only_long: bool = True) -> list:
    dow = df["Date"].dt.dayofweek
    ok  = dow.between(0, 3)
    lm  = (df["Regime"] == "ALCISTA") & (df["EMA200_slope"] > 0) & \
          df["MACD_cross_up"] & (df["RSI"].between(RSI_LOW, RSI_HIGH)) & ok
    sm  = pd.Series(False, index=df.index) if only_long else \
          (df["Regime"] == "BAJISTA") & (df["EMA200_slope"] < 0) & \
          df["MACD_cross_down"] & (df["RSI"].between(RSI_LOW, RSI_HIGH)) & ok

    df = df.copy()
    df["Signal"] = 0
    df.loc[lm, "Signal"] =  1
    df.loc[sm, "Signal"] = -1

    trades, in_trade = [], False
    for i in range(50, len(df) - 1):
        sig = df["Signal"].iloc[i]
        if not in_trade and sig != 0:
            direction = sig
            atr       = df["ATR"].iloc[i]
            ep        = df["Close"].iloc[i]
            entry     = ep
            sl        = entry - sl_mult * atr if direction == 1 else entry + sl_mult * atr
            tp        = entry + tp_mult * atr if direction == 1 else entry - tp_mult * atr
            entry_idx = i
            in_trade  = True
        elif in_trade:
            h, l = df["High"].iloc[i], df["Low"].iloc[i]
            if (direction == 1 and l <= sl) or (direction == -1 and h >= sl):
                trades.append({"result": "LOSS", "pct": -abs(entry - sl) / entry * 100})
                in_trade = False
            elif (direction == 1 and h >= tp) or (direction == -1 and l <= tp):
                trades.append({"result": "WIN",  "pct":  abs(entry - tp) / entry * 100})
                in_trade = False
            elif (i - entry_idx) >= 12:
                pnl = (df["Close"].iloc[i] - entry) * direction / entry * 100
                trades.append({"result": "WIN" if pnl > 0 else "LOSS", "pct": pnl})
                in_trade = False
    return trades


def recalibrate_parameters(df: pd.DataFrame, lookback_days: int = 90) -> dict:
    cutoff = df["Date"].max() - timedelta(days=lookback_days)
    recent = df[df["Date"] >= cutoff].copy()

    if len(recent) < 50:
        return {"sl": SL_MULT, "tp": TP_MULT, "recalibrated": False, "pf_recent": 1.0}

    best_pf, best_sl, best_tp = 0, SL_MULT, TP_MULT
    for sl in [1.5, 2.0, 2.5]:
        for tp in [2.5, 3.0, 3.5, 4.0]:
            trades = _quick_backtest(recent, sl, tp)
            if len(trades) < 3:
                continue
            wins   = [t for t in trades if t["result"] == "WIN"]
            losses = [t for t in trades if t["result"] == "LOSS"]
            if not losses:
                continue
            pf = abs(sum(t["pct"] for t in wins)) / abs(sum(t["pct"] for t in losses))
            if pf > best_pf:
                best_pf, best_sl, best_tp = pf, sl, tp

    return {"sl": best_sl, "tp": best_tp, "pf_recent": round(best_pf, 2),
            "lookback_days": lookback_days, "recalibrated": True}


# ── ANÁLISIS DE SEÑAL ─────────────────────────────────────────────────────────
def analyze_signal(df: pd.DataFrame, params: dict, only_long: bool = True) -> dict:
    if len(df) < 210:
        return {"signal": "SIN_DATOS"}

    last   = df.iloc[-1]
    regime = last["Regime"]

    cond_ema   = regime != "LATERAL"
    cond_slope = (last["EMA200_slope"] > 0) if regime == "ALCISTA" else \
                 (last["EMA200_slope"] < 0) if regime == "BAJISTA" else False
    cond_macd  = bool(last["MACD_cross_up"])  if regime == "ALCISTA" else \
                 bool(last["MACD_cross_down"]) if regime == "BAJISTA" else False
    cond_rsi   = RSI_LOW <= last["RSI"] <= RSI_HIGH
    cond_dow   = last["Date"].dayofweek in [0, 1, 2, 3]
    conds_met  = sum([cond_ema, cond_slope, cond_macd, cond_rsi])

    long_ok  = regime == "ALCISTA" and cond_slope and cond_macd and cond_rsi and cond_dow
    short_ok = not only_long and regime == "BAJISTA" and cond_slope and cond_macd and cond_rsi and cond_dow

    sl_m = params.get("sl", SL_MULT)
    tp_m = params.get("tp", TP_MULT)

    if long_ok:       signal, direction = "LONG",    1
    elif short_ok:    signal, direction = "SHORT",  -1
    else:             signal, direction = "ESPERAR", 0

    entry = float(last["Close"])
    atr   = float(last["ATR"])

    return {
        "signal": signal, "direction": direction, "regime": regime,
        "price":  round(entry, 5), "rsi": round(float(last["RSI"]), 1),
        "atr":    round(atr, 5),   "macd_hist": round(float(last["MACD_hist"]), 6),
        "conds_met": conds_met,    "last_candle": str(last["Date"]),
        "entry": round(entry, 5),
        "sl":    round(entry - sl_m*atr if direction == 1 else entry + sl_m*atr, 5),
        "tp":    round(entry + tp_m*atr if direction == 1 else entry - tp_m*atr, 5),
        "sl_pct": round(sl_m * atr / entry * 100, 3),
        "tp_pct": round(tp_m * atr / entry * 100, 3),
        "sl_mult": sl_m, "tp_mult": tp_m,
        "conditions": {"ema_aligned": cond_ema, "ema_slope": cond_slope,
                       "macd_cross": cond_macd, "rsi_ok": cond_rsi, "day_ok": cond_dow},
    }


# ── ANÁLISIS IA ───────────────────────────────────────────────────────────────
def ai_analysis(symbol: str, market_info: dict, signal_data: dict,
                recal: dict, hist_trades: list) -> str:
    name  = market_info["name"]
    today = datetime.now().strftime("%A %d de %B de %Y")
    wins  = [t for t in hist_trades if t.get("result") == "WIN"]
    losses= [t for t in hist_trades if t.get("result") == "LOSS"]
    wr    = round(len(wins) / len(hist_trades) * 100, 1) if hist_trades else 0
    pf_v  = round(abs(sum(t["pct"] for t in wins)) / abs(sum(t["pct"] for t in losses) or 1), 2)
    recent= ", ".join(f"{'WIN' if t['result']=='WIN' else 'LOSS'}({t['pct']:+.2f}%)"
                      for t in hist_trades[-5:]) if hist_trades else "Sin historial"

    prompt = f"""Analista de trading profesional. Hoy: {today}.

{name} ({symbol}) — Señal: {signal_data['signal']}
Precio: {signal_data['price']} | RSI: {signal_data['rsi']} | Régimen: {signal_data['regime']}
Condiciones: {signal_data['conds_met']}/4 | WR reciente: {wr}% | PF: {pf_v}
Últimas ops: {recent}

Busca info actual y responde en español, máximo 200 palabras:
1. **MACRO HOY**: factores clave para {name}
2. **SEÑAL {signal_data['signal']}**: ¿tiene sentido con el contexto?
3. **CONFIANZA** (0-100%): número + razón breve
4. **VEREDICTO**: operar o esperar"""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        )
        return " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
    except Exception as e:
        log.error(f"Error IA: {e}")
        return f"Análisis IA no disponible ({e})"


# ── CONFIANZA ─────────────────────────────────────────────────────────────────
def compute_confidence(signal: dict, recal: dict, hist_trades: list) -> int:
    score = signal["conds_met"] * 10
    if hist_trades:
        wins = [t for t in hist_trades if t.get("result") == "WIN"]
        score += int(len(wins) / len(hist_trades) * 30)
    pf = recal.get("pf_recent", 1.0)
    if isinstance(pf, float):
        score += min(20, int((pf - 1.0) * 20))
    if signal["regime"] != "LATERAL":
        score += 10
    return min(100, score)


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
        if r.ok:
            log.info("✅ Telegram enviado")
            return True
        log.error(f"Telegram: {r.status_code} {r.text[:100]}")
        return False
    except Exception as e:
        log.error(f"Telegram: {e}")
        return False


def format_signal_message(symbol: str, market: dict, signal: dict,
                          recal: dict, confidence: int, ai_text: str) -> str:
    if signal["signal"] not in ("LONG", "SHORT"):
        return ""
    is_long = signal["signal"] == "LONG"
    color   = "🟢" if is_long else "🔴"
    arrow   = "📈 LONG COMPRAR" if is_long else "📉 SHORT VENDER"
    bar     = "█" * (confidence // 10) + "░" * (10 - confidence // 10)
    c       = signal["conditions"]
    conds   = (f"  {'✅' if c['ema_aligned'] else '❌'} EMA alineadas\n"
               f"  {'✅' if c['ema_slope']   else '❌'} EMA200 pendiente\n"
               f"  {'✅' if c['macd_cross']  else '❌'} MACD cruce\n"
               f"  {'✅' if c['rsi_ok']      else '❌'} RSI {signal['rsi']}")
    recal_s = f"SL×{recal['sl']} TP×{recal['tp']} PF:{recal.get('pf_recent','?')}"
    ai_s    = " ".join(ai_text.split()[:50]) + "..." if len(ai_text.split()) > 50 else ai_text
    ai_s    = ai_s.replace("**", "*")

    return (f"{color} *SEÑAL — {market['emoji']} {market['name']} ({symbol})* {color}\n"
            f"{arrow}\n\n"
            f"📊 *Confianza: {confidence}%* `{bar}`\n\n"
            f"💰 *Niveles:*\n"
            f"  Entrada:     `{signal['entry']}`\n"
            f"  Stop Loss:   `{signal['sl']}` (-{signal['sl_pct']}%)\n"
            f"  Take Profit: `{signal['tp']}` (+{signal['tp_pct']}%)\n"
            f"  {recal_s}\n\n"
            f"📋 *Condiciones ({signal['conds_met']}/4):*\n{conds}\n\n"
            f"🤖 _{ai_s}_\n\n"
            f"⏰ _{signal['last_candle']}_ | _Max 2% capital/op_")


# ── LOG ───────────────────────────────────────────────────────────────────────
SIGNALS_LOG = os.environ.get("SIGNALS_LOG_PATH", "/data/signals_log.json")

def save_signal(symbol: str, signal: dict, confidence: int, ai_text: str):
    try:
        os.makedirs(os.path.dirname(SIGNALS_LOG), exist_ok=True)
        data = json.load(open(SIGNALS_LOG)) if os.path.exists(SIGNALS_LOG) else []
        data.append({"ts": datetime.now().isoformat(), "symbol": symbol,
                     "signal": signal["signal"], "price": signal["price"],
                     "entry": signal["entry"], "sl": signal["sl"], "tp": signal["tp"],
                     "confidence": confidence, "rsi": signal["rsi"],
                     "regime": signal["regime"], "ai": ai_text[:200], "result": None})
        json.dump(data[-500:], open(SIGNALS_LOG, "w"), indent=2)
    except Exception as e:
        log.warning(f"Log no guardado: {e}")


# ── CICLO PRINCIPAL ───────────────────────────────────────────────────────────
def run_cycle():
    log.info("=" * 50)
    log.info(f"Ciclo — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 50)

    found = []
    for symbol, mkt in MARKETS.items():
        log.info(f"→ {symbol} ({mkt['name']})...")

        df = fetch_candles(symbol, mkt["yf"])
        if df is None or len(df) < 250:
            log.warning(f"{symbol}: solo {len(df) if df is not None else 0} velas, saltando")
            continue

        df     = calculate_indicators(df)
        recal  = recalibrate_parameters(df)
        signal = analyze_signal(df, recal, mkt["only_long"])

        log.info(f"{symbol}: {signal['signal']} | RSI={signal.get('rsi','?')} | "
                 f"Conds={signal.get('conds_met','?')}/4 | {signal.get('regime','?')}")

        if signal["signal"] == "SIN_DATOS":
            continue

        cutoff   = df["Date"].max() - timedelta(days=180)
        hist     = _quick_backtest(df[df["Date"] >= cutoff].copy(),
                                   recal["sl"], recal["tp"], mkt["only_long"])
        conf     = compute_confidence(signal, recal, hist)
        log.info(f"{symbol}: confianza={conf}%")

        if signal["signal"] != "ESPERAR":
            log.info(f"⚡ {symbol} {signal['signal']} — analizando con IA...")
            ai = ai_analysis(symbol, mkt, signal, recal, hist)
            msg = format_signal_message(symbol, mkt, signal, recal, conf, ai)
            if msg:
                send_telegram(msg)
                save_signal(symbol, signal, conf, ai)
                found.append(f"{symbol} {signal['signal']}")

        time.sleep(2)

    if not found:
        log.info("Sin señales — todos en ESPERAR")
        now_utc    = datetime.now(timezone.utc).replace(tzinfo=None)
        now_madrid = now_utc + timedelta(hours=2)  # UTC+2 verano España
        # Resumen matinal 8:00-8:20h Madrid, solo lunes-viernes
        if now_madrid.hour == 8 and now_madrid.minute < 20 and now_madrid.weekday() < 5:
            market_summary = "\n".join(
                f"  {MARKETS[s]['emoji']} {MARKETS[s]['name']}: ESPERAR"
                for s in MARKETS
            )
            send_telegram(
                f"☀️ *Buenos días — {now_madrid.strftime('%d/%m/%Y %H:%M')} Madrid*\n\n"
                f"Sin señales activas. Estado de mercados:\n{market_summary}\n\n"
                f"_Análisis automático cada {CHECK_INTERVAL//3600}h. Te aviso si hay señal._ ✅"
            )
    else:
        log.info(f"Enviadas: {', '.join(found)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("🚀 Trading Signals PRO")
    log.info(f"Mercados: {list(MARKETS.keys())}")
    log.info(f"Fuente: yfinance (gratis, sin API key)")
    log.info(f"Intervalo: {CHECK_INTERVAL//3600}h | SL×{SL_MULT} TP×{TP_MULT}")
    log.info(f"Telegram: {'✅' if TELEGRAM_TOKEN else '⚠️  no configurado'}")

    # Mensaje de prueba si está activado via variable de entorno
    if os.environ.get("SEND_TEST_MESSAGE", "").lower() == "true":
        log.info("🧪 Modo prueba — enviando mensaje de test a Telegram...")
        now_madrid = datetime.now(timezone.utc) + timedelta(hours=2)
        ok = send_telegram(
            f"✅ *Trading Signals PRO — Test OK*\n\n"
            f"🚀 El sistema está funcionando correctamente.\n\n"
            f"*Configuración activa:*\n"
            f"  • Mercados: {', '.join(MARKETS.keys())}\n"
            f"  • Intervalo: cada {CHECK_INTERVAL//3600}h\n"
            f"  • SL: {SL_MULT}×ATR | TP: {TP_MULT}×ATR\n"
            f"  • Hora Madrid: {now_madrid.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"_Cuando haya señal recibirás un mensaje como este pero con los niveles de entrada, SL y TP._\n\n"
            f"⚠️ Recuerda borrar la variable SEND_TEST_MESSAGE en Railway."
        )
        if ok:
            log.info("✅ Mensaje de prueba enviado correctamente a Telegram")
        else:
            log.error("❌ Error enviando mensaje de prueba — revisa TELEGRAM_TOKEN y TELEGRAM_CHAT_ID")

    while True:
        try:
            run_cycle()
        except KeyboardInterrupt:
            log.info("Detenido")
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            send_telegram(f"⚠️ Error\n`{str(e)[:200]}`\nReintentando en 30 min...")
            time.sleep(1800)
            continue
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
