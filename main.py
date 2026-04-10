"""
Trading Signals PRO — Sistema completo
- Analiza 5 mercados cada 4 horas con yfinance
- Filtra eventos macro del calendario económico
- Dashboard web con URL pública en Railway
- Alertas Telegram con análisis IA
"""

import os
import json
import time
import logging
import threading
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from anthropic import Anthropic
from flask import Flask, jsonify, render_template_string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_KEY", "")
CHECK_INTERVAL   = int(os.environ.get("CHECK_INTERVAL", "14400"))
SL_MULT  = float(os.environ.get("SL_MULT",  "2.0"))
TP_MULT  = float(os.environ.get("TP_MULT",  "3.5"))
RSI_LOW  = float(os.environ.get("RSI_LOW",  "38"))
RSI_HIGH = float(os.environ.get("RSI_HIGH", "62"))
PORT     = int(os.environ.get("PORT", "8080"))

MARKETS = {
    "XAUUSD": {"name": "Oro",      "emoji": "🥇", "only_long": True,  "yf": "GC=F"},
    "EURUSD": {"name": "EUR/USD",  "emoji": "💶", "only_long": False, "yf": "EURUSD=X"},
    "USDCHF": {"name": "USD/CHF",  "emoji": "🇨🇭", "only_long": False, "yf": "CHF=X"},
    "US500":  {"name": "S&P 500",  "emoji": "🏛️", "only_long": True,  "yf": "ES=F"},
    "NDAQ":   {"name": "Nasdaq",   "emoji": "📈", "only_long": True,  "yf": "NQ=F"},
}

claude = Anthropic(api_key=ANTHROPIC_KEY)

# Estado global compartido entre el hilo de análisis y el servidor web
STATE = {
    "last_update":        None,
    "next_update":        None,
    "markets":            {},
    "signals_today":      [],
    "macro_events":       [],
    "macro_blocked":      False,
    "signals_log":        [],
    "system_ok":          True,
    "consecutive_losses": 0,
    "current_risk_pct":   2.0,
    "risk_mode":          "normal",
    "streak_since":       None,
    "vix":                None,
    "fed_bce_week":       False,
    "fed_bce_reason":     "",
    "risk_reason":        "Iniciando...",
}

# Parámetros de gestión de rachas
STREAK_REDUCE_AFTER = int(os.environ.get("STREAK_REDUCE_AFTER", "3"))
RISK_REDUCED_PCT    = float(os.environ.get("RISK_REDUCED_PCT", "1.0"))
RISK_BASE_PCT       = float(os.environ.get("RISK_BASE_PCT", "2.0"))

# Umbrales de filtros adicionales
VIX_THRESHOLD       = float(os.environ.get("VIX_THRESHOLD", "25.0"))
VIX_EXTREME         = float(os.environ.get("VIX_EXTREME", "35.0"))
BRIDGE_URL          = os.environ.get("BRIDGE_URL", "")        # URL ngrok del bridge MT5
BRIDGE_SECRET       = os.environ.get("BRIDGE_SECRET", "")     # clave secreta bridge

SIGNALS_LOG = os.environ.get("SIGNALS_LOG_PATH", "/data/signals_log.json")


# ══════════════════════════════════════════════════════════════════════════════
# CALENDARIO ECONÓMICO
# ══════════════════════════════════════════════════════════════════════════════

def fetch_macro_events() -> list:
    """
    Descarga eventos macro de alto impacto del día desde investing.com/forexfactory
    via Claude con búsqueda web.
    Si falla, devuelve lista vacía (estrategia sigue funcionando).
    """
    today = datetime.now(timezone.utc).strftime("%A %d %B %Y")
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content":
                f"Busca el calendario económico de hoy {today}. "
                f"Lista SOLO los eventos de impacto ALTO (rojo) de hoy, "
                f"especialmente: Fed, BCE, NFP, CPI, GDP, PMI, tasas de interés. "
                f"Responde SOLO con JSON array: "
                f'[{{"time":"HH:MM UTC","event":"nombre","impact":"HIGH","currency":"USD"}}]. '
                f"Si no hay eventos de alto impacto hoy, responde: []"
            }]
        )
        text = " ".join(b.text for b in response.content if hasattr(b, "text")).strip()
        # Extraer JSON del texto
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start >= 0 and end > start:
            events = json.loads(text[start:end])
            log.info(f"Calendario: {len(events)} eventos macro de alto impacto hoy")
            return events
    except Exception as e:
        log.warning(f"Calendario: no disponible — {e}")
    return []


def is_macro_blocked(events: list) -> tuple[bool, str]:
    """Devuelve (True, razon) si estamos dentro de ventana ±2h de evento macro."""
    now_utc = datetime.now(timezone.utc)
    for ev in events:
        try:
            t = datetime.strptime(ev.get("time", ""), "%H:%M").replace(
                year=now_utc.year, month=now_utc.month, day=now_utc.day,
                tzinfo=timezone.utc
            )
            diff_hours = abs((now_utc - t).total_seconds()) / 3600
            if diff_hours <= 2:
                return True, f"{ev.get('event','Evento')} ({ev.get('time','?')} UTC) — ventana ±2h"
        except Exception:
            continue
    return False, ""


def fetch_vix() -> float:
    """
    Descarga el VIX actual desde yfinance.
    VIX = índice de volatilidad del S&P500.
    >25 = mercados nerviosos | >35 = pánico
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="2d", interval="1h")
        if not hist.empty:
            val = float(hist["Close"].iloc[-1])
            log.info(f"VIX actual: {val:.1f}")
            STATE["vix"] = round(val, 1)
            return val
    except Exception as e:
        log.warning(f"VIX no disponible: {e}")
    STATE["vix"] = None
    return 0.0


def check_fed_bce_week() -> tuple[bool, str]:
    """
    Comprueba si esta semana hay reunión Fed o BCE usando Claude.
    Si es semana de banco central → reducir riesgo toda la semana.
    """
    today = datetime.now(timezone.utc)
    # Fed: normalmente semanas 2 y 6 de cada mes, martes-miércoles
    # BCE: cada 6 semanas aprox, jueves
    # Usamos Claude para confirmarlo con búsqueda web
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content":
                f"¿Esta semana (semana del {today.strftime('%d %B %Y')}) hay reunión "
                f"de la Fed (FOMC) o del BCE? Responde SOLO: SI o NO y el evento."
            }]
        )
        text = " ".join(b.text for b in response.content if hasattr(b, "text")).strip().upper()
        is_central_bank_week = text.startswith("SI") or "FOMC" in text or "FED" in text or "BCE" in text
        reason = text[:100] if is_central_bank_week else ""
        if is_central_bank_week:
            log.info(f"⚠️ Semana de banco central: {reason}")
            STATE["fed_bce_week"] = True
            STATE["fed_bce_reason"] = reason
        else:
            STATE["fed_bce_week"] = False
            STATE["fed_bce_reason"] = ""
        return is_central_bank_week, reason
    except Exception as e:
        log.warning(f"Check Fed/BCE: {e}")
        STATE["fed_bce_week"] = False
        return False, ""


def get_risk_level(events: list, vix: float, fed_bce_week: bool) -> tuple[float, str]:
    """
    Determina el % de riesgo según las 3 capas + racha:

    Capa 1 — Evento ±2h:        0.0% (bloqueo total)
    Capa 2 — VIX > 35:          0.0% (pánico extremo)
    Capa 3 — VIX > 25:          0.5% (mercados nerviosos)
    Capa 4 — Semana Fed/BCE:     1.0% (riesgo reducido semana)
    Capa 5 — Racha ≥3 pérdidas: 1.0% (gestión de rachas)
    Normal:                      2.0%
    """
    blocked, block_reason = is_macro_blocked(events)

    if blocked:
        return 0.0, f"🚫 BLOQUEADO: {block_reason}"

    if vix >= VIX_EXTREME:
        return 0.0, f"🚫 VIX EXTREMO ({vix:.1f} > {VIX_EXTREME}) — mercado en pánico"

    if vix >= VIX_THRESHOLD:
        risk = min(RISK_REDUCED_PCT, 0.5)
        return risk, f"⚠️ VIX ALTO ({vix:.1f} > {VIX_THRESHOLD}) — riesgo reducido al {risk}%"

    if fed_bce_week:
        return RISK_REDUCED_PCT, f"⚠️ Semana Fed/BCE — riesgo reducido al {RISK_REDUCED_PCT}%"

    if STATE["consecutive_losses"] >= STREAK_REDUCE_AFTER:
        return RISK_REDUCED_PCT, f"⚠️ Racha {STATE['consecutive_losses']} pérdidas — riesgo reducido"

    return RISK_BASE_PCT, f"✅ Condiciones normales — riesgo {RISK_BASE_PCT}%"



# ══════════════════════════════════════════════════════════════════════════════
# GESTIÓN DE RACHAS Y TAMAÑO DE POSICIÓN
# ══════════════════════════════════════════════════════════════════════════════

def update_streak(result: str):
    """Actualiza el contador de pérdidas consecutivas y ajusta el riesgo."""
    if result == "LOSS":
        STATE["consecutive_losses"] += 1
        if STATE["streak_since"] is None:
            STATE["streak_since"] = datetime.now().strftime("%d/%m/%Y")
    else:
        STATE["consecutive_losses"] = 0
        STATE["streak_since"] = None
    STATE["current_risk_pct"] = RISK_REDUCED_PCT if STATE["consecutive_losses"] >= STREAK_REDUCE_AFTER else RISK_BASE_PCT
    STATE["risk_mode"] = "reducido" if STATE["consecutive_losses"] >= STREAK_REDUCE_AFTER else "normal"


def get_position_size(capital: float, entry: float, sl: float) -> dict:
    """Calcula tamaño de posición y riesgo en dinero según estado de racha."""
    risk_pct    = STATE["current_risk_pct"]
    risk_amount = capital * risk_pct / 100
    sl_pct      = abs(entry - sl) / entry * 100
    note = (f"⚠️ REDUCIDO — {STATE['consecutive_losses']} pérdidas seguidas desde {STATE['streak_since']}"
            if STATE["risk_mode"] == "reducido"
            else f"✅ Normal ({risk_pct}% del capital)")
    return {
        "risk_pct": risk_pct, "risk_usd": round(risk_amount, 2),
        "risk_mode": STATE["risk_mode"],
        "consecutive_losses": STATE["consecutive_losses"],
        "sl_distance_pct": round(sl_pct, 3), "note": note,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATOS Y ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_candles(symbol: str, yf_ticker: str) -> pd.DataFrame | None:
    try:
        ticker = yf.Ticker(yf_ticker)
        df = ticker.history(period="2y", interval="1h", auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={"Datetime": "Date", "Date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)
        df["H4_group"] = df["Date"].dt.floor("4h")
        df_h4 = df.groupby("H4_group").agg(
            Open=("Open","first"), High=("High","max"),
            Low=("Low","min"), Close=("Close","last"), Volume=("Volume","sum")
        ).reset_index().rename(columns={"H4_group": "Date"})
        df_h4 = df_h4.dropna().reset_index(drop=True)
        log.info(f"{symbol}: {len(df_h4)} velas H4 ({yf_ticker})")
        return df_h4
    except Exception as e:
        log.error(f"{symbol}: yfinance error — {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for span in [21, 50, 200, 12, 26]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    df["EMA200_slope"] = df["EMA200"] - df["EMA200"].shift(6)
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["TR"]  = np.maximum(df["High"]-df["Low"],
                np.maximum(abs(df["High"]-df["Close"].shift(1)),
                           abs(df["Low"]-df["Close"].shift(1))))
    df["ATR"] = df["TR"].rolling(14).mean()
    df["MACD"]      = df["EMA12"] - df["EMA26"]
    df["MACD_sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    df["MACD_cross_up"]   = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)
    df["MACD_cross_down"] = (df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0)
    df["Regime"] = "LATERAL"
    df.loc[(df["EMA21"]>df["EMA50"])&(df["EMA50"]>df["EMA200"]), "Regime"] = "ALCISTA"
    df.loc[(df["EMA21"]<df["EMA50"])&(df["EMA50"]<df["EMA200"]), "Regime"] = "BAJISTA"
    return df


def _quick_backtest(df, sl_mult, tp_mult, only_long=True):
    dow = df["Date"].dt.dayofweek
    ok  = dow.between(0, 3)
    lm  = (df["Regime"]=="ALCISTA")&(df["EMA200_slope"]>0)&df["MACD_cross_up"]&(df["RSI"].between(RSI_LOW,RSI_HIGH))&ok
    sm  = pd.Series(False,index=df.index) if only_long else \
          (df["Regime"]=="BAJISTA")&(df["EMA200_slope"]<0)&df["MACD_cross_down"]&(df["RSI"].between(RSI_LOW,RSI_HIGH))&ok
    df = df.copy(); df["Signal"] = 0
    df.loc[lm,"Signal"] = 1; df.loc[sm,"Signal"] = -1
    trades, in_trade = [], False
    for i in range(50, len(df)-1):
        sig = df["Signal"].iloc[i]
        if not in_trade and sig != 0:
            direction=sig; atr=df["ATR"].iloc[i]; ep=df["Close"].iloc[i]
            entry=ep; sl=entry-sl_mult*atr if direction==1 else entry+sl_mult*atr
            tp=entry+tp_mult*atr if direction==1 else entry-tp_mult*atr
            entry_idx=i; in_trade=True
        elif in_trade:
            h,l = df["High"].iloc[i], df["Low"].iloc[i]
            if (direction==1 and l<=sl) or (direction==-1 and h>=sl):
                trades.append({"result":"LOSS","pct":-abs(entry-sl)/entry*100}); in_trade=False
            elif (direction==1 and h>=tp) or (direction==-1 and l<=tp):
                trades.append({"result":"WIN","pct":abs(entry-tp)/entry*100}); in_trade=False
            elif (i-entry_idx)>=12:
                pnl=(df["Close"].iloc[i]-entry)*direction/entry*100
                trades.append({"result":"WIN" if pnl>0 else "LOSS","pct":pnl}); in_trade=False
    return trades


def recalibrate(df, lookback_days=90):
    cutoff = df["Date"].max() - timedelta(days=lookback_days)
    recent = df[df["Date"]>=cutoff].copy()
    if len(recent) < 50:
        return {"sl":SL_MULT,"tp":TP_MULT,"recalibrated":False,"pf_recent":1.0}
    best_pf, best_sl, best_tp = 0, SL_MULT, TP_MULT
    for sl in [1.5,2.0,2.5]:
        for tp in [2.5,3.0,3.5,4.0]:
            trades=_quick_backtest(recent,sl,tp)
            if len(trades)<3: continue
            wins=[t for t in trades if t["result"]=="WIN"]
            losses=[t for t in trades if t["result"]=="LOSS"]
            if not losses: continue
            pf=abs(sum(t["pct"] for t in wins))/abs(sum(t["pct"] for t in losses))
            if pf>best_pf: best_pf,best_sl,best_tp=pf,sl,tp
    return {"sl":best_sl,"tp":best_tp,"pf_recent":round(best_pf,2),"lookback_days":lookback_days,"recalibrated":True}


def analyze_signal(df, params, only_long=True):
    if len(df) < 210: return {"signal":"SIN_DATOS"}
    last=df.iloc[-1]; regime=last["Regime"]
    cond_ema   = regime != "LATERAL"
    cond_slope = (last["EMA200_slope"]>0) if regime=="ALCISTA" else (last["EMA200_slope"]<0) if regime=="BAJISTA" else False
    cond_macd  = bool(last["MACD_cross_up"]) if regime=="ALCISTA" else bool(last["MACD_cross_down"]) if regime=="BAJISTA" else False
    cond_rsi   = RSI_LOW <= last["RSI"] <= RSI_HIGH
    cond_dow   = last["Date"].dayofweek in [0,1,2,3]
    conds_met  = sum([cond_ema,cond_slope,cond_macd,cond_rsi])
    long_ok  = regime=="ALCISTA" and cond_slope and cond_macd and cond_rsi and cond_dow
    short_ok = not only_long and regime=="BAJISTA" and cond_slope and cond_macd and cond_rsi and cond_dow
    sl_m=params.get("sl",SL_MULT); tp_m=params.get("tp",TP_MULT)
    if long_ok:    signal,direction="LONG",1
    elif short_ok: signal,direction="SHORT",-1
    else:          signal,direction="ESPERAR",0
    entry=float(last["Close"]); atr=float(last["ATR"])
    return {
        "signal":signal,"direction":direction,"regime":regime,
        "price":round(entry,5),"rsi":round(float(last["RSI"]),1),
        "atr":round(atr,5),"macd_hist":round(float(last["MACD_hist"]),6),
        "conds_met":conds_met,"last_candle":str(last["Date"]),
        "entry":round(entry,5),
        "sl":round(entry-sl_m*atr if direction==1 else entry+sl_m*atr,5),
        "tp":round(entry+tp_m*atr if direction==1 else entry-tp_m*atr,5),
        "sl_pct":round(sl_m*atr/entry*100,3),"tp_pct":round(tp_m*atr/entry*100,3),
        "sl_mult":sl_m,"tp_mult":tp_m,
        "conditions":{"ema_aligned":cond_ema,"ema_slope":cond_slope,
                      "macd_cross":cond_macd,"rsi_ok":cond_rsi,"day_ok":cond_dow},
    }


def compute_confidence(signal, recal, hist_trades):
    score = signal["conds_met"] * 10
    if hist_trades:
        wins = [t for t in hist_trades if t.get("result")=="WIN"]
        score += int(len(wins)/len(hist_trades)*30)
    pf = recal.get("pf_recent",1.0)
    if isinstance(pf,float): score += min(20,int((pf-1.0)*20))
    if signal["regime"] != "LATERAL": score += 10
    return min(100,score)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message); return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"Markdown"},
            timeout=10
        )
        if r.ok: log.info("✅ Telegram enviado"); return True
        log.error(f"Telegram: {r.status_code}"); return False
    except Exception as e:
        log.error(f"Telegram: {e}"); return False


def format_signal_msg(symbol, market, signal, recal, confidence, ai_text, macro_events):
    if signal["signal"] not in ("LONG","SHORT"): return ""
    is_long = signal["signal"]=="LONG"
    color   = "🟢" if is_long else "🔴"
    arrow   = "📈 LONG — COMPRAR" if is_long else "📉 SHORT — VENDER"
    bar     = "█"*(confidence//10) + "░"*(10-confidence//10)
    c       = signal["conditions"]
    conds   = (f"  {'✅' if c['ema_aligned'] else '❌'} EMA alineadas ({signal['regime']})\n"
               f"  {'✅' if c['ema_slope']   else '❌'} EMA200 pendiente\n"
               f"  {'✅' if c['macd_cross']  else '❌'} MACD cruce\n"
               f"  {'✅' if c['rsi_ok']      else '❌'} RSI {signal['rsi']}")
    ai_s = " ".join(ai_text.split()[:50])+"..." if len(ai_text.split())>50 else ai_text
    ai_s = ai_s.replace("**","*")
    macro_str = ""
    if macro_events:
        evs = " | ".join(f"{e.get('time','?')} {e.get('event','?')}" for e in macro_events[:3])
        macro_str = f"\n⚠️ _Eventos macro hoy: {evs}_"
    # Tamaño de posición con riesgo real según capas
    vix_val = STATE.get("vix", 0) or 0
    risk_reason_short = STATE.get("risk_reason", "")[:60]
    pos = get_position_size(10000, signal["entry"], signal["sl"])
    risk_line = (
        f"  💼 Riesgo: *{pos['risk_pct']}% del capital* = `${pos['risk_usd']:,.0f}` por $10k\n"
        f"  📊 VIX: {vix_val:.1f} | {risk_reason_short}"
    )

    return (
        f"{color} *SEÑAL — {market['emoji']} {market['name']} ({symbol})* {color}\n"
        f"{arrow}\n\n"
        f"📊 *Confianza: {confidence}%* `{bar}`\n\n"
        f"💰 *Niveles:*\n"
        f"  Entrada:     `{signal['entry']}`\n"
        f"  Stop Loss:   `{signal['sl']}` (-{signal['sl_pct']}%)\n"
        f"  Take Profit: `{signal['tp']}` (+{signal['tp_pct']}%)\n"
        f"  SL×{recal['sl']} TP×{recal['tp']} PF:{recal.get('pf_recent','?')}\n\n"
        f"📐 *Tamaño de posición:*\n{risk_line}\n\n"
        f"📋 *Condiciones ({signal['conds_met']}/4):*\n{conds}\n"
        f"{macro_str}\n\n"
        f"🤖 _{ai_s}_\n\n"
        f"⏰ _{signal['last_candle']}_ | _Ajusta lotes según tu capital real_"
    )


def ai_analysis_fn(symbol, market_info, signal_data, recal, hist_trades):
    name  = market_info["name"]
    today = datetime.now().strftime("%A %d de %B de %Y")
    wins  = [t for t in hist_trades if t.get("result")=="WIN"]
    losses= [t for t in hist_trades if t.get("result")=="LOSS"]
    wr    = round(len(wins)/len(hist_trades)*100,1) if hist_trades else 0
    pf_v  = round(abs(sum(t["pct"] for t in wins))/abs(sum(t["pct"] for t in losses) or 1),2)
    recent= ", ".join(f"{'WIN' if t['result']=='WIN' else 'LOSS'}({t['pct']:+.2f}%)" for t in hist_trades[-5:]) if hist_trades else "Sin historial"
    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=500,
            tools=[{"type":"web_search_20250305","name":"web_search"}],
            messages=[{"role":"user","content":
                f"Trading pro. Hoy: {today}. {name} ({symbol}) — Señal: {signal_data['signal']}. "
                f"Precio: {signal_data['price']} RSI: {signal_data['rsi']} Régimen: {signal_data['regime']}. "
                f"WR: {wr}% PF: {pf_v}. Últimas: {recent}. "
                f"Busca info y responde en español <200 palabras: "
                f"1.**MACRO** 2.**SEÑAL válida?** 3.**CONFIANZA %** 4.**VEREDICTO**"
            }]
        )
        return " ".join(b.text for b in response.content if hasattr(b,"text")).strip()
    except Exception as e:
        return f"IA no disponible: {e}"



def notify_bridge(symbol: str, market: dict, signal: dict,
                  recal: dict, confidence: int, ai_text: str) -> bool:
    """
    Notifica al MT5 Bridge que hay una señal.
    El bridge enviará a Telegram los botones de confirmación.
    """
    if not BRIDGE_URL or not BRIDGE_SECRET:
        return False
    try:
        payload = {
            "secret":     BRIDGE_SECRET,
            "symbol":     symbol,
            "name":       market["name"],
            "signal":     signal["signal"],
            "entry":      signal["entry"],
            "sl":         signal["sl"],
            "tp":         signal["tp"],
            "sl_pct":     signal["sl_pct"],
            "tp_pct":     signal["tp_pct"],
            "risk_pct":   STATE.get("current_risk_pct", 1.5),
            "confidence": confidence,
            "ai_text":    ai_text[:500],
            "rsi":        signal["rsi"],
            "regime":     signal["regime"],
            "vix":        STATE.get("vix", 0),
        }
        r = requests.post(
            f"{BRIDGE_URL}/nueva_operacion",
            json=payload,
            timeout=10
        )
        if r.ok:
            log.info(f"✅ Bridge notificado: {symbol} {signal['signal']}")
            return True
        log.warning(f"Bridge error: {r.status_code}")
        return False
    except Exception as e:
        log.warning(f"Bridge no disponible: {e}")
        return False

def save_signal_log(symbol, signal, confidence, ai_text, result=None):
    try:
        os.makedirs(os.path.dirname(SIGNALS_LOG), exist_ok=True)
        data = json.load(open(SIGNALS_LOG)) if os.path.exists(SIGNALS_LOG) else []
        entry = {"ts":datetime.now().isoformat(),"symbol":symbol,
                 "signal":signal["signal"],"price":signal["price"],
                 "entry":signal["entry"],"sl":signal["sl"],"tp":signal["tp"],
                 "confidence":confidence,"rsi":signal["rsi"],
                 "regime":signal["regime"],"ai":ai_text[:200],
                 "risk_pct":STATE["current_risk_pct"],
                 "risk_mode":STATE["risk_mode"],
                 "result":result}
        data.append(entry)
        json.dump(data[-500:], open(SIGNALS_LOG,"w"), indent=2)
        STATE["signals_log"] = data[-20:]
        # Actualizar racha si tenemos resultado
        if result:
            update_streak(result)
    except Exception as e:
        log.warning(f"Log: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CICLO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_cycle():
    log.info("="*50)
    now_madrid = datetime.now(timezone.utc) + timedelta(hours=2)
    log.info(f"Ciclo — {now_madrid.strftime('%Y-%m-%d %H:%M')} Madrid")
    log.info("="*50)

    STATE["last_update"] = now_madrid.strftime("%d/%m/%Y %H:%M")
    next_dt = now_madrid + timedelta(seconds=CHECK_INTERVAL)
    STATE["next_update"] = next_dt.strftime("%d/%m/%Y %H:%M")

    # 1. Obtener todas las capas de protección
    events       = fetch_macro_events()
    vix          = fetch_vix()
    fed_bce, fb_reason = check_fed_bce_week()
    risk_pct, risk_reason = get_risk_level(events, vix, fed_bce)

    STATE["macro_events"]  = events
    STATE["macro_blocked"] = (risk_pct == 0.0)
    STATE["current_risk_pct"] = risk_pct
    STATE["risk_reason"]   = risk_reason

    vix_str = f"VIX: {vix:.1f}" if vix else "VIX: N/D"
    log.info(f"Protecciones: {risk_reason} | {vix_str} | Fed/BCE semana: {fed_bce}")

    if risk_pct == 0.0:
        log.warning(f"🚫 OPERACIÓN BLOQUEADA: {risk_reason}")
        send_telegram(
            f"🚫 *Operación bloqueada*\n\n"
            f"Motivo: {risk_reason}\n"
            f"VIX actual: {vix:.1f if vix else 'N/D'}\n"
            f"_El sistema reanudará en el próximo ciclo._"
        )
        STATE["markets"] = {s: {"signal":"BLOQUEADO","regime":"—","rsi":0,"conds_met":0,
                                "name":MARKETS[s]["name"],"emoji":MARKETS[s]["emoji"]} for s in MARKETS}
        return

    # 2. Analizar mercados
    found = []
    markets_state = {}

    for symbol, mkt in MARKETS.items():
        log.info(f"→ {symbol}...")
        df = fetch_candles(symbol, mkt["yf"])
        if df is None or len(df) < 250:
            markets_state[symbol] = {"signal":"ERROR","regime":"—","rsi":0,"conds_met":0,"name":mkt["name"],"emoji":mkt["emoji"]}
            continue

        df     = calculate_indicators(df)
        recal  = recalibrate(df)
        signal = analyze_signal(df, recal, mkt["only_long"])
        cutoff = df["Date"].max() - timedelta(days=180)
        hist   = _quick_backtest(df[df["Date"]>=cutoff].copy(), recal["sl"], recal["tp"], mkt["only_long"])
        conf   = compute_confidence(signal, recal, hist)

        # Guardar estado para el dashboard
        wins_h = [t for t in hist if t.get("result")=="WIN"]
        markets_state[symbol] = {
            **signal,
            "name":       mkt["name"],
            "emoji":      mkt["emoji"],
            "confidence": conf,
            "recal":      recal,
            "wr_recent":  round(len(wins_h)/len(hist)*100,1) if hist else 0,
            "pf_recent":  recal.get("pf_recent",1.0),
        }

        log.info(f"{symbol}: {signal.get('signal')} | RSI={signal.get('rsi')} | "
                 f"Conds={signal.get('conds_met')}/4 | {signal.get('regime')} | Conf={conf}%")

        if signal.get("signal") not in ("ESPERAR","SIN_DATOS"):
            log.info(f"⚡ SEÑAL {symbol} {signal['signal']} — llamando IA...")
            ai = ai_analysis_fn(symbol, mkt, signal, recal, hist)

            # Si hay bridge MT5 activo → botones de confirmación en Telegram
            if BRIDGE_URL and BRIDGE_SECRET:
                bridge_ok = notify_bridge(symbol, mkt, signal, recal, conf, ai)
                if bridge_ok:
                    log.info(f"Bridge notificado — esperando confirmación del usuario")
                    found.append(f"{symbol} {signal['signal']}")
                    save_signal_log(symbol, signal, conf, ai)
                    continue  # No mandar mensaje normal, el bridge lo gestiona

            # Sin bridge → mensaje Telegram estándar
            msg = format_signal_msg(symbol, mkt, signal, recal, conf, ai, events)
            if msg:
                send_telegram(msg)
                save_signal_log(symbol, signal, conf, ai)
                found.append(f"{symbol} {signal['signal']}")

        time.sleep(2)

    STATE["markets"]       = markets_state
    STATE["signals_today"] = found

    # 3. Resumen matinal 8h Madrid
    if not found and now_madrid.hour == 8 and now_madrid.minute < 20 and now_madrid.weekday() < 5:
        summary = "\n".join(f"  {MARKETS[s]['emoji']} {MARKETS[s]['name']}: ESPERAR" for s in MARKETS)
        macro_str = ""
        if events:
            macro_str = "\n\n⚠️ *Eventos macro hoy:*\n" + "\n".join(
                f"  {e.get('time','?')} UTC — {e.get('event','?')} ({e.get('currency','?')})"
                for e in events[:5]
            )
        send_telegram(
            f"☀️ *Buenos días — {now_madrid.strftime('%d/%m/%Y %H:%M')} Madrid*\n\n"
            f"Sin señales activas.\n\n*Estado:*\n{summary}"
            f"{macro_str}\n\n"
            f"_Análisis cada {CHECK_INTERVAL//3600}h. Te aviso si hay señal._ ✅"
        )

    if found:
        log.info(f"Señales: {', '.join(found)}")
    else:
        log.info("Sin señales — todos en ESPERAR")

    STATE["system_ok"] = True


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD WEB (Flask)
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="900">
<title>Trading Signals PRO</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@600;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#07070f;--surf:#0f0f1a;--card:#141422;--border:#1f1f32;--gold:#e8b84b;--green:#2ecc8a;--red:#e74c5e;--blue:#4fa3ff;--dim:#555577;--text:#ddddf5;--textdim:#8888b0}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;min-height:100vh}
body::before{content:"";position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 59px,rgba(232,184,75,0.03) 60px),repeating-linear-gradient(90deg,transparent,transparent 59px,rgba(232,184,75,0.03) 60px);pointer-events:none}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px;position:relative;z-index:1}
nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:36px;padding-bottom:18px;border-bottom:1px solid var(--border)}
.brand{font-family:"Syne",sans-serif;font-size:18px;font-weight:800;letter-spacing:-0.02em}
.brand span{color:var(--gold)}
.status{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--dim)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.strip{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border);border-radius:10px;overflow:hidden;margin-bottom:28px}
.strip-item{background:var(--surf);padding:14px 16px}
.strip-label{font-size:9px;color:var(--dim);letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px}
.strip-val{font-size:20px;font-weight:600;letter-spacing:-.02em}
.macro-bar{padding:12px 18px;border-radius:8px;margin-bottom:24px;font-size:12px;display:flex;align-items:center;gap:10px}
.macro-ok{background:rgba(46,204,138,.07);border:1px solid rgba(46,204,138,.2);color:var(--green)}
.macro-block{background:rgba(231,76,94,.1);border:1px solid rgba(231,76,94,.3);color:var(--red)}
.macro-events{background:rgba(79,163,255,.07);border:1px solid rgba(79,163,255,.2);color:var(--blue)}
.sec-label{font-size:9px;color:var(--dim);letter-spacing:.2em;text-transform:uppercase;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px;margin-bottom:32px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:transform .2s}
.card:hover{transform:translateY(-2px)}
.card.long{border-color:rgba(232,184,75,.4)}
.card.long::before{content:"";display:block;height:2px;background:linear-gradient(90deg,var(--gold),#f0c060)}
.card.short{border-color:rgba(231,76,94,.4)}
.card.short::before{content:"";display:block;height:2px;background:linear-gradient(90deg,var(--red),#ff7090)}
.card-head{padding:16px 16px 0;display:flex;justify-content:space-between;align-items:flex-start}
.mkt-id{display:flex;align-items:center;gap:9px}
.mkt-emoji{font-size:20px}
.mkt-name{font-size:16px;font-weight:600;letter-spacing:-.01em}
.mkt-sym{font-size:9px;color:var(--dim);letter-spacing:.1em;margin-top:1px}
.chip{padding:4px 10px;border-radius:100px;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.chip-long{background:rgba(232,184,75,.15);color:var(--gold);border:1px solid rgba(232,184,75,.3)}
.chip-short{background:rgba(231,76,94,.15);color:var(--red);border:1px solid rgba(231,76,94,.3)}
.chip-wait{background:rgba(255,255,255,.04);color:var(--dim);border:1px solid var(--border)}
.chip-err{background:rgba(79,163,255,.1);color:var(--blue);border:1px solid rgba(79,163,255,.2)}
.price-row{padding:10px 16px 0;font-size:22px;font-weight:600;letter-spacing:-.02em}
.conds{padding:10px 16px;display:grid;grid-template-columns:1fr 1fr;gap:4px}
.cond{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--textdim)}
.cond.ok{color:var(--text)}.cdot{width:5px;height:5px;border-radius:50%}
.cond.ok .cdot{background:var(--green)}.cond .cdot{background:var(--dim)}
.inds{padding:0 16px 12px;display:flex;gap:14px;flex-wrap:wrap}
.ind-l{font-size:8px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase}
.ind-v{font-size:12px;font-weight:500}
.reg{display:inline-block;padding:1px 6px;border-radius:3px;font-size:8px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.reg-up{background:rgba(232,184,75,.12);color:var(--gold)}
.reg-dn{background:rgba(231,76,94,.12);color:var(--red)}
.reg-lat{background:rgba(255,255,255,.05);color:var(--dim)}
.conf-bar{height:3px;background:var(--border);border-radius:2px;margin:0 16px 6px;overflow:hidden}
.conf-fill{height:100%;border-radius:2px;transition:width .5s}
.levels{margin:0 16px 16px;background:rgba(0,0,0,.2);border-radius:6px;padding:10px 12px;display:none}
.card.long .levels,.card.short .levels{display:block}
.lev-title{font-size:8px;color:var(--dim);letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px}
.lev-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.lev-l{font-size:8px;color:var(--dim);letter-spacing:.08em;text-transform:uppercase}
.lev-v{font-size:11px;font-weight:600;margin-top:1px}
.c-e{color:var(--text)}.c-sl{color:var(--red)}.c-tp{color:var(--green)}
.lev-p{font-size:8px;color:var(--dim)}
.log-table{width:100%;border-collapse:collapse;font-size:11px}
.log-table th{font-size:8px;color:var(--dim);letter-spacing:.12em;text-transform:uppercase;padding:7px 10px;text-align:left;border-bottom:1px solid var(--border)}
.log-table td{padding:8px 10px;border-bottom:1px solid rgba(31,31,50,.5);color:var(--textdim)}
.log-table tr:hover td{background:rgba(255,255,255,.02)}
.tw{color:var(--green);font-weight:600}.tl{color:var(--red);font-weight:600}
.tlong{color:var(--gold)}.tshort{color:var(--red)}
.no-log{color:var(--dim);font-size:12px;padding:20px;text-align:center}
footer{border-top:1px solid var(--border);padding-top:20px;display:flex;justify-content:space-between;font-size:10px;color:var(--dim);flex-wrap:wrap;gap:8px}
@media(max-width:600px){.strip{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<nav>
  <div class="brand">Trading Signals <span>PRO</span></div>
  <div class="status"><div class="dot"></div><span id="status-txt">cargando...</span></div>
</nav>

<div class="strip" id="strip">
  <div class="strip-item"><div class="strip-label">Actualizado</div><div class="strip-val" id="s-upd" style="font-size:13px">—</div></div>
  <div class="strip-item"><div class="strip-label">Próximo ciclo</div><div class="strip-val" id="s-next" style="font-size:13px">—</div></div>
  <div class="strip-item"><div class="strip-label">Señales activas</div><div class="strip-val" id="s-sigs" style="color:var(--gold)">—</div></div>
  <div class="strip-item"><div class="strip-label">Win Rate global</div><div class="strip-val" style="color:var(--green)">51%</div></div>
  <div class="strip-item"><div class="strip-label">Profit Factor</div><div class="strip-val" style="color:var(--gold)">1.47</div></div>
</div>

<div id="macro-area"></div>

<div class="sec-label">▸ Estado de mercados</div>
<div class="grid" id="grid"></div>

<div class="sec-label">▸ Historial de señales</div>
<div style="background:var(--surf);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:32px">
  <div id="log-area"><div class="no-log">Cargando historial...</div></div>
</div>

<footer>
  <span>Trading Signals PRO · Estrategia validada 20 años · PF 1.47 · WR 51%</span>
  <span>Actualiza cada 15 min · <a href="/api/state" style="color:var(--dim)">API JSON</a></span>
</footer>
</div>

<script>
function regClass(r){return r==="ALCISTA"?"reg-up":r==="BAJISTA"?"reg-dn":"reg-lat"}
function fmt(p,s){
  if(s==="EURUSD"||s==="USDCHF") return parseFloat(p).toFixed(5);
  if(p>1000) return parseFloat(p).toLocaleString("es-ES",{minimumFractionDigits:2,maximumFractionDigits:2});
  return parseFloat(p).toFixed(2);
}

async function load(){
  try{
    const r = await fetch("/api/state");
    const d = await r.json();

    // Strip
    document.getElementById("s-upd").textContent  = d.last_update || "—";
    document.getElementById("s-next").textContent = d.next_update || "—";
    const sigs = Object.values(d.markets).filter(m=>m.signal==="LONG"||m.signal==="SHORT");
    document.getElementById("s-sigs").textContent = sigs.length || "0";
    document.getElementById("status-txt").textContent = d.system_ok ? "Sistema activo" : "Error";

    // Macro
    const ma = document.getElementById("macro-area");
    if(d.macro_blocked){
      ma.innerHTML = `<div class="macro-bar macro-block">🚫 <strong>Análisis bloqueado</strong> — Evento macro de alto impacto en curso (±2h)</div>`;
    } else if(d.macro_events && d.macro_events.length>0){
      const evs = d.macro_events.map(e=>`${e.time||"?"} UTC: ${e.event||"?"}`).join(" · ");
      ma.innerHTML = `<div class="macro-bar macro-events">📅 <strong>Eventos macro hoy:</strong> ${evs}</div>`;
    } else {
      ma.innerHTML = `<div class="macro-bar macro-ok">✅ Sin eventos macro de alto impacto hoy — estrategia activa</div>`;
    }

    // Market cards
    const grid = document.getElementById("grid");
    grid.innerHTML = Object.entries(d.markets).map(([sym,m])=>{
      const isL=m.signal==="LONG", isS=m.signal==="SHORT";
      const cls = isL?"long":isS?"short":"";
      const chipCls = isL?"chip-long":isS?"chip-short":m.signal==="ERROR"?"chip-err":"chip-wait";
      const chipTxt = isL?"▲ LONG":isS?"▼ SHORT":m.signal==="BLOQUEADO"?"⏸ BLOQUEADO":m.signal==="ERROR"?"⚠ ERROR":"◦ ESPERAR";
      const rsiColor = (m.rsi||0)>70?"var(--red)":(m.rsi||0)<30?"var(--green)":"var(--gold)";
      const confPct  = m.confidence||0;
      const confColor= confPct>=70?"var(--green)":confPct>=50?"var(--gold)":"var(--dim)";
      const conds    = m.conditions||{};
      const lev = (isL||isS) ? `
        <div class="levels">
          <div class="lev-title">▸ Niveles</div>
          <div class="lev-grid">
            <div><div class="lev-l">Entrada</div><div class="lev-v c-e">${fmt(m.entry,sym)}</div></div>
            <div><div class="lev-l">Stop Loss</div><div class="lev-v c-sl">${fmt(m.sl,sym)}</div><div class="lev-p">-${m.sl_pct}%</div></div>
            <div><div class="lev-l">Take Profit</div><div class="lev-v c-tp">${fmt(m.tp,sym)}</div><div class="lev-p">+${m.tp_pct}%</div></div>
          </div>
        </div>` : "";
      return `<div class="card ${cls}">
        <div class="card-head">
          <div class="mkt-id"><div class="mkt-emoji">${m.emoji||""}</div>
            <div><div class="mkt-name">${m.name||sym}</div><div class="mkt-sym">${sym} · H4</div></div>
          </div>
          <div class="chip ${chipCls}">${chipTxt}</div>
        </div>
        <div class="price-row">${fmt(m.price||0,sym)}</div>
        <div class="conds">
          <div class="cond ${conds.ema_aligned?"ok":""}"><div class="cdot"></div>EMA alineadas</div>
          <div class="cond ${conds.ema_slope?"ok":""}"><div class="cdot"></div>EMA200 pendiente</div>
          <div class="cond ${conds.macd_cross?"ok":""}"><div class="cdot"></div>MACD cruce</div>
          <div class="cond ${conds.rsi_ok?"ok":""}"><div class="cdot"></div>RSI ${m.rsi||"—"}</div>
        </div>
        <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%;background:${confColor}"></div></div>
        <div class="inds">
          <div><div class="ind-l">Régimen</div><div class="ind-v"><span class="reg ${regClass(m.regime||"LATERAL")}">${m.regime||"—"}</span></div></div>
          <div><div class="ind-l">ATR</div><div class="ind-v">${fmt(m.atr||0,sym)}</div></div>
          <div><div class="ind-l">Confianza</div><div class="ind-v" style="color:${confColor}">${confPct}%</div></div>
          <div><div class="ind-l">WR recent</div><div class="ind-v" style="color:var(--green)">${m.wr_recent||0}%</div></div>
        </div>
        ${lev}
      </div>`;
    }).join("");

    // Signal log
    const logs = d.signals_log||[];
    const logArea = document.getElementById("log-area");
    if(!logs.length){
      logArea.innerHTML = `<div class="no-log">Sin señales registradas aún. El sistema enviará alertas cuando detecte señales.</div>`;
    } else {
      logArea.innerHTML = `<table class="log-table">
        <thead><tr><th>Fecha</th><th>Mercado</th><th>Señal</th><th>Precio</th><th>SL</th><th>TP</th><th>Conf</th><th>Resultado</th></tr></thead>
        <tbody>${[...logs].reverse().map(l=>`<tr>
          <td>${(l.ts||"").substring(0,16).replace("T"," ")}</td>
          <td>${l.symbol||"—"}</td>
          <td class="${l.signal==="LONG"?"tlong":"tshort"}">${l.signal==="LONG"?"▲ LONG":"▼ SHORT"}</td>
          <td>${fmt(l.price||0,l.symbol||"")}</td>
          <td class="tl">${fmt(l.sl||0,l.symbol||"")}</td>
          <td class="tw">${fmt(l.tp||0,l.symbol||"")}</td>
          <td>${l.confidence||0}%</td>
          <td>${l.result?`<span class="${l.result==="WIN"?"tw":"tl"}">${l.result}</span>`:"<span style='color:var(--dim)'>—</span>"}</td>
        </tr>`).join("")}</tbody>
      </table>`;
    }

  } catch(e){
    document.getElementById("status-txt").textContent = "Error cargando datos";
  }
}

load();
setInterval(load, 60000); // recarga cada 1 minuto
</script>
</body>
</html>'''


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/state")
def api_state():
    """API JSON con el estado actual — la web lo consulta cada minuto."""
    try:
        if os.path.exists(SIGNALS_LOG):
            STATE["signals_log"] = json.load(open(SIGNALS_LOG))[-20:]
    except Exception:
        pass
    # Convertir numpy bools y otros tipos no serializables
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        if hasattr(obj, 'item'):   # numpy scalar
            return obj.item()
        if isinstance(obj, bool):
            return bool(obj)
        return obj
    return jsonify(clean(STATE))

@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now().isoformat()})


# ══════════════════════════════════════════════════════════════════════════════
# ARRANQUE
# ══════════════════════════════════════════════════════════════════════════════

def analysis_loop():
    """Hilo de análisis — corre en background."""
    log.info("🔍 Hilo de análisis arrancado")
    while True:
        try:
            run_cycle()
        except Exception as e:
            log.error(f"Error en ciclo: {e}", exc_info=True)
            STATE["system_ok"] = False
            send_telegram(f"⚠️ Error\n`{str(e)[:200]}`\nReintentando en 30 min...")
            time.sleep(1800)
            continue
        time.sleep(CHECK_INTERVAL)


def main():
    log.info("🚀 Trading Signals PRO — v3.0")
    log.info(f"Mercados: {list(MARKETS.keys())}")
    log.info(f"Puerto web: {PORT} | Intervalo: {CHECK_INTERVAL//3600}h")
    log.info(f"Telegram: {'✅' if TELEGRAM_TOKEN else '⚠️'} | IA: {'✅' if ANTHROPIC_KEY else '⚠️'}")

    # Test message
    if os.environ.get("SEND_TEST_MESSAGE","").lower() == "true":
        now_m = datetime.now(timezone.utc) + timedelta(hours=2)
        send_telegram(
            f"✅ *Trading Signals PRO v3 — OK*\n\n"
            f"Dashboard: tu URL de Railway\n"
            f"Mercados: {', '.join(MARKETS.keys())}\n"
            f"Intervalo: {CHECK_INTERVAL//3600}h | SL×{SL_MULT} TP×{TP_MULT}\n"
            f"Hora Madrid: {now_m.strftime('%d/%m/%Y %H:%M')}\n\n"
            f"⚠️ Borra SEND_TEST_MESSAGE en Railway."
        )

    # Hilo de análisis en background
    t = threading.Thread(target=analysis_loop, daemon=True)
    t.start()

    # Servidor web en primer plano (Railway necesita esto)
    log.info(f"🌐 Dashboard disponible en tu URL de Railway (puerto {PORT})")
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
