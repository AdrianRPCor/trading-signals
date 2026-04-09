"""
Trading Signals PRO — Motor principal
Corre cada 4 horas, analiza 6 mercados, manda alertas por Telegram
"""

import os
import json
import time
import asyncio
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from anthropic import Anthropic

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────────
ALPHAVANTAGE_KEY = os.environ.get("ALPHAVANTAGE_KEY", "")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_KEY", "")

# Intervalo de chequeo en segundos (4 horas = 14400)
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "14400"))

# Parámetros de la estrategia (ajustables via env vars)
SL_MULT  = float(os.environ.get("SL_MULT",  "2.0"))
TP_MULT  = float(os.environ.get("TP_MULT",  "3.5"))
RSI_LOW  = float(os.environ.get("RSI_LOW",  "38"))
RSI_HIGH = float(os.environ.get("RSI_HIGH", "62"))

# Mercados a analizar
MARKETS = {
    "XAUUSD": {"name": "Oro",        "emoji": "🥇", "only_long": True,  "type": "physical_currency"},
    "EURUSD": {"name": "EUR/USD",    "emoji": "💶", "only_long": False, "type": "physical_currency"},
    "USDCHF": {"name": "USD/CHF",    "emoji": "🇨🇭", "only_long": False, "type": "physical_currency"},
}

# ── CLIENTES ─────────────────────────────────────────────────────────────────
claude  = Anthropic(api_key=ANTHROPIC_KEY)

# ── ALPHA VANTAGE — DATOS OHLC ───────────────────────────────────────────────
def fetch_candles_alphavantage(symbol: str, outputsize: str = "full") -> pd.DataFrame | None:
    """
    Descarga velas H4 desde Alpha Vantage.
    Combina velas H1 en H4 para mayor precisión.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function":   "FX_INTRADAY" if symbol != "XAUUSD" else "TIME_SERIES_INTRADAY",
        "interval":   "60min",
        "outputsize": outputsize,
        "apikey":     ALPHAVANTAGE_KEY,
        "datatype":   "json",
    }

    if symbol == "XAUUSD":
        # El oro usa un endpoint diferente en Alpha Vantage
        params["function"] = "TIME_SERIES_INTRADAY"
        params["symbol"]   = "XAUUSD"
        params["interval"] = "60min"
    elif "/" in symbol or len(symbol) == 6:
        params["function"]      = "FX_INTRADAY"
        params["from_symbol"]   = symbol[:3]
        params["to_symbol"]     = symbol[3:]
        params.pop("symbol", None)

    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        # Detectar la clave de datos correcta
        ts_key = None
        for k in data.keys():
            if "Time Series" in k:
                ts_key = k
                break

        if not ts_key:
            log.warning(f"{symbol}: No se encontraron datos en Alpha Vantage. Respuesta: {list(data.keys())}")
            return None

        ts = data[ts_key]
        rows = []
        for dt_str, ohlcv in ts.items():
            rows.append({
                "Date":   pd.to_datetime(dt_str),
                "Open":   float(ohlcv.get("1. open",  ohlcv.get("1a. open (USD)", 0))),
                "High":   float(ohlcv.get("2. high",  ohlcv.get("2a. high (USD)", 0))),
                "Low":    float(ohlcv.get("3. low",   ohlcv.get("3a. low (USD)", 0))),
                "Close":  float(ohlcv.get("4. close", ohlcv.get("4a. close (USD)", 0))),
                "Volume": float(ohlcv.get("5. volume", 0)),
            })

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

        # Convertir H1 → H4 (agrupa cada 4 velas)
        df["H4_group"] = df["Date"].dt.floor("4h")
        df_h4 = df.groupby("H4_group").agg(
            Open   = ("Open",   "first"),
            High   = ("High",   "max"),
            Low    = ("Low",    "min"),
            Close  = ("Close",  "last"),
            Volume = ("Volume", "sum"),
        ).reset_index().rename(columns={"H4_group": "Date"})

        log.info(f"{symbol}: {len(df_h4)} velas H4 descargadas")
        return df_h4

    except Exception as e:
        log.error(f"{symbol}: Error descargando datos — {e}")
        return None


# ── INDICADORES ───────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores de la estrategia."""
    df = df.copy()

    # EMAs
    for span in [21, 50, 200, 12, 26]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    df["EMA200_slope"] = df["EMA200"] - df["EMA200"].shift(6)

    # RSI
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # ATR
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"]  - df["Close"].shift(1))
        )
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    # MACD
    df["MACD"]      = df["EMA12"] - df["EMA26"]
    df["MACD_sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

    # Cruces MACD
    df["MACD_cross_up"]   = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)
    df["MACD_cross_down"] = (df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0)

    # Régimen
    df["Regime"] = "LATERAL"
    df.loc[(df["EMA21"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"]), "Regime"] = "ALCISTA"
    df.loc[(df["EMA21"] < df["EMA50"]) & (df["EMA50"] < df["EMA200"]), "Regime"] = "BAJISTA"

    return df


# ── RECALIBRACIÓN DINÁMICA (walk-forward) ─────────────────────────────────────
def recalibrate_parameters(df: pd.DataFrame, lookback_days: int = 90) -> dict:
    """
    Recalibra SL/TP multiplicadores con los últimos N días.
    Busca la combinación que maximiza el Profit Factor en el período reciente.
    """
    cutoff = df["Date"].max() - timedelta(days=lookback_days)
    recent = df[df["Date"] >= cutoff].copy()

    if len(recent) < 50:
        return {"sl": SL_MULT, "tp": TP_MULT, "recalibrated": False}

    best_pf = 0
    best_sl = SL_MULT
    best_tp = TP_MULT

    for sl in [1.5, 2.0, 2.5]:
        for tp in [2.5, 3.0, 3.5, 4.0]:
            trades = _quick_backtest(recent, sl, tp)
            if not trades:
                continue
            wins   = [t for t in trades if t["result"] == "WIN"]
            losses = [t for t in trades if t["result"] == "LOSS"]
            if not losses:
                continue
            pf = abs(sum(t["pct"] for t in wins)) / abs(sum(t["pct"] for t in losses))
            if pf > best_pf and len(trades) >= 3:
                best_pf = pf
                best_sl = sl
                best_tp = tp

    return {
        "sl":           best_sl,
        "tp":           best_tp,
        "pf_recent":    round(best_pf, 2),
        "lookback_days": lookback_days,
        "recalibrated": True
    }


def _quick_backtest(df: pd.DataFrame, sl_mult: float, tp_mult: float,
                    only_long: bool = True) -> list:
    """Backtest rápido para recalibración."""
    dow = df["Date"].dt.dayofweek
    ok  = dow.between(0, 3)

    lm = (df["Regime"] == "ALCISTA") & (df["EMA200_slope"] > 0) & \
         df["MACD_cross_up"] & (df["RSI"].between(RSI_LOW, RSI_HIGH)) & ok
    sm = pd.Series(False, index=df.index) if only_long else \
         (df["Regime"] == "BAJISTA") & (df["EMA200_slope"] < 0) & \
         df["MACD_cross_down"] & (df["RSI"].between(RSI_LOW, RSI_HIGH)) & ok

    df = df.copy()
    df["Signal"] = 0
    df.loc[lm, "Signal"] = 1
    df.loc[sm, "Signal"] = -1

    trades, in_trade = [], False
    for i in range(50, len(df) - 1):
        sig = df["Signal"].iloc[i]
        if not in_trade and sig != 0:
            direction  = sig
            atr        = df["ATR"].iloc[i]
            ep         = df["Close"].iloc[i]
            entry      = ep + ep * 0.0002 if direction == 1 else ep - ep * 0.0002
            sl         = entry - sl_mult * atr if direction == 1 else entry + sl_mult * atr
            tp         = entry + tp_mult * atr if direction == 1 else entry - tp_mult * atr
            entry_idx  = i
            in_trade   = True
        elif in_trade:
            h, l = df["High"].iloc[i], df["Low"].iloc[i]
            hit_sl = (direction == 1 and l <= sl) or (direction == -1 and h >= sl)
            hit_tp = (direction == 1 and h >= tp) or (direction == -1 and l <= tp)
            if hit_sl:
                trades.append({"result": "LOSS", "pct": -abs(entry - sl) / entry * 100})
                in_trade = False
            elif hit_tp:
                trades.append({"result": "WIN",  "pct":  abs(entry - tp) / entry * 100})
                in_trade = False
            elif (i - entry_idx) >= 12:
                pnl = (df["Close"].iloc[i] - entry) * direction / entry * 100
                trades.append({"result": "WIN" if pnl > 0 else "LOSS", "pct": pnl})
                in_trade = False

    return trades


# ── ANÁLISIS DE SEÑAL ─────────────────────────────────────────────────────────
def analyze_signal(df: pd.DataFrame, params: dict, only_long: bool = True) -> dict:
    """Analiza la última vela y determina si hay señal."""
    if len(df) < 210:
        return {"signal": "SIN_DATOS", "reason": "Pocos datos"}

    last = df.iloc[-1]
    regime = last["Regime"]

    # Condiciones
    cond_ema    = regime != "LATERAL"
    cond_slope  = (last["EMA200_slope"] > 0) if regime == "ALCISTA" else \
                  (last["EMA200_slope"] < 0) if regime == "BAJISTA" else False
    cond_macd   = bool(last["MACD_cross_up"]) if regime == "ALCISTA" else \
                  bool(last["MACD_cross_down"]) if regime == "BAJISTA" else False
    cond_rsi    = RSI_LOW <= last["RSI"] <= RSI_HIGH
    cond_dow    = last["Date"].dayofweek in [0, 1, 2, 3]  # lun-jue

    conds_met   = sum([cond_ema, cond_slope, cond_macd, cond_rsi])

    # Determinar señal
    long_ok  = regime == "ALCISTA" and cond_slope and cond_macd and cond_rsi and cond_dow
    short_ok = not only_long and regime == "BAJISTA" and cond_slope and cond_macd and cond_rsi and cond_dow

    sl_mult = params.get("sl", SL_MULT)
    tp_mult = params.get("tp", TP_MULT)

    if long_ok:
        signal    = "LONG"
        direction = 1
    elif short_ok:
        signal    = "SHORT"
        direction = -1
    else:
        signal    = "ESPERAR"
        direction = 0

    entry = float(last["Close"])
    atr   = float(last["ATR"])

    return {
        "signal":     signal,
        "direction":  direction,
        "regime":     regime,
        "price":      round(entry, 5),
        "rsi":        round(float(last["RSI"]), 1),
        "atr":        round(atr, 5),
        "macd_hist":  round(float(last["MACD_hist"]), 6),
        "ema21":      round(float(last["EMA21"]), 5),
        "ema50":      round(float(last["EMA50"]), 5),
        "ema200":     round(float(last["EMA200"]), 5),
        "slope_ok":   cond_slope,
        "conds_met":  conds_met,
        "entry":      round(entry, 5),
        "sl":         round(entry - sl_mult * atr if direction == 1 else entry + sl_mult * atr, 5),
        "tp":         round(entry + tp_mult * atr if direction == 1 else entry - tp_mult * atr, 5),
        "sl_pct":     round(sl_mult * atr / entry * 100, 3),
        "tp_pct":     round(tp_mult * atr / entry * 100, 3),
        "sl_mult":    sl_mult,
        "tp_mult":    tp_mult,
        "last_candle": str(last["Date"]),
        "conditions": {
            "ema_aligned": cond_ema,
            "ema_slope":   cond_slope,
            "macd_cross":  cond_macd,
            "rsi_ok":      cond_rsi,
            "day_ok":      cond_dow,
        }
    }


# ── ANÁLISIS IA CON CLAUDE ────────────────────────────────────────────────────
def ai_analysis(symbol: str, market_info: dict, signal_data: dict,
                recal: dict, hist_trades: list) -> str:
    """
    Llama a Claude con búsqueda web para análisis fundamental + contexto.
    """
    name    = market_info["name"]
    today   = datetime.now().strftime("%A %d de %B de %Y")
    wins    = [t for t in hist_trades if t.get("result") == "WIN"]
    losses  = [t for t in hist_trades if t.get("result") == "LOSS"]
    wr      = round(len(wins) / len(hist_trades) * 100, 1) if hist_trades else 0
    pf      = round(abs(sum(t["pct"] for t in wins)) / abs(sum(t["pct"] for t in losses) or 1), 2)

    recent_str = ", ".join(
        f"{'WIN' if t['result']=='WIN' else 'LOSS'}({t['pct']:+.2f}%)"
        for t in hist_trades[-5:]
    ) if hist_trades else "Sin historial"

    prompt = f"""Eres un analista de trading profesional. Hoy es {today}.

MERCADO: {name} ({symbol})
SEÑAL TÉCNICA: {signal_data['signal']}
PRECIO ACTUAL: {signal_data['price']}
RSI: {signal_data['rsi']} | ATR: {signal_data['atr']}
RÉGIMEN: {signal_data['regime']}
MACD histograma: {signal_data['macd_hist']}
Condiciones cumplidas: {signal_data['conds_met']}/4

PARÁMETROS RECALIBRADOS (últimos {recal.get('lookback_days',90)} días):
SL: {recal.get('sl', SL_MULT)}×ATR | TP: {recal.get('tp', TP_MULT)}×ATR | PF reciente: {recal.get('pf_recent','N/A')}

HISTORIAL RECIENTE ({len(hist_trades)} ops): WR={wr}% | PF={pf}
Últimas 5: {recent_str}

Por favor busca información actual y proporciona:

1. **CONTEXTO MACRO HOY**: ¿Qué factores fundamentales relevantes hay para {name}? (inflación, Fed/BCE, geopolítica, datos económicos esta semana)

2. **EVALUACIÓN DE LA SEÑAL**: ¿La señal técnica {signal_data['signal']} tiene sentido dado el contexto fundamental actual?

3. **NIVEL DE CONFIANZA** (0-100%): Basándote en técnico + fundamental + historial reciente, ¿qué confianza le darías a esta operación?

4. **RIESGOS ESPECÍFICOS HOY**: ¿Hay algún evento inminente que podría invalidar la señal?

5. **RECOMENDACIÓN FINAL**: ¿Operar o esperar? Una frase concisa.

Responde en español, máximo 300 palabras, usa **negrita** para lo importante."""

    try:
        response = claude.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 800,
            tools      = [{"type": "web_search_20250305", "name": "web_search"}],
            messages   = [{"role": "user", "content": prompt}]
        )
        text = " ".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )
        return text.strip()
    except Exception as e:
        log.error(f"Error en análisis IA: {e}")
        return f"Análisis IA no disponible: {e}"


# ── CONFIANZA COMPUESTA ───────────────────────────────────────────────────────
def compute_confidence(signal: dict, recal: dict, hist_trades: list) -> int:
    """
    Calcula un score de confianza 0-100 combinando:
    - Condiciones técnicas cumplidas
    - Win rate histórico reciente
    - Profit Factor recalibrado
    """
    score = 0

    # Condiciones técnicas (40 pts)
    score += signal["conds_met"] * 10

    # Win rate histórico (30 pts)
    if hist_trades:
        wins = [t for t in hist_trades if t.get("result") == "WIN"]
        wr   = len(wins) / len(hist_trades)
        score += int(wr * 30)

    # Profit Factor reciente (20 pts)
    pf = recal.get("pf_recent", 1.0)
    if isinstance(pf, float):
        score += min(20, int((pf - 1.0) * 20))

    # Régimen no lateral (10 pts)
    if signal["regime"] != "LATERAL":
        score += 10

    return min(100, score)


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Manda mensaje a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram no configurado — mostrando en consola:")
        print("\n" + "="*60)
        print(message)
        print("="*60 + "\n")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.ok:
            log.info("✅ Telegram enviado")
            return True
        else:
            log.error(f"Telegram error: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log.error(f"Telegram exception: {e}")
        return False


def format_signal_message(symbol: str, market: dict, signal: dict,
                           recal: dict, confidence: int, ai_text: str) -> str:
    """Formatea el mensaje de Telegram."""
    name  = market["name"]
    emoji = market["emoji"]
    sig   = signal["signal"]

    if sig == "LONG":
        arrow = "📈 LONG — COMPRAR"
        color = "🟢"
    elif sig == "SHORT":
        arrow = "📉 SHORT — VENDER"
        color = "🔴"
    else:
        return ""  # No mandar mensaje si es ESPERAR

    conf_bar = "█" * (confidence // 10) + "░" * (10 - confidence // 10)

    # Condiciones
    conds = signal["conditions"]
    c_str = "\n".join([
        f"  {'✅' if conds['ema_aligned'] else '❌'} EMA alineadas ({signal['regime']})",
        f"  {'✅' if conds['ema_slope']   else '❌'} EMA200 pendiente correcta",
        f"  {'✅' if conds['macd_cross']  else '❌'} MACD cruce confirmado",
        f"  {'✅' if conds['rsi_ok']      else '❌'} RSI zona neutra ({signal['rsi']})",
    ])

    # Parámetros recalibrados
    recal_str = f"SL×{recal['sl']} TP×{recal['tp']}"
    if recal.get("recalibrated"):
        recal_str += f" *(auto-optimizado, PF reciente: {recal.get('pf_recent','?')})*"

    # Resumen IA (primeras 2 líneas)
    ai_short = " ".join(ai_text.split()[:60]) + "..." if len(ai_text.split()) > 60 else ai_text
    # Limpiar markdown de Claude para Telegram
    ai_short = ai_short.replace("**", "*")

    msg = f"""{color} *SEÑAL DETECTADA* {color}
{emoji} *{name} ({symbol})* — {arrow}

📊 *Confianza: {confidence}%*
`{conf_bar}`

💰 *Niveles de operación:*
  • Entrada:    `{signal['entry']}`
  • Stop Loss:  `{signal['sl']}` (-{signal['sl_pct']}%)
  • Take Profit:`{signal['tp']}` (+{signal['tp_pct']}%)
  • Parámetros: {recal_str}

📋 *Condiciones ({signal['conds_met']}/4):*
{c_str}

📈 *Indicadores:*
  RSI: {signal['rsi']} | ATR: {signal['atr']} | MACD: {signal['macd_hist']:+.5f}

🤖 *Análisis IA:*
_{ai_short}_

⏰ _Vela H4: {signal['last_candle']}_
_Gestión: máximo 2% del capital en riesgo_"""

    return msg


# ── LOG DE SEÑALES (archivo JSON local / Railway volume) ─────────────────────
SIGNALS_LOG = os.environ.get("SIGNALS_LOG_PATH", "/data/signals_log.json")

def save_signal(symbol: str, signal: dict, confidence: int, ai_text: str):
    """Guarda la señal en el log para seguimiento."""
    try:
        os.makedirs(os.path.dirname(SIGNALS_LOG), exist_ok=True)
        log_data = []
        if os.path.exists(SIGNALS_LOG):
            with open(SIGNALS_LOG) as f:
                log_data = json.load(f)

        log_data.append({
            "ts":         datetime.now().isoformat(),
            "symbol":     symbol,
            "signal":     signal["signal"],
            "price":      signal["price"],
            "entry":      signal["entry"],
            "sl":         signal["sl"],
            "tp":         signal["tp"],
            "confidence": confidence,
            "rsi":        signal["rsi"],
            "regime":     signal["regime"],
            "ai_summary": ai_text[:200],
            "result":     None,  # Se actualiza después
        })

        with open(SIGNALS_LOG, "w") as f:
            json.dump(log_data[-500:], f, indent=2)  # Mantener últimas 500
        log.info(f"Señal guardada en log: {symbol} {signal['signal']}")
    except Exception as e:
        log.warning(f"No se pudo guardar el log: {e}")


# ── LOOP PRINCIPAL ────────────────────────────────────────────────────────────
def run_analysis_cycle():
    """Ejecuta un ciclo completo de análisis para todos los mercados."""
    log.info(f"{'='*50}")
    log.info(f"Iniciando ciclo de análisis — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'='*50}")

    signals_found = []

    for symbol, market_info in MARKETS.items():
        log.info(f"Analizando {symbol} ({market_info['name']})...")

        # 1. Descargar datos
        df = fetch_candles_alphavantage(symbol)
        if df is None or len(df) < 250:
            log.warning(f"{symbol}: Datos insuficientes, saltando")
            continue

        # 2. Calcular indicadores
        df = calculate_indicators(df)

        # 3. Recalibrar parámetros con últimos 90 días
        recal = recalibrate_parameters(df, lookback_days=90)
        log.info(f"{symbol}: Parámetros recalibrados — SL×{recal['sl']} TP×{recal['tp']} PF:{recal.get('pf_recent','N/A')}")

        # 4. Analizar señal actual
        signal = analyze_signal(df, recal, market_info["only_long"])
        log.info(f"{symbol}: Señal = {signal['signal']} | RSI={signal['rsi']} | Conds={signal['conds_met']}/4")

        if signal["signal"] == "SIN_DATOS":
            continue

        # 5. Backtest reciente para historial
        recent_cutoff = df["Date"].max() - timedelta(days=180)
        recent_df     = df[df["Date"] >= recent_cutoff].copy()
        hist_trades   = _quick_backtest(
            recent_df, recal["sl"], recal["tp"], market_info["only_long"]
        )

        # 6. Calcular confianza
        confidence = compute_confidence(signal, recal, hist_trades)
        log.info(f"{symbol}: Confianza = {confidence}%")

        # 7. Solo analizar con IA si hay señal O si la confianza es alta
        if signal["signal"] != "ESPERAR":
            log.info(f"{symbol}: 🎯 SEÑAL DETECTADA — llamando a IA...")
            ai_text = ai_analysis(symbol, market_info, signal, recal, hist_trades)

            # 8. Formatear y mandar mensaje
            msg = format_signal_message(symbol, market_info, signal, recal, confidence, ai_text)
            if msg:
                send_telegram(msg)
                save_signal(symbol, signal, confidence, ai_text)
                signals_found.append(f"{symbol} {signal['signal']}")

        # Rate limit Alpha Vantage (5 calls/min en free tier)
        time.sleep(15)

    if not signals_found:
        log.info("Sin señales en este ciclo — todos los mercados en ESPERAR")
        # Opcional: mandar resumen silencioso cada mañana a las 8h
        now = datetime.now()
        if now.hour == 8 and now.minute < 15:
            send_telegram(
                f"📊 *Resumen matinal — {now.strftime('%d/%m/%Y')}*\n"
                f"Sin señales activas. Mercados en modo ESPERAR.\n"
                f"_Próximo chequeo en {CHECK_INTERVAL//3600}h_"
            )
    else:
        log.info(f"Señales encontradas: {', '.join(signals_found)}")

    log.info(f"Ciclo completado — próximo en {CHECK_INTERVAL//3600}h")


def main():
    log.info("🚀 Trading Signals PRO — Iniciando sistema")
    log.info(f"Mercados: {list(MARKETS.keys())}")
    log.info(f"Intervalo: {CHECK_INTERVAL//3600}h | SL×{SL_MULT} TP×{TP_MULT}")
    log.info(f"Telegram: {'✅ configurado' if TELEGRAM_TOKEN else '⚠️ no configurado'}")
    log.info(f"Alpha Vantage: {'✅ configurado' if ALPHAVANTAGE_KEY else '⚠️ no configurado'}")

    while True:
        try:
            run_analysis_cycle()
        except KeyboardInterrupt:
            log.info("Detenido por el usuario")
            break
        except Exception as e:
            log.error(f"Error en ciclo principal: {e}", exc_info=True)
            send_telegram(f"⚠️ *Error en el sistema*\n`{str(e)[:200]}`\nReintentando en 30 min...")
            time.sleep(1800)
            continue

        log.info(f"Esperando {CHECK_INTERVAL//3600}h hasta el próximo ciclo...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
