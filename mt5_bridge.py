"""
MT5 Bridge — Trading Signals PRO
Corre en tu PC Windows con MT5 instalado.
Recibe órdenes desde Railway via Telegram webhook y las ejecuta en MT5.

Instalación:
  pip install MetaTrader5 requests flask

Arranque:
  python mt5_bridge.py
"""

import os
import json
import time
import logging
import threading
import requests
import MetaTrader5 as mt5
from datetime import datetime
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "TU_TOKEN_AQUI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "TU_CHAT_ID_AQUI")
BRIDGE_SECRET    = os.environ.get("BRIDGE_SECRET", "clave_secreta_bridge")

# MT5: ruta al ejecutable (ajusta si es diferente)
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Puerto local del bridge
PORT = int(os.environ.get("BRIDGE_PORT", "5001"))

# Símbolo MT5 por activo de nuestro sistema
SYMBOL_MAP = {
    "XAUUSD": "XAUUSD",   # Oro
    "EURUSD": "EURUSD",   # Euro/Dólar
    "USDCHF": "USDCHF",   # Dólar/Franco
    "US500":  "SP500m",   # S&P500 (ajusta al símbolo exacto de tu broker)
    "NDAQ":   "NAS100m",  # Nasdaq (ajusta al símbolo exacto de tu broker)
}

# Operaciones pendientes de confirmación
PENDING_OPS = {}

app = Flask(__name__)

# ── MT5 CONEXIÓN ──────────────────────────────────────────────────────────────
def connect_mt5() -> bool:
    """Conecta con MetaTrader 5."""
    if not mt5.initialize(MT5_PATH):
        log.error(f"MT5 no conectado: {mt5.last_error()}")
        return False
    info = mt5.terminal_info()
    log.info(f"MT5 conectado: {info.company} | Cuenta: {mt5.account_info().login}")
    return True


def get_lot_size(symbol: str, risk_pct: float, entry: float, sl: float,
                 account_balance: float) -> float:
    """
    Calcula el tamaño de lote exacto para arriesgar risk_pct% del balance.
    """
    risk_amount   = account_balance * risk_pct / 100
    sl_points     = abs(entry - sl)
    symbol_info   = mt5.symbol_info(symbol)

    if not symbol_info:
        log.error(f"Símbolo {symbol} no encontrado en MT5")
        return 0.01

    point      = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    tick_size  = symbol_info.trade_tick_size

    if tick_size == 0:
        return 0.01

    # Valor monetario por lote por punto
    value_per_lot = (sl_points / tick_size) * tick_value

    if value_per_lot == 0:
        return 0.01

    lot_size = risk_amount / value_per_lot

    # Redondear al volumen mínimo del broker
    min_lot  = symbol_info.volume_min
    lot_step = symbol_info.volume_step
    lot_size = max(min_lot, round(lot_size / lot_step) * lot_step)
    lot_size = min(lot_size, symbol_info.volume_max)

    log.info(f"Lote calculado: {lot_size:.2f} ({risk_pct}% de ${account_balance:,.0f})")
    return round(lot_size, 2)


def execute_order(op_id: str) -> dict:
    """
    Ejecuta la operación confirmada en MT5.
    Devuelve resultado con detalle del ticket.
    """
    if op_id not in PENDING_OPS:
        return {"ok": False, "error": "Operación no encontrada o expirada"}

    op      = PENDING_OPS[op_id]
    symbol  = SYMBOL_MAP.get(op["symbol"], op["symbol"])
    direction = mt5.ORDER_TYPE_BUY if op["signal"] == "LONG" else mt5.ORDER_TYPE_SELL

    if not connect_mt5():
        return {"ok": False, "error": "MT5 no disponible"}

    # Obtener precio actual
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return {"ok": False, "error": f"No se pudo obtener precio de {symbol}"}

    price = tick.ask if op["signal"] == "LONG" else tick.bid

    # Balance de la cuenta
    account  = mt5.account_info()
    balance  = account.balance
    lot_size = get_lot_size(symbol, op["risk_pct"], price, op["sl"], balance)

    if lot_size <= 0:
        return {"ok": False, "error": "Lote calculado inválido"}

    # Preparar orden
    request_mt5 = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot_size,
        "type":         direction,
        "price":        price,
        "sl":           op["sl"],
        "tp":           op["tp"],
        "deviation":    20,
        "magic":        20260409,
        "comment":      f"TradingSignals|{op['symbol']}|{op_id[:8]}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request_mt5)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_msg = f"Error MT5: {result.retcode} — {result.comment}"
        log.error(error_msg)
        return {"ok": False, "error": error_msg, "retcode": result.retcode}

    log.info(f"✅ Orden ejecutada: #{result.order} | {symbol} | {op['signal']} | {lot_size} lotes")

    # Limpiar pendiente
    del PENDING_OPS[op_id]

    return {
        "ok":       True,
        "ticket":   result.order,
        "symbol":   symbol,
        "signal":   op["signal"],
        "lots":     lot_size,
        "price":    price,
        "sl":       op["sl"],
        "tp":       op["tp"],
        "balance":  balance,
        "risk_pct": op["risk_pct"],
    }


# ── TELEGRAM HELPERS ──────────────────────────────────────────────────────────
def send_telegram(text: str, reply_markup=None) -> bool:
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.ok
    except Exception as e:
        log.error(f"Telegram: {e}")
        return False


def answer_callback(callback_query_id: str, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
    requests.post(url, data={
        "callback_query_id": callback_query_id,
        "text":              text,
        "show_alert":        False,
    }, timeout=5)


def edit_message_buttons(chat_id: str, message_id: int, text: str):
    """Edita el mensaje original para quitar los botones tras confirmar."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
    requests.post(url, data={
        "chat_id":    chat_id,
        "message_id": message_id,
        "text":       text,
        "parse_mode": "Markdown",
    }, timeout=5)


# ── FLASK ENDPOINTS ───────────────────────────────────────────────────────────
@app.route("/health")
def health():
    mt5_ok = mt5.initialize() if not mt5.terminal_info() else True
    return jsonify({"ok": True, "mt5": mt5_ok, "pending": len(PENDING_OPS)})


@app.route("/nueva_operacion", methods=["POST"])
def nueva_operacion():
    """
    Railway llama a este endpoint cuando hay señal.
    Recibe los datos y manda Telegram con botones de confirmación.
    """
    data   = request.json or {}
    secret = data.get("secret", "")

    if secret != BRIDGE_SECRET:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    # Generar ID único para esta operación
    op_id = f"op_{int(time.time())}_{data.get('symbol','X')}"

    # Guardar operación pendiente (expira en 30 minutos)
    PENDING_OPS[op_id] = {
        "symbol":   data.get("symbol"),
        "signal":   data.get("signal"),
        "entry":    data.get("entry"),
        "sl":       data.get("sl"),
        "tp":       data.get("tp"),
        "sl_pct":   data.get("sl_pct"),
        "tp_pct":   data.get("tp_pct"),
        "risk_pct": data.get("risk_pct", 1.5),
        "conf":     data.get("confidence", 0),
        "ts":       time.time(),
        "expires":  time.time() + 1800,  # 30 min
    }

    op = PENDING_OPS[op_id]
    is_long  = op["signal"] == "LONG"
    emoji    = "🟢" if is_long else "🔴"
    arrow    = "📈 LONG" if is_long else "📉 SHORT"

    # Mensaje de confirmación
    msg = (
        f"{emoji} *SEÑAL DETECTADA — PENDIENTE DE CONFIRMACIÓN*\n\n"
        f"*{data.get('name','?')} ({op['symbol']})* — {arrow}\n\n"
        f"💰 *Niveles propuestos:*\n"
        f"  Entrada:     `{op['entry']}`\n"
        f"  Stop Loss:   `{op['sl']}` (-{op['sl_pct']}%)\n"
        f"  Take Profit: `{op['tp']}` (+{op['tp_pct']}%)\n\n"
        f"📊 Confianza: *{op['conf']}%* | Riesgo: *{op['risk_pct']}%* del capital\n\n"
        f"⏱️ _Esta confirmación expira en 30 minutos_"
    )

    # Botones inline de Telegram
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "✅ CONFIRMAR Y OPERAR", "callback_data": f"confirm:{op_id}"},
            ],
            [
                {"text": "❌ Cancelar",           "callback_data": f"cancel:{op_id}"},
                {"text": "📊 Ver análisis IA",    "callback_data": f"ai:{op_id}"},
            ]
        ]
    }

    send_telegram(msg, reply_markup=keyboard)

    # Limpiar operaciones expiradas
    now = time.time()
    expired = [k for k, v in PENDING_OPS.items() if v["expires"] < now]
    for k in expired:
        del PENDING_OPS[k]

    return jsonify({"ok": True, "op_id": op_id})


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Recibe callbacks de los botones de Telegram.
    """
    update = request.json or {}

    if "callback_query" not in update:
        return jsonify({"ok": True})

    cb         = update["callback_query"]
    cb_id      = cb["id"]
    data_str   = cb.get("data", "")
    message    = cb.get("message", {})
    message_id = message.get("message_id")
    chat_id    = message.get("chat", {}).get("id")

    if ":" not in data_str:
        return jsonify({"ok": True})

    action, op_id = data_str.split(":", 1)

    # ── CONFIRMAR ──────────────────────────────────────────────────────────────
    if action == "confirm":
        answer_callback(cb_id, "⏳ Ejecutando orden en MT5...")

        # Ejecutar en hilo separado para no bloquear
        def run_order():
            result = execute_order(op_id)
            if result["ok"]:
                msg_ok = (
                    f"✅ *ORDEN EJECUTADA*\n\n"
                    f"Ticket: #{result['ticket']}\n"
                    f"Símbolo: {result['symbol']} {result['signal']}\n"
                    f"Precio entrada: `{result['price']:.5f}`\n"
                    f"Lotes: {result['lots']}\n"
                    f"Stop Loss: `{result['sl']:.5f}`\n"
                    f"Take Profit: `{result['tp']:.5f}`\n"
                    f"Riesgo: {result['risk_pct']}% = ${result['balance']*result['risk_pct']/100:,.0f}\n\n"
                    f"_Gestión: máximo 48h en la operación_"
                )
                send_telegram(msg_ok)
                if message_id and chat_id:
                    edit_message_buttons(chat_id, message_id,
                        f"✅ *EJECUTADO* — #{result['ticket']} | {result['symbol']} {result['signal']}")
            else:
                send_telegram(f"❌ *Error al ejecutar:*\n`{result['error']}`")

        threading.Thread(target=run_order, daemon=True).start()

    # ── CANCELAR ───────────────────────────────────────────────────────────────
    elif action == "cancel":
        answer_callback(cb_id, "❌ Operación cancelada")
        if op_id in PENDING_OPS:
            del PENDING_OPS[op_id]
        if message_id and chat_id:
            edit_message_buttons(chat_id, message_id,
                "❌ *Operación cancelada* — señal ignorada")

    # ── VER ANÁLISIS IA ────────────────────────────────────────────────────────
    elif action == "ai":
        answer_callback(cb_id, "Cargando análisis IA...")
        if op_id in PENDING_OPS:
            op = PENDING_OPS[op_id]
            ai_text = op.get("ai_text", "Análisis no disponible")
            send_telegram(f"🤖 *Análisis IA — {op['symbol']}*\n\n{ai_text}")

    return jsonify({"ok": True})


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("🚀 MT5 Bridge arrancando...")

    # Verificar MT5
    if connect_mt5():
        acc = mt5.account_info()
        log.info(f"✅ MT5 OK — Balance: ${acc.balance:,.0f} | Demo: {acc.trade_mode == 0}")
    else:
        log.warning("⚠️ MT5 no conectado — asegúrate de que MetaTrader 5 esté abierto")

    log.info(f"🌐 Bridge escuchando en puerto {PORT}")
    log.info(f"   /health         → estado del bridge")
    log.info(f"   /nueva_operacion → recibe señales de Railway")
    log.info(f"   /webhook        → callbacks de botones Telegram")

    # En producción usar ngrok o VPS para exponer el puerto
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
