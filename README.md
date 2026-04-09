# Trading Signals PRO 🚀

Sistema automático de señales de trading que:
- Analiza 6 mercados cada 4 horas en tiempo real
- Recalibra parámetros automáticamente con los últimos 90 días
- Consulta a Claude (IA) para análisis fundamental con búsqueda web
- Calcula puntuación de confianza 0-100%
- Manda alertas a Telegram cuando hay señal

## Estrategia validada
- Triple EMA (21/50/200) + MACD cross + RSI zona neutra
- 20 años de datos históricos | PF 1.47 | WR 50.4% | 80% años positivos
- SL: 2×ATR | TP: 3.5×ATR (auto-optimizable)

---

## 🔧 Setup en 4 pasos

### Paso 1 — Clona el repo y configura variables

```bash
git clone https://github.com/TU_USUARIO/trading-signals.git
cd trading-signals
cp .env.example .env
# Edita .env con tus API keys
```

### Paso 2 — Consigue las API keys gratuitas

**Alpha Vantage** (datos de mercado):
1. Ve a https://alphavantage.co/support/#api-key
2. Regístrate gratis → copia tu API key
3. Pégala en ALPHAVANTAGE_KEY

**Telegram Bot** (alertas en móvil):
1. Abre Telegram → busca `@BotFather`
2. Escribe `/newbot` → ponle nombre → copia el token
3. Busca `@userinfobot` → te dice tu chat_id
4. Pega ambos en TELEGRAM_TOKEN y TELEGRAM_CHAT_ID

**Anthropic** (análisis IA):
1. Ve a https://console.anthropic.com
2. API Keys → Create Key → copia
3. Pega en ANTHROPIC_KEY

### Paso 3 — Deploy en Railway

1. Ve a https://railway.app → New Project → Deploy from GitHub
2. Selecciona tu repo `trading-signals`
3. Ve a Variables → añade todas las del .env.example
4. Railway despliega automáticamente ✅

### Paso 4 — Verifica que funciona

En los logs de Railway deberías ver:
```
🚀 Trading Signals PRO — Iniciando sistema
Analizando XAUUSD (Oro)...
XAUUSD: 150 velas H4 descargadas
XAUUSD: Señal = ESPERAR | RSI=59.9 | Conds=1/4
Ciclo completado — próximo en 4h
```

---

## 📱 Mensaje de alerta en Telegram

Cuando hay señal recibirás algo así:

```
🟢 SEÑAL DETECTADA 🟢
🥇 Oro (XAUUSD) — 📈 LONG — COMPRAR

📊 Confianza: 84%
████████░░

💰 Niveles de operación:
  • Entrada:    3312.50
  • Stop Loss:  3245.80 (-2.0%)
  • Take Profit:3545.90 (+3.5%)
  • Parámetros: SL×2.0 TP×3.5 (auto-optimizado, PF reciente: 1.82)

📋 Condiciones (4/4):
  ✅ EMA alineadas (ALCISTA)
  ✅ EMA200 pendiente correcta
  ✅ MACD cruce confirmado
  ✅ RSI zona neutra (48.3)

📈 Indicadores:
  RSI: 48.3 | ATR: 33.35 | MACD: +0.00021

🤖 Análisis IA:
El oro mantiene sesgo alcista apoyado por incertidumbre 
geopolítica y expectativas de bajada de tipos. Sin eventos 
macro críticos hasta el jueves (CPI USA)...

⏰ Vela H4: 2026-04-09 08:00
Gestión: máximo 2% del capital en riesgo
```

---

## ⚙️ Configuración avanzada

### Añadir más mercados

En `main.py`, edita el diccionario `MARKETS`:

```python
MARKETS = {
    "XAUUSD": {"name": "Oro",     "emoji": "🥇", "only_long": True,  "type": "physical_currency"},
    "EURUSD": {"name": "EUR/USD", "emoji": "💶", "only_long": False, "type": "physical_currency"},
    "USDCHF": {"name": "USD/CHF", "emoji": "🇨🇭", "only_long": False, "type": "physical_currency"},
    # Añade más aquí:
    # "GBPUSD": {"name": "GBP/USD", "emoji": "💷", "only_long": False, "type": "physical_currency"},
}
```

### Cambiar parámetros via Railway Variables

| Variable | Default | Descripción |
|----------|---------|-------------|
| CHECK_INTERVAL | 14400 | Segundos entre ciclos (4h) |
| SL_MULT | 2.0 | Multiplicador Stop Loss (×ATR) |
| TP_MULT | 3.5 | Multiplicador Take Profit (×ATR) |
| RSI_LOW | 38 | RSI mínimo para señal |
| RSI_HIGH | 62 | RSI máximo para señal |

### Ver el log de señales

En Railway → tu proyecto → Files → `/data/signals_log.json`

---

## 🏗️ Arquitectura

```
Railway (24/7)
│
├── main.py ← loop principal cada 4h
│   ├── fetch_candles_alphavantage() ← datos OHLC en tiempo real
│   ├── calculate_indicators()       ← EMA/MACD/RSI/ATR
│   ├── recalibrate_parameters()     ← auto-optimización 90 días
│   ├── analyze_signal()             ← detecta señal
│   ├── ai_analysis()                ← Claude + web search
│   ├── compute_confidence()         ← score 0-100%
│   └── send_telegram()              ← alerta al móvil
│
└── signals_log.json ← historial de señales
```

---

## ❓ FAQ

**¿Necesito tener el PC encendido?**
No. Railway corre 24/7 en la nube.

**¿Es gratuito?**
Alpha Vantage: gratis (5 calls/min). Railway: ~$5/mes (tu plan ya lo cubre).
Anthropic: ~$0.01 por análisis = ~$2-3/mes con uso normal.

**¿Puedo añadir índices (SP500, Nasdaq)?**
Alpha Vantage tiene SPY, QQQ como ETFs. Los puedes añadir al diccionario MARKETS.

**¿Cómo mejoro la estrategia?**
La recalibración automática (walk-forward de 90 días) ajusta SL/TP cada ciclo.
Para cambios más profundos, edita los parámetros en Railway Variables sin tocar código.

**¿Puedo conectarlo con MT5 para que opere solo?**
Sí — es el siguiente paso. Requiere añadir la librería MetaTrader5 Python y
configurar las credenciales del broker.
