@echo off
echo ========================================
echo  MT5 Bridge - Trading Signals PRO
echo  Setup para Windows
echo ========================================
echo.

echo [1/3] Instalando dependencias Python...
pip install MetaTrader5 requests flask

echo.
echo [2/3] Instalando ngrok para exponer puerto...
echo Descarga ngrok desde https://ngrok.com/download
echo Descomprime ngrok.exe en esta carpeta
echo.

echo [3/3] Configurando variables de entorno...
echo.
echo Edita las siguientes variables en un archivo .env:
echo   TELEGRAM_TOKEN=tu_token_aqui
echo   TELEGRAM_CHAT_ID=tu_chat_id_aqui
echo   BRIDGE_SECRET=elige_una_clave_secreta
echo   BRIDGE_PORT=5001
echo.

echo ========================================
echo  Para arrancar el bridge:
echo.
echo  1. Abre MetaTrader 5
echo  2. Ejecuta: python mt5_bridge.py
echo  3. Ejecuta: ngrok http 5001
echo  4. Copia la URL de ngrok (https://xxxx.ngrok.io)
echo  5. Pon esa URL en Railway como BRIDGE_URL
echo ========================================
pause
