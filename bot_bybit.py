# -*- coding: utf-8 -*-
import atexit
import csv
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import sqlite3
import subprocess
import requests
import sys
import random
import threading
import time
from datetime import datetime, timedelta
import yaml

import pandas as pd
from cachetools import cached, TTLCache
from cachetools.keys import hashkey
from filelock import FileLock, Timeout
from pybit.unified_trading import HTTP

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from model_trainer import obter_args_padrao, treinar_um_modelo
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if os.getenv("BOT_DEBUG", "0") == "1" else logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    try:
        file_handler = RotatingFileHandler("bot.log", mode='a', maxBytes=5 * 1024 * 1024, backupCount=5,
                                           encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        print(f"ERRO CRÍTICO DE LOG: Não foi possível criar o arquivo 'bot.log'. Verifique as permissões. Erro: {e}")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    logging.info("Arquivo 'config.yaml' carregado com sucesso.")
except FileNotFoundError:
    logging.critical("ERRO CRÍTICO: O arquivo 'config.yaml' não foi encontrado. O bot não pode iniciar.")
    exit()
except yaml.YAMLError as e:
    logging.critical(f"ERRO CRÍTICO: Falha ao ler o arquivo 'config.yaml'. Verifique a sintaxe. Erro: {e}")
    exit()


# Configurações principais do bot
PARES = ["BTCUSDT", "ETHUSDT"]
COOLDOWN_MINUTOS = 0
LOG_CSV = "historico_trades.csv"
LOCK_CSV_FILE = "historico_trades.csv.lock"
DB_PATH = "banco.db"
USE_TESTNET = True
BOT_STATUS_FILE = "bot_status.flag"
PAUSE_STATUS_FILE = "pause_new_trades.flag"
LOCK_FILE = os.path.abspath("bot_instance.lock")
LAST_KNOWN_EQUITY = 0.0

limites_cache = TTLCache(maxsize=len(PARES) * 2, ttl=28800)
ACTIVE_STRATEGY = 'strategy_ml'

# Parâmetros de risco
ALAVANCAGEM = 10  # cuidado com isso aqui

# peguei do config.yaml pra não ficar hardcoded
RISK_PERC = CONFIG['risk_perc']

# metas diárias - calculadas sobre o equity inicial
META_LUCRO_PCT = 0.05  # 5% já tá bom
STOP_PERDA_PCT = 0.03  # 3% de perda máxima

META_LUCRO_DIARIA = None
STOP_PERDA_DIARIO = None

TAKER_FEE = 0.00055  # taxa da bybit

# não arriscar mais que 8% do equity por trade
MAX_NOTIONAL_PERC_EQUITY = 0.08

# controle de threads
PNL_LOCK = threading.Lock()
EQUITY_LOCK = threading.Lock()

EQUITY_INICIAL_DIA = 0.0  # resetado todo dia

# banco de dados
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

limites_cache = TTLCache(maxsize=len(PARES) * 2, ttl=28800)  # 8h
candles_cache = TTLCache(maxsize=100, ttl=600)  # 10min
ACTIVE_STRATEGY = 'strategy_ml'


def treinar_modelos_se_necessario():
    args = obter_args_padrao()
    for par in PARES:
        modelo_saida = f'model_{par}.pkl'
        if not os.path.exists(modelo_saida):
            print(f"[BOT] Modelo de {par} ausente. Iniciando treino padrão...")
            treinar_um_modelo(par, args)
        else:
            print(f"[BOT] Modelo de {par} encontrado: {modelo_saida}. Pulando treino.")


def add_column_if_missing(connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table})")
        existing_columns = [row[1] for row in cursor.fetchall()]

        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            connection.commit()  # Explicita o commit da transação DDL.
            logging.info(f"Coluna '{column}' adicionada com sucesso à tabela '{table}'.")

    except sqlite3.Error as e:
        logging.error(f"Falha de banco de dados ao tentar adicionar coluna '{column}' em '{table}': {e}", exc_info=True)
        raise


def migrar_schema_trades_para_order_id_unique(connection: sqlite3.Connection):
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if cursor.fetchone() is None:
            logging.info("Tabela 'trades' não existe. Será criada. Nenhuma migração necessária.")
            return

        cursor.execute("PRAGMA table_info(trades)")
        columns_info = cursor.fetchall()
        columns = [row[1] for row in columns_info]

        if 'order_id' in columns:
            logging.info("Esquema do banco de dados já está atualizado com 'order_id'. Nenhuma migração necessária.")
            return

        logging.warning(
            "Detectado esquema antigo. Iniciando migração da tabela 'trades' para adicionar 'order_id UNIQUE'.")

        cursor.execute("BEGIN TRANSACTION;")

        cursor.execute("ALTER TABLE trades RENAME TO trades_old;")

        cursor.execute(
            '''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                par TEXT,
                acao TEXT,
                preco_entrada_esperado REAL,
                preco_entrada_executado REAL,
                preco_saida REAL,
                tp REAL,
                sl REAL,
                qtd REAL,
                resultado TEXT,
                pnl_bruto REAL,
                taxas REAL,
                pnl_liquido REAL,
                slippage_percent REAL,
                estrategia TEXT,
                justificativa TEXT,
                order_id TEXT UNIQUE
            )
            '''
        )
        logging.info("Nova tabela 'trades' criada com o esquema correto.")


        cursor.execute("PRAGMA table_info(trades_old)")
        old_columns = [row[1] for row in cursor.fetchall()]
        columns_str = ", ".join(old_columns)

        cursor.execute(f"INSERT INTO trades ({columns_str}) SELECT {columns_str} FROM trades_old;")
        logging.info(f"Dados da tabela antiga copiados para a nova. Linhas movidas: {cursor.rowcount}")


        cursor.execute("DROP TABLE trades_old;")
        logging.info("Tabela 'trades_old' removida com sucesso.")


        connection.commit()
        logging.info("Migração da tabela 'trades' concluída com sucesso.")

    except sqlite3.Error as e:
        logging.error(f"Falha CRÍTICA durante a migração do banco de dados: {e}", exc_info=True)
        connection.rollback()
        logging.info("Transação revertida (rollback) devido a erro.")
        raise


try:
    c = conn.cursor()
    c.execute(
        '''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            par TEXT,
            acao TEXT,
            preco_entrada_esperado REAL,
            preco_entrada_executado REAL,
            preco_saida REAL,
            tp REAL,
            sl REAL,
            qtd REAL,
            resultado TEXT,
            pnl_bruto REAL,
            taxas REAL,
            pnl_liquido REAL,
            slippage_percent REAL,
            estrategia TEXT,
            justificativa TEXT,
            order_id TEXT UNIQUE
        )
        '''
    )
    conn.commit()
except sqlite3.Error as e:
    logging.critical(f"Erro CRÍTICO ao criar a tabela 'trades': {e}", exc_info=True)
    exit()

try:
    migrar_schema_trades_para_order_id_unique(conn)

except sqlite3.Error as e:
    logging.critical(
        f"Falha na migração do esquema do banco de dados: {e}. O bot não pode continuar de forma segura e será encerrado.",
        exc_info=True)
    exit()


ULTIMAS_ENTRADAS = {}
PNL_DIARIO = 0.0
DATA_ATUAL = datetime.now().date()

# Insira suas chaves de API aqui
API_KEY = "API KEY"
API_SECRET = "API KEY"


try:
    session = HTTP(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=USE_TESTNET
    )

    session.get_server_time()
    logging.info("Cliente da API Bybit inicializado e conexão verificada com sucesso.")
except Exception as e:
    logging.critical(
        f"Falha CRÍTICA ao inicializar a sessão da Bybit. Verifique as chaves de API e a conexão. Erro: {e}",
        exc_info=True)
    exit()


def registrar_log_csv_robusto(info: dict) -> None:
    header = [
        "data_hora", "par", "acao", "preco_entrada", "preco_saida", "tp", "sl",
        "resultado", "pnl_usdt", "pnl_brl", "estrategia", "justificativa", "tempo_operacao_min"
    ]
    lock = FileLock(LOCK_CSV_FILE, timeout=5)

    try:
        with lock:
            file_exists = os.path.isfile(LOG_CSV)

            with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore', restval=None)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(info)


    except Timeout:
        logging.error(
            f"Falha ao adquirir lock para escrever no CSV '{LOG_CSV}'. Outro processo pode estar usando o arquivo.")
    except (IOError, PermissionError) as e:
        logging.error(f"Erro de I/O ao escrever no arquivo CSV '{LOG_CSV}': {e}", exc_info=True)
    except (ValueError, KeyError) as e:
        logging.error(f"Erro de dados ao preparar linha para o CSV. Verifique o dicionário 'info'. Erro: {e}",
                      exc_info=True)
    except Exception as e:
        logging.critical(f"Erro inesperado e crítico na função 'registrar_log_csv_robusto': {e}", exc_info=True)


@cached(limites_cache)
def obter_limites(par: str):
    logging.info(
        f"[{par}] Buscando limites via API (não cacheado)...")  # Este log só aparecerá uma vez a cada 8h por par
    try:
        info = session.get_instruments_info(category="linear", symbol=par)["result"]["list"][0]
        min_qty = float(info["lotSizeFilter"]["minOrderQty"])
        qty_step = float(info["lotSizeFilter"]["qtyStep"])
        return min_qty, qty_step
    except Exception as e:
        logging.error(f"Não foi possível obter limites para {par}: {e}. Usando padrões de fallback.")
        return (0.001, 0.001) if "BTC" in par else (0.1, 0.1)


@cached(cache=candles_cache, key=lambda par, limit: hashkey(par, limit))
def obter_candles(par: str, limit: int = 750) -> pd.DataFrame:
    logging.info(f"[{par}] Buscando candles via API (não cacheado)...")
    try:
        dados_resp = session.get_kline(category="linear", symbol=par, interval=INTERVAL, limit=limit)
        dados = dados_resp.get("result", {}).get("list", [])
        if not dados:
            logging.warning(f"Dados de candles vazios para o par {par}.")
            return pd.DataFrame()

        colunas = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(dados, columns=colunas)

        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        colunas_numericas = ["open", "high", "low", "close", "volume", "turnover"]
        for col in colunas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        return df

    except Exception as e:
        logging.error(f"Falha crítica ao obter ou processar candles para {par}: {e}", exc_info=True)
        return pd.DataFrame()


def trailing_sl_monitor(par: str, order_id: str, direcao: str, sl_inicial: float, atr_multiplier: float = 1.8,
                        check_interval_seconds: int = 300):
    logging.info(f"[TRAILING-SL] Monitor iniciado para {par}, Ordem ID: {order_id}")
    sl_atual = sl_inicial

    while True:
        try:
            time.sleep(check_interval_seconds)

            with sqlite3.connect(DB_PATH, check_same_thread=False) as temp_conn:
                trade_status = temp_conn.execute(
                    "SELECT resultado FROM trades WHERE order_id = ?", (order_id,)
                ).fetchone()

            if not trade_status or trade_status[0] != 'aberto':
                logging.info(
                    f"[TRAILING-SL] Posição para {par} (Ordem ID: {order_id}) não está mais 'aberta'. Encerrando monitor.")
                break

            df_candles = obter_candles(par, limit=100)
            if df_candles.empty:
                continue

            df_com_indicadores = apply_all_indicators(df_candles)
            last_candle = df_com_indicadores.iloc[-1]
            preco_atual = last_candle['close']
            atr_atual = last_candle['atr']

            if pd.isna(atr_atual) or atr_atual == 0:
                continue

            distancia_sl = atr_atual * atr_multiplier
            novo_sl = 0.0

            if direcao == 'buy':
                novo_sl = preco_atual - distancia_sl

                if novo_sl > sl_atual:
                    novo_sl = ajustar_preco_por_tick(par, novo_sl, "buy", "sl")
                    logging.info(
                        f"[TRAILING-SL] Melhoria detectada para {par} (BUY). Ajustando SL de {sl_atual:.4f} para {novo_sl:.4f}")
                    session.set_trading_stop(category="linear", symbol=par, stopLoss=str(novo_sl))
                    sl_atual = novo_sl
                    with sqlite3.connect(DB_PATH, check_same_thread=False) as temp_conn:
                        temp_conn.execute("UPDATE trades SET sl = ? WHERE order_id = ?", (novo_sl, order_id))

            elif direcao == 'sell':
                novo_sl = preco_atual + distancia_sl
                if novo_sl < sl_atual:
                    novo_sl = ajustar_preco_por_tick(par, novo_sl, "sell", "sl")
                    logging.info(
                        f"[TRAILING-SL] Melhoria detectada para {par} (SELL). Ajustando SL de {sl_atual:.4f} para {novo_sl:.4f}")
                    session.set_trading_stop(category="linear", symbol=par, stopLoss=str(novo_sl))
                    sl_atual = novo_sl
                    with sqlite3.connect(DB_PATH, check_same_thread=False) as temp_conn:
                        temp_conn.execute("UPDATE trades SET sl = ? WHERE order_id = ?", (novo_sl, order_id))

        except Exception as e:
            logging.error(f"[TRAILING-SL-ERROR] Erro no monitor para {par} (Ordem ID: {order_id}): {e}", exc_info=True)
            time.sleep(check_interval_seconds)


def apply_all_indicators(df: pd.DataFrame):
    df = df.copy()

    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx_indicator.adx()

    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    stoch_indicator = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3)
    df['stoch_k'] = stoch_indicator.stoch()
    df['ema_tendencia'] = df['close'].ewm(span=50,
                                          adjust=False).mean()
    df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'],
                                            window=14).volume_weighted_average_price()

    return df


def strategy_dynamic_momentum_scalper(df: pd.DataFrame):
    prev = df.iloc[-3]
    last = df.iloc[-2]
    current_price = df.iloc[-1]['close']

    required_cols = ['ema_9', 'ema_21', 'rsi', 'atr']
    if any(pd.isna(last[col]) for col in required_cols) or any(pd.isna(prev[col]) for col in ['ema_9', 'ema_21']):
        return None, None, None, "Aguardando dados dos indicadores."

    ema_cross_up = last['ema_9'] > last['ema_21'] and prev['ema_9'] <= prev['ema_21']
    price_above_ema21 = last['close'] > last['ema_21']
    rsi_confirms_buy = last['rsi'] > 50

    if ema_cross_up and price_above_ema21 and rsi_confirms_buy:
        distancia_sl = 1.5 * last['atr']
        sl = current_price - distancia_sl
        tp = current_price + (2 * last['atr'])
        justificativa = f"Scalper Dinâmico: Cruzamento de EMA para Cima (RSI: {last['rsi']:.1f})."
        return "buy", tp, sl, justificativa


    ema_cross_down = last['ema_9'] < last['ema_21'] and prev['ema_9'] >= prev['ema_21']
    price_below_ema21 = last['close'] < last['ema_21']
    rsi_confirms_sell = last['rsi'] < 50

    if ema_cross_down and price_below_ema21 and rsi_confirms_sell:
        distancia_sl = 1.5 * last['atr']
        sl = current_price + distancia_sl
        tp = current_price - (2 * last['atr'])  # Relação Risco/Retorno de ~1:1.33
        justificativa = f"Scalper Dinâmico: Cruzamento de EMA para Baixo (RSI: {last['rsi']:.1f})."
        return "sell", tp, sl, justificativa

    return None, None, None, "Aguardando cruzamento (Scalper Dinâmico)."


def strategy_machine_learning(df: pd.DataFrame, par: str):

    LIMIARES_POR_PAR = CONFIG['ml_thresholds']
    ADX_THRESHOLD = 25
    ATR_MULTIPLIER_SL = 1.8
    ATR_MULTIPLIER_TP = 5.0


    if par not in LIMIARES_POR_PAR:
        return None, None, None, f"Par {par} não possui um limiar de ML definido em config.yaml."

    limiar_de_confianca_atual = LIMIARES_POR_PAR[par]

    if len(df) < 2:
        return None, None, None, "Dados insuficientes para detectar cruzamento."


    candle_anterior = df.iloc[-2]
    candle_atual = df.iloc[-1]


    buy_crossover = candle_atual['ema_9'] > candle_atual['ema_21'] and candle_anterior['ema_9'] <= candle_anterior[
        'ema_21']
    sell_crossover = candle_atual['ema_9'] < candle_atual['ema_21'] and candle_anterior['ema_9'] >= candle_anterior[
        'ema_21']

    sinal_gatilho = None
    if buy_crossover: sinal_gatilho = 'buy'
    if sell_crossover: sinal_gatilho = 'sell'

    if not sinal_gatilho:
        return None, None, None, "Aguardando gatilho de cruzamento de EMAs."


    try:
        adx = candle_atual.get('adx', 0)
        if adx < ADX_THRESHOLD:
            return None, None, None, f"Cruzamento detectado, mas mercado lateral (ADX {adx:.1f} < {ADX_THRESHOLD})."


        preco_atual = candle_atual['close']
        ema_200 = candle_atual['ema_200']
        if (sinal_gatilho == 'buy' and preco_atual < ema_200) or \
                (sinal_gatilho == 'sell' and preco_atual > ema_200):
            return None, None, None, "Cruzamento detectado, mas contra a tendência principal (EMA 200)."


        PREDICT_URL = 'http://localhost:5000/predict'
        TIMEOUT_SECONDS = 3.0

        candle_cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df_payload = df[candle_cols].copy()
        df_payload['timestamp'] = df_payload['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        payload = {'par': par, 'candles': df_payload.to_dict('records')}

        response = requests.post(PREDICT_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        previsao = response.json()

        if not previsao.get('success'):
            error_msg = previsao.get('error', 'Erro desconhecido do servidor.')
            return None, None, None, f"Servidor de ML: {error_msg}"

        prob_alta = previsao.get('prob_alta', 0.0)
        prob_baixa = previsao.get('prob_baixa', 0.0)

        if sinal_gatilho == 'buy' and prob_alta > limiar_de_confianca_atual:
            ultimo_atr = candle_atual['atr']
            if pd.isna(ultimo_atr) or ultimo_atr <= 0: return None, None, None, "ATR inválido para cálculo de TP/SL."
            sl = calcular_stop_loss_dinamico('long', preco_atual, ultimo_atr, atr_multiplier=ATR_MULTIPLIER_SL)
            tp = preco_atual + (ultimo_atr * ATR_MULTIPLIER_TP)
            justificativa = f"HÍBRIDO: Cruzamento de Alta confirmado por ML ({prob_alta:.1%}) e ADX ({adx:.0f})."
            return "buy", tp, sl, justificativa

        elif sinal_gatilho == 'sell' and prob_baixa > limiar_de_confianca_atual:
            ultimo_atr = candle_atual['atr']
            if pd.isna(ultimo_atr) or ultimo_atr <= 0: return None, None, None, "ATR inválido para cálculo de TP/SL."
            sl = calcular_stop_loss_dinamico('short', preco_atual, ultimo_atr, atr_multiplier=ATR_MULTIPLIER_SL)
            tp = preco_atual - (ultimo_atr * ATR_MULTIPLIER_TP)
            justificativa = f"HÍBRIDO: Cruzamento de Baixa confirmado por ML ({prob_baixa:.1%}) e ADX ({adx:.0f})."
            return "sell", tp, sl, justificativa


    except requests.exceptions.ConnectionError:
        return None, None, None, "Servidor de ML offline (falha de conexão)."
    except requests.exceptions.Timeout:
        return None, None, None, f"Servidor de ML não respondeu em {TIMEOUT_SECONDS}s (timeout)."
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"[{par}] Erro HTTP do servidor de ML: {http_err} - Resposta: {http_err.response.text}")
        return None, None, None, f"Erro HTTP {http_err.response.status_code} do servidor de ML."
    except Exception as e:
        logging.critical(f"[{par}] Erro inesperado na estratégia Híbrida: {e}", exc_info=True)
        return None, None, None, "Erro inesperado ao consultar modelo de ML."

    return None, None, None, "Cruzamento detectado, mas ML não confirmou com confiança suficiente."

def strategy_active_day_trader(df: pd.DataFrame):
    prev = df.iloc[-3]
    last = df.iloc[-2]
    current_price = df.iloc[-1]['close']

    required_cols = ['ema_9', 'ema_21', 'ema_tendencia', 'rsi', 'atr', 'volume', 'volume_ma']
    if any(pd.isna(last[col]) for col in required_cols) or any(pd.isna(prev[col]) for col in ['ema_9', 'ema_21']):
        return None, None, None, "Aguardando dados completos dos indicadores."


    is_mid_term_uptrend = last['close'] > last['ema_tendencia']
    ema_cross_up = last['ema_9'] > last['ema_21'] and prev['ema_9'] <= prev['ema_21']
    rsi_confirms_buy = last['rsi'] > 52
    volume_confirms = last['volume'] > last['volume_ma']

    if is_mid_term_uptrend and ema_cross_up and rsi_confirms_buy and volume_confirms:
        distancia_sl = 1.8 * last['atr']
        distancia_tp = 3.0 * last['atr']

        sl = current_price - distancia_sl
        tp = current_price + distancia_tp

        justificativa = f"Day Trader Ativo: Sinal de compra alinhado com tendência de médio prazo (EMA50) e confirmado por volume e RSI ({last['rsi']:.1f})."
        return "buy", tp, sl, justificativa


    is_mid_term_downtrend = last['close'] < last['ema_tendencia']
    ema_cross_down = last['ema_9'] < last['ema_21'] and prev['ema_9'] >= prev['ema_21']
    rsi_confirms_sell = last['rsi'] < 48

    if is_mid_term_downtrend and ema_cross_down and rsi_confirms_sell and volume_confirms:
        distancia_sl = 1.8 * last['atr']
        distancia_tp = 3.0 * last['atr']

        sl = current_price + distancia_sl
        tp = current_price - distancia_tp

        justificativa = f"Day Trader Ativo: Sinal de venda alinhado com tendência de médio prazo (EMA50) e confirmado por volume e RSI ({last['rsi']:.1f})."
        return "sell", tp, sl, justificativa

    return None, None, None, "Aguardando oportunidade de Day Trade com boa probabilidade."


def human_readable_motivo(par: str, motivo: str) -> str:

    motivo_seguro = (motivo or "Motivo não especificado").lower()

    par_curto = par.replace("USDT", "")

    if "aguardando dados" in motivo_seguro or "convergência" in motivo_seguro:
        return f"[{par_curto}] ⏳ Juntando dados (indicadores não prontos). Aguardando um sinal claro..."

    if "aguardando cruzamento" in motivo_seguro or "esperando cruzamento" in motivo_seguro:
        return f"[{par_curto}] 🤏 Nenhum cruzamento de EMAs detectado. Apenas monitorando o mercado."

    if "sem sinal claro" in motivo_seguro:
        return f"[{par_curto}] 🔎 Mercado sem direção definida no momento. Melhor aguardar."

    if "volume" in motivo_seguro and "breakout" in motivo_seguro:
        return f"[{par_curto}] 🔥 Pico de volume detectado. Observando para possível entrada em rompimento."


    return f"[{par_curto}] ℹ️ Status: {motivo.capitalize()}"


def human_intent_log(par: str, sinal: str, preco: float, tp: float, sl: float, qtd_sugerida: float = None) -> str:
    abbrev = par.replace("USDT", "")
    side_br = "COMPRAR" if sinal.lower() == "buy" else "VENDER"
    qty_txt = f" | qty ≈ {qtd_sugerida}" if qtd_sugerida else ""
    return (f"[{par}] 🚀 SINAL: {side_br} {abbrev} — entry ≈ {preco:.6f} | TP {tp:.6f} | SL {sl:.6f}{qty_txt} "
            f"(confirmo execução em seguida).")


def executar_ordem(par: str, direcao: str, preco_esperado: float, tp: float, sl: float, justificativa: str) -> bool:

    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    trade_id_db = None
    order_id = None
    try:
        equity = consultar_saldo_real()
        if not equity or equity <= 0:
            logging.error(f"[{par}] Saldo da conta inválido ou zerado (${equity}). Abortando ordem.")
            return False

        qtd = calcular_quantidade_ordem(
            par, preco_esperado, sl, equity,
            RISK_PERC, MAX_NOTIONAL_PERC_EQUITY, session
        )
        if qtd <= 0:
            raise ValueError(f"Quantidade calculada inválida ({qtd}). Abortando ordem.")


        side = "Buy" if direcao.lower() == "buy" else "Sell"
        risco_em_usd = equity * RISK_PERC


        logging.info(f"[{par}] Sinal de {side.upper()} detectado. Motivo: {justificativa}")
        logging.info(f"    Preço de entrada ideal: ${preco_esperado:,.4f}")
        logging.info(f"    Saldo da conta: ${equity:,.2f}")
        logging.info(f"    Risco da operação ({RISK_PERC:.2%}): ${risco_em_usd:,.2f}")
        logging.info(f"    Enviando ordem de {side.upper()} a mercado...")

        ordem_response = session.place_order(
            category="linear",
            symbol=par,
            side=side,
            order_type="Market",
            qty=str(qtd)
        )

        order_id = (ordem_response or {}).get("result", {}).get("orderId")
        if not order_id:
            raise ValueError(f"API não retornou um orderId. Resposta: {ordem_response}")

        logging.info(f"[{par}] Ordem a mercado enviada. OrderID: {order_id}. Confirmando detalhes...")


        avg_price = None
        for _ in range(12):
            time.sleep(0.5)
            fills = session.get_order_history(category="linear", orderId=order_id, limit=1)
            order_list = (fills or {}).get("result", {}).get("list", [])
            if order_list:
                item = order_list[0]
                avg_price = float(item.get("avgPrice") or 0) or None
                if avg_price:
                    break
        if not avg_price:
            raise RuntimeError(f"Não foi possível obter avgPrice para {par}. Resp: {fills}")

        preco_executado = avg_price

        slippage_percent = ((preco_executado - preco_esperado) / preco_esperado) * 100
        if side == "Sell":
            slippage_percent *= -1


        dist_tp = abs(float(tp) - float(preco_esperado)) if tp else 0.0
        dist_sl = abs(float(sl) - float(preco_esperado)) if sl else 0.0

        if direcao.lower() == "buy":
            tp_price = preco_executado + dist_tp
            sl_price = preco_executado - max(dist_sl, 0.0)
        else:
            tp_price = preco_executado - max(dist_tp, 0.0)
            sl_price = preco_executado + dist_sl

        tp_price = ajustar_preco_por_tick(par, tp_price, direcao.lower(), "tp")
        sl_price = ajustar_preco_por_tick(par, sl_price, direcao.lower(), "sl")

        if direcao.lower() == "sell" and sl_price <= preco_executado:
            sl_price = ajustar_preco_por_tick(par, preco_executado + obter_filtros_instrumento(par)["tickSize"], "sell",
                                              "sl")
        if direcao.lower() == "buy" and sl_price >= preco_executado:
            sl_price = ajustar_preco_por_tick(par, preco_executado - obter_filtros_instrumento(par)["tickSize"], "buy",
                                              "sl")


        tpsl_resp = session.set_trading_stop(
            category="linear",
            symbol=par,
            takeProfit=str(tp_price) if tp else None,
            stopLoss=str(sl_price) if sl else None,
        )
        if (tpsl_resp or {}).get("retCode", 0) != 0:
            raise RuntimeError(f"Falha ao aplicar TP/SL via set_trading_stop para {par}. Resp: {tpsl_resp}")

        with db_conn:
            cursor = db_conn.cursor()
            cursor.execute(
                """INSERT INTO trades (timestamp, par, acao, preco_entrada_esperado, preco_entrada_executado, tp, sl, resultado, estrategia, justificativa, qtd, order_id, slippage_percent)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), par, direcao.lower(), preco_esperado,
                 preco_executado, tp_price, sl_price, "aberto", ACTIVE_STRATEGY, justificativa, qtd, order_id,
                 slippage_percent)
            )
            trade_id_db = cursor.lastrowid

        notional = qtd * preco_executado
        log_final = (
            f"✅ [TRADE EXECUTADO] Posição Aberta [DB ID: {trade_id_db}]\n"
            f"    - Ativo: {par}\n"
            f"    - Ação: {side.upper()} {qtd} {par.replace('USDT', '')}\n"
            f"    - Preço de Entrada Executado: ${preco_executado:,.4f} (Slippage: {slippage_percent:.4f}%)\n"
            f"    - Valor da Posição (Notional): ${notional:,.2f}\n"
            f"    - Take Profit Configurado: ${tp_price:,.4f}\n"
            f"    - Stop Loss Configurado: ${sl_price:,.4f}"
        )
        logging.info(log_final)

        thread_monitor = threading.Thread(
            target=trailing_sl_monitor,
            args=(par, order_id, direcao.lower(), sl_price),
            daemon=True
        )
        thread_monitor.start()

        return True

    except Exception as e:
        logging.critical(f"[{par}] ❌ Falha CRÍTICA no processo de execução de ordem: {e}", exc_info=True)
        return False
    finally:
        if db_conn:
            db_conn.close()


def obter_filtros_instrumento(par: str) -> dict:
    resp = session.get_instruments_info(category="linear", symbol=par)
    info = (resp or {}).get("result", {}).get("list", [])
    if not info:
        raise RuntimeError(f"Não foi possível obter filtros do instrumento para {par}. Resposta: {resp}")
    it = info[0]
    price_filter = it.get("priceFilter", {}) or {}
    lot_filter = it.get("lotSizeFilter", {}) or {}
    return {
        "tickSize": float(price_filter.get("tickSize", "0.0001")),
        "minPrice": float(price_filter.get("minPrice", "0")),
        "maxPrice": float(price_filter.get("maxPrice", "1e12")),
        "qtyStep": float(lot_filter.get("qtyStep", "0.001")),
        "minQty": float(lot_filter.get("minOrderQty", lot_filter.get("minTradingQty", "0.0")) or 0.0)
    }


def _round_step(value: float, step: float, mode: str = "nearest") -> float:
    if step <= 0:
        return value
    q = value / step
    if mode == "down":
        return math.floor(q) * step
    if mode == "up":
        return math.ceil(q) * step
    return round(q) * step


def ajustar_preco_por_tick(par: str, preco: float, direcao: str, tipo: str) -> float:
    f = obter_filtros_instrumento(par)
    tick, pmin, pmax = f["tickSize"], f["minPrice"], f["maxPrice"]

    mode = "nearest"
    if direcao == "buy":
        mode = "up" if tipo == "tp" else "down"
    else:  # sell
        mode = "down" if tipo == "tp" else "up"

    p = _round_step(preco, tick, mode)
    if pmin:
        p = max(p, pmin)
    if pmax and p > pmax:
        p = _round_step(pmax, tick, "down")
    return float(f"{p:.10f}")


def calcular_quantidade_ordem(
        par: str,
        preco_entrada: float,
        preco_sl: float,
        equity_atual: float,
        risco_perc: float,
        max_notional_perc: float,
        session_api: HTTP
) -> float:
    min_qty, qty_step = obter_limites(par)

    risco_por_trade_usd = equity_atual * risco_perc

    distancia_sl = abs(preco_entrada - preco_sl)
    if distancia_sl < 1e-8:
        raise ValueError("Distância do Stop Loss é zero. Impossível calcular quantidade.")

    qtd_bruta = risco_por_trade_usd / distancia_sl

    notional_bruto = qtd_bruta * preco_entrada
    max_notional_permitido = equity_atual * max_notional_perc

    if notional_bruto > max_notional_permitido:
        fator_ajuste = max_notional_permitido / notional_bruto
        qtd_bruta *= fator_ajuste
        logging.warning(f"[{par}] Quantidade ajustada para respeitar notional máximo. "
                        f"Nova qtd: {qtd_bruta:.4f}")

    qtd_ajustada = math.floor(qtd_bruta / qty_step) * qty_step

    if qtd_ajustada < min_qty:
        notional_minimo = min_qty * preco_entrada
        if notional_minimo > max_notional_permitido * 1.5:  # Adiciona uma margem de segurança
            raise ValueError(f"Quantidade mínima ({min_qty}) resultaria em notional "
                             f"(${notional_minimo:.2f}) excessivo. Ordem abortada.")
        logging.warning(f"[{par}] Quantidade calculada ({qtd_ajustada:.4f}) abaixo da mínima ({min_qty}). "
                        f"Usando quantidade mínima.")
        qtd_ajustada = min_qty

    return qtd_ajustada


def sincronizar_posicoes_e_banco(db_conn: sqlite3.Connection, api_session: HTTP):
    logging.info("=" * 20 + " INICIANDO SINCRONIZAÇÃO DE ESTADO " + "=" * 20)
    try:
        posicoes_api_raw = api_session.get_positions(category="linear", settleCoin="USDT")["result"]["list"]
        simbolos_abertos_api = {p['symbol'] for p in posicoes_api_raw if float(p.get('size', 0)) > 0}
        logging.info(f"Posições Abertas na API: {simbolos_abertos_api if simbolos_abertos_api else 'Nenhuma'}")

        trades_abertos_db_cursor = db_conn.execute("SELECT id, par FROM trades WHERE resultado = 'aberto'")
        trades_abertos_db = trades_abertos_db_cursor.fetchall()
        simbolos_abertos_db = {par for id_, par in trades_abertos_db}
        logging.info(f"Trades 'Abertos' no Banco de Dados: {simbolos_abertos_db if simbolos_abertos_db else 'Nenhum'}")


        trades_a_fechar = simbolos_abertos_db - simbolos_abertos_api
        if trades_a_fechar:
            logging.warning(f"Posições para reconciliar (fechar no DB): {trades_a_fechar}")
            for par in trades_a_fechar:
                try:
                    logging.info(f"[SYNC] Tentando reconciliar e fechar o par {par} no DB...")
                    atualizar_pnl_trade_fechado(par, db_conn, api_session, PNL_LOCK)
                    time.sleep(1)
                except Exception as e_reconcile:
                    logging.error(f"[SYNC-FAIL] Falha ao reconciliar o par {par}. "
                                  f"Verificar manualmente. Erro: {e_reconcile}", exc_info=True)

        posicoes_desconhecidas = simbolos_abertos_api - simbolos_abertos_db
        if posicoes_desconhecidas:
            logging.critical(f"ALERTA: Posições detectadas na API que não constam no DB: {posicoes_desconhecidas}")
            logging.critical("Essas posições devem ser gerenciadas manualmente! O bot não irá tocá-las.")


        sincronizados = simbolos_abertos_api.intersection(simbolos_abertos_db)
        if sincronizados:
            logging.info(f"Posições em sincronia e monitoradas: {sincronizados}")

    except Exception as e:
        logging.critical("Erro CRÍTICO durante a sincronização de estado. O bot não pode continuar de forma segura.",
                         exc_info=True)
        raise

    logging.info("=" * 25 + " SINCRONIZAÇÃO CONCLUÍDA " + "=" * 25)


def atualizar_pnl_trade_fechado(par: str, db_conn: sqlite3.Connection, api_session: HTTP, pnl_lock: threading.Lock):
    global PNL_DIARIO

    MAX_RETRIES = 4
    BACKOFF_BASE = 5
    TOLERANCIA_SEGUNDOS = 180

    trade_aberto_sql = """
        SELECT id, timestamp, preco_entrada_executado, qtd, acao, tp, sl, estrategia, justificativa, order_id
        FROM trades
        WHERE par = ? AND resultado = 'aberto'
        ORDER BY id DESC LIMIT 1
    """
    try:
        cursor = db_conn.cursor()
        trade_info = cursor.execute(trade_aberto_sql, (par,)).fetchone()

        if not trade_info:
            logging.warning(f"[RECONCILE] Nenhum trade 'aberto' encontrado para {par} no DB.")
            return

        (trade_id, ts_str, preco_entrada, qtd, acao, tp, sl,
         estrategia, justificativa, order_id) = trade_info
        ts_entrada = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

        historico_fechado = []
        for attempt in range(MAX_RETRIES):
            with pnl_lock:
                try:
                    resp = api_session.get_closed_pnl(category="linear", symbol=par, limit=20)
                    historico_fechado = resp.get("result", {}).get("list", [])
                    if historico_fechado:
                        break
                except Exception as e:
                    logging.error(f"[RECONCILE] Falha ao chamar get_closed_pnl ({par}): {e}", exc_info=True)
                    if attempt == MAX_RETRIES - 1:
                        raise

            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            logging.warning(f"[RECONCILE] Tentativa {attempt + 1}/{MAX_RETRIES} falhou. Retentando em {wait:.1f}s...")
            time.sleep(wait)

        if not historico_fechado:
            logging.error(
                f"[RECONCILE-FAIL] API não retornou histórico de PnL para {par}. Trade ID {trade_id}, Ordem {order_id}.")
            marcar_trade_falhou(db_conn, trade_id, "api_empty")
            return

        try:
            import json
            logging.debug(
                f"[RECONCILE-DEBUG] {par} Resposta completa da API:\n{json.dumps(historico_fechado, indent=2)}")
        except Exception:
            pass


        trade_correto = None
        for trade_api in historico_fechado:
            oid_api = str(trade_api.get("orderId"))
            ts_api = datetime.fromtimestamp(int(trade_api.get("createdTime", 0)) / 1000.0)
            delta = abs((ts_api - ts_entrada).total_seconds())

            if oid_api == order_id:
                trade_correto = trade_api
                logging.info(f"[RECONCILE] Match exato por OrderID em {par}: {order_id}")
                break
            elif delta <= TOLERANCIA_SEGUNDOS:
                trade_correto = trade_api

        if not trade_correto:
            logging.error(
                f"[RECONCILE-FAIL] Nenhum match encontrado para {par} [ID: {trade_id}, Ordem: {order_id}, Tolerância: {TOLERANCIA_SEGUNDOS}s]")
            marcar_trade_falhou(db_conn, trade_id, "match_fail")
            return

        pnl_bruto = float(trade_correto.get("closedPnl", 0.0))
        preco_saida = float(trade_correto.get("avgExitPrice", 0.0))

        if preco_saida == 0.0:
            logging.error(f"[RECONCILE-FAIL] Preço de saída inválido (0.0) em {par} [ID: {trade_id}]")
            marcar_trade_falhou(db_conn, trade_id, "exit_price_zero")
            return

        valor_entrada = preco_entrada * qtd
        valor_saida = preco_saida * qtd
        taxas = (valor_entrada * TAKER_FEE) + (valor_saida * TAKER_FEE)
        pnl_liquido = pnl_bruto - taxas
        resultado = "WIN" if pnl_liquido > 0 else "LOSS"


        with db_conn:
            cursor.execute(
                """UPDATE trades SET
                   resultado = ?, pnl_bruto = ?, preco_saida = ?, taxas = ?, pnl_liquido = ?
                   WHERE id = ?""",
                (resultado, pnl_bruto, preco_saida, taxas, pnl_liquido, trade_id)
            )

        with pnl_lock:
            PNL_DIARIO += pnl_liquido

        resultado_str = f"🏆 VITÓRIA LÍQUIDA de ${pnl_liquido:,.2f}" if pnl_liquido > 0 else f"💔 DERROTA LÍQUIDA de ${pnl_liquido:,.2f}"
        logging.info(
            f"💰 {resultado_str} [RECONCILIADO]: {par} fechado. (Bruto: ${pnl_bruto:,.2f}, Taxas: ${taxas:,.2f})")

        # Registro no CSV
        info_csv = {
            "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "par": par, "acao": acao,
            "preco_entrada": preco_entrada, "preco_saida": preco_saida, "tp": tp, "sl": sl,
            "resultado": resultado, "pnl_usdt": pnl_liquido, "pnl_brl": None,
            "estrategia": estrategia, "justificativa": justificativa, "tempo_operacao_min": None
        }
        registrar_log_csv_robusto(info_csv)

    except (sqlite3.Error, sqlite3.Warning) as e:
        logging.error(f"Erro de Banco de Dados ao reconciliar PnL para {par}: {e}", exc_info=True)
    except (KeyError, ValueError, TypeError) as e:
        logging.error(f"Erro de dados ao processar resposta da API para {par}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Erro inesperado ao apurar PnL para {par}: {e}", exc_info=True)


def marcar_trade_falhou(db_conn: sqlite3.Connection, trade_id: int, motivo: str):
    try:
        cursor = db_conn.cursor()
        cursor.execute("UPDATE trades SET resultado=? WHERE id=?", (f"reconcile_failed:{motivo}", trade_id))
        db_conn.commit()
        logging.warning(f"[RECONCILE] Trade {trade_id} marcado como reconcile_failed ({motivo}).")
    except Exception as e:
        logging.error(f"Falha ao marcar trade {trade_id} como reconcile_failed: {e}", exc_info=True)


def atualizar_resultados_e_pnl():

    TIMEOUT_MINUTES = 30

    try:
        with conn:
            trades_em_aberto_db = conn.execute(
                "SELECT id, par, timestamp FROM trades WHERE resultado = 'aberto'"
            ).fetchall()

        if not trades_em_aberto_db:
            return

        posicoes_api = session.get_positions(category="linear", settleCoin="USDT")["result"]["list"]
        simbolos_abertos_api = {p['symbol'] for p in posicoes_api if float(p.get('size', 0)) > 0}

        for trade_id, par, timestamp_str in trades_em_aberto_db:

            if par not in simbolos_abertos_api:
                logging.info(
                    f"Detectada posição fechada para {par} (ID: {trade_id}). Iniciando reconciliação de PnL...")
                atualizar_pnl_trade_fechado(par, conn, session, PNL_LOCK)
                time.sleep(1)
                continue


            try:
                timestamp_entrada = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


                if (datetime.now() - timestamp_entrada) > timedelta(minutes=TIMEOUT_MINUTES):
                    logging.warning(
                        f"[TIMEOUT] Trade ID {trade_id} ({par}) está aberto por mais de {TIMEOUT_MINUTES} minutos. "
                        f"Marcando como 'timeout_failed' para evitar PnL pendente."
                    )

                    with conn:
                        conn.execute(
                            "UPDATE trades SET resultado = ?, justificativa = ? WHERE id = ?",
                            ('timeout_failed', f'Excedeu o tempo limite de {TIMEOUT_MINUTES} min.', trade_id)
                        )

            except (ValueError, TypeError) as e:
                logging.error(
                    f"Erro ao processar timestamp para o trade ID {trade_id}: {e}. Timestamp no DB: '{timestamp_str}'")


    except Exception as e:
        logging.error(f"Erro crítico ao verificar e atualizar resultados e PnL: {e}", exc_info=True)


def consultar_saldo_real() -> float:
    global LAST_KNOWN_EQUITY

    try:
        response = session.get_wallet_balance(accountType="UNIFIED")

        result_list = response.get("result", {}).get("list", [])

        if not result_list:
            logging.warning("API retornou uma lista de contas vazia. Usando último saldo conhecido.")
            with EQUITY_LOCK:
                return LAST_KNOWN_EQUITY


        balance_data = result_list[0]
        equity_str = balance_data.get("totalEquity")

        if equity_str is None:
            logging.warning(
                "A chave 'totalEquity' não foi encontrada na resposta da API. Usando último saldo conhecido.")
            with EQUITY_LOCK:
                return LAST_KNOWN_EQUITY

        equity = float(equity_str)


        with EQUITY_LOCK:
            LAST_KNOWN_EQUITY = equity
            return equity

    except Exception as e:
        logging.error(f"Falha crítica ao consultar saldo: {e}. Usando último saldo conhecido.", exc_info=True)
        with EQUITY_LOCK:
            return LAST_KNOWN_EQUITY


def verificar_metas_diarias(db_conn: sqlite3.Connection, pnl_lock: threading.Lock) -> bool:
    global DATA_ATUAL, PNL_DIARIO, META_LUCRO_DIARIA, STOP_PERDA_DIARIO, EQUITY_INICIAL_DIA

    hoje = datetime.now().date()

    if hoje != DATA_ATUAL:
        with pnl_lock:
            PNL_DIARIO = 0.0
        DATA_ATUAL = hoje

        try:
            equity_atual = float(consultar_saldo_real())
        except Exception:
            equity_atual = LAST_KNOWN_EQUITY

        EQUITY_INICIAL_DIA = float(equity_atual or 0.0)

        META_LUCRO_DIARIA = EQUITY_INICIAL_DIA * META_LUCRO_PCT
        STOP_PERDA_DIARIO = -EQUITY_INICIAL_DIA * STOP_PERDA_PCT

        logging.info("=" * 66)
        logging.info(f"[DIA {hoje}] Equity inicial: ${EQUITY_INICIAL_DIA:,.2f}")
        logging.info(f"[DIA {hoje}] Metas dinâmicas -> LUCRO: +${META_LUCRO_DIARIA:,.2f} "
                     f"({META_LUCRO_PCT * 100:.2f}%) | STOP: ${STOP_PERDA_DIARIO:,.2f} "
                     f"(-{STOP_PERDA_PCT * 100:.2f}%)")
        logging.info("=" * 66)


    with pnl_lock:
        pnl_atual = float(PNL_DIARIO)

    if META_LUCRO_DIARIA is None or STOP_PERDA_DIARIO is None:
        return True

    if pnl_atual >= META_LUCRO_DIARIA:
        logging.warning(f"[META-DIÁRIA] Lucro do dia atingido: ${pnl_atual:,.2f} "
                        f"(meta: ${META_LUCRO_DIARIA:,.2f}). Pausando operações.")
        return False

    if pnl_atual <= STOP_PERDA_DIARIO:
        logging.warning(f"[STOP-DIÁRIO] Perda do dia atingida: ${pnl_atual:,.2f} "
                        f"(stop: ${STOP_PERDA_DIARIO:,.2f}). Pausando operações.")
        return False

    return True


def verificar_condicoes_de_mercado(df_candles: pd.DataFrame, adx_period=14, adx_threshold=25, atr_period=14,
                                   atr_vol_min_perc=0.15) -> (bool, str):
    if len(df_candles) < adx_period + 5:
        return False, "Dados insuficientes para análise de regime."

    adx_indicator = ADXIndicator(high=df_candles['high'], low=df_candles['low'], close=df_candles['close'],
                                 window=adx_period)
    df_candles['adx'] = adx_indicator.adx()
    df_candles['adx_pos'] = adx_indicator.adx_pos()
    df_candles['adx_neg'] = adx_indicator.adx_neg()

    last_adx = df_candles['adx'].iloc[-1]
    last_di_pos = df_candles['adx_pos'].iloc[-1]
    last_di_neg = df_candles['adx_neg'].iloc[-1]

    is_trending = last_adx > adx_threshold

    atr_indicator = AverageTrueRange(high=df_candles['high'], low=df_candles['low'], close=df_candles['close'],
                                     window=atr_period)
    df_candles['atr'] = atr_indicator.average_true_range()

    last_atr = df_candles['atr'].iloc[-1]
    last_close = df_candles['close'].iloc[-1]

    volatility_perc = (last_atr / last_close) * 100
    has_min_volatility = volatility_perc > atr_vol_min_perc

    # 3. Lógica de Decisão Final
    if not is_trending:
        return False, f"MERCADO LATERAL (ADX {last_adx:.2f} <= {adx_threshold}). Operações bloqueadas."

    if not has_min_volatility:
        return False, f"VOLATILIDADE BAIXA (ATR {volatility_perc:.2f}% <= {atr_vol_min_perc}%). Operações bloqueadas."

    # Determina a direção da tendência
    trend_direction = "ALTA" if last_di_pos > last_di_neg else "BAIXA"

    return True, f"MERCADO EM TENDÊNCIA DE {trend_direction} (ADX {last_adx:.2f}) com volatilidade ({volatility_perc:.2f}%). Sinais permitidos."


def calcular_tamanho_posicao(capital_total, risco_por_trade_perc, preco_entrada, stop_loss_preco, simbolo):
    risco_financeiro = capital_total * (risco_por_trade_perc / 100.0)
    distancia_stop = abs(preco_entrada - stop_loss_preco)

    if distancia_stop == 0:
        return 0

    tamanho_posicao = risco_financeiro / distancia_stop

    if simbolo in ["BTCUSDT", "ETHUSDT"]:
        return round(tamanho_posicao, 3)
    else:
        return round(tamanho_posicao, 1)



def calcular_stop_loss_dinamico(tipo_ordem, preco_entrada, ultimo_atr, atr_multiplier=2.0):
    distancia_stop = ultimo_atr * atr_multiplier

    if tipo_ordem == 'long':
        stop_loss_preco = preco_entrada - distancia_stop
    elif tipo_ordem == 'short':
        stop_loss_preco = preco_entrada + distancia_stop
    else:
        raise ValueError("Tipo de ordem inválido. Use 'long' ou 'short'.")

    return stop_loss_preco


def iniciar_logica_bot():
    logging.info("=" * 50)
    logging.info(f"===== BOT MULTI-ESTRATÉGIA V4.5 CORRIGIDO =====")  # Version bump para controle
    logging.info(f"ESTRATÉGIA ATIVA: {ACTIVE_STRATEGY.upper()}")
    logging.info(
        f"RISCO/TRADE: {RISK_PERC * 100:.2f}% do saldo | Metas dinâmicas: +{META_LUCRO_PCT * 100:.2f}% / -{STOP_PERDA_PCT * 100:.2f}% sobre o equity inicial do dia"
    )

    strategy_map = {
        'dynamic_momentum_scalper': (strategy_dynamic_momentum_scalper, "1"),
        'active_day_trader': (strategy_active_day_trader, "5"),
        'strategy_ml': (strategy_machine_learning, "5")
    }

    if ACTIVE_STRATEGY not in strategy_map:
        logging.critical(f"Estratégia '{ACTIVE_STRATEGY}' é inválida! Verifique as configurações.")
        exit()

    global INTERVAL
    active_strategy_func, INTERVAL = strategy_map[ACTIVE_STRATEGY]
    logging.info(f"Timeframe para a estratégia '{ACTIVE_STRATEGY}': {INTERVAL} minuto(s).")

    treinar_modelos_se_necessario()

    MIN_CANDLES_REQUIRED = 500

    lock = FileLock(LOCK_FILE, timeout=10)

    try:
        with lock:
            logging.info(
                f"Trava de instância adquirida com sucesso ({LOCK_FILE}). O bot é a única instância em execução.")

            try:
                sincronizar_posicoes_e_banco(conn, session)
            except Exception as e_sync:
                logging.critical(f"Erro CRÍTICO ao sincronizar posições na inicialização: {e_sync}", exc_info=True)
                return

            while True:
                try:
                    if not os.path.exists(BOT_STATUS_FILE):
                        logging.warning(
                            "Bot pausado pelo painel (arquivo 'bot_status.flag' não encontrado). Aguardando...")
                        time.sleep(10)
                        continue

                    if not verificar_metas_diarias(conn, PNL_LOCK):
                        logging.info(
                            "Meta diária de lucro ou stop de perda atingido. Pausando novas operações até o próximo dia.")
                        time.sleep(3600)
                        continue

                    if os.path.exists(PAUSE_STATUS_FILE):
                        logging.info("PAUSADO: Novas entradas desativadas. Apenas monitorando posições abertas.")
                        try:
                            atualizar_resultados_e_pnl()
                        except Exception as e_upd:
                            logging.error(f"Erro ao atualizar resultados durante PAUSA: {e_upd}", exc_info=True)
                        time.sleep(30)
                        continue

                    try:
                        saldo_real = consultar_saldo_real()
                        atualizar_resultados_e_pnl()
                    except Exception as e_upd_state:
                        logging.error(f"Erro ao atualizar estado financeiro (saldo/pnl): {e_upd_state}", exc_info=True)
                        saldo_real = LAST_KNOWN_EQUITY  # Fallback para o último saldo conhecido

                    logging.info(
                        f"--- [SALDO: ${saldo_real:.2f} | PNL DIA: ${PNL_DIARIO:.2f}] Analisando com {ACTIVE_STRATEGY.upper()} ---")

                    for par in PARES:
                        try:
                            with conn:
                                trade_aberto_info = conn.execute(
                                    "SELECT 1 FROM trades WHERE par = ? AND resultado = 'aberto' LIMIT 1",
                                    (par,)
                                ).fetchone()

                            if trade_aberto_info:
                                logging.debug(
                                    f"Já existe uma posição aberta para {par}. Monitorando e pulando análise.")
                                continue

                            agora = datetime.now()
                            if par in ULTIMAS_ENTRADAS and (agora - ULTIMAS_ENTRADAS[par]) < timedelta(
                                    minutes=COOLDOWN_MINUTOS):
                                logging.info(f"[COOLDOWN] {par} | Aguardando tempo para nova análise.")
                                continue

                            df_candles = obter_candles(par, limit=MIN_CANDLES_REQUIRED + 250)

                            if df_candles.empty or len(df_candles) < MIN_CANDLES_REQUIRED:
                                logging.warning(
                                    f"[{par}] Histórico de velas insuficiente recebido da exchange ({len(df_candles)} velas), "
                                    f"necessário no mínimo {MIN_CANDLES_REQUIRED}. Pulando análise neste ciclo."
                                )
                                continue

                            df_com_indicadores = apply_all_indicators(df_candles)

                            sinal, tp, sl, motivo = active_strategy_func(df_com_indicadores, par)

                            if sinal:
                                preco_de_entrada = float(df_com_indicadores.iloc[-1]['close'])
                                if executar_ordem(par, sinal, preco_de_entrada, tp, sl, motivo):
                                    ULTIMAS_ENTRADAS[par] = datetime.now()
                                    time.sleep(5)
                            else:
                                logging.info(f"[{par}] Nenhuma condição de trade atendida. Motivo: {motivo}")


                        except Exception as e_par:
                            logging.error(f"Falha crítica ao processar o par {par}: {e_par}", exc_info=True)

                        time.sleep(1.2)

                    logging.info("--- Ciclo de análise completo. Aguardando 15 segundos. ---")
                    time.sleep(15)

                except KeyboardInterrupt:
                    logging.info("Bot interrompido pelo usuário via Ctrl+C. Realizando limpeza e desligando...")
                    break
                except Exception as e_loop:
                    logging.critical(f"ERRO GRAVE E INESPERADO NO LOOP PRINCIPAL: {e_loop}", exc_info=True)
                    logging.info("Aguardando 60 segundos antes de tentar novamente para evitar sobrecarga.")
                    time.sleep(60)

    except Timeout:
        logging.error("=" * 60)
        logging.error("ERRO CRÍTICO: Não foi possível adquirir a trava de execução.")
        logging.error("Outra instância do bot já está em execução. Abortando.")
        logging.error(f"Verifique o arquivo de trava: {os.path.abspath(LOCK_FILE)}")
        logging.error("=" * 60)
    finally:
        try:
            if 'conn' in locals() and conn:
                conn.close()
                logging.info("Conexão com o banco de dados foi finalizada com sucesso.")
        except Exception as e_close:
            logging.error(f"Erro ao fechar conexão com o DB: {e_close}", exc_info=True)
        logging.info("Bot finalizado.")


if __name__ == "__main__":
    setup_logging()

    trainer_script = 'model_trainer.py'
    server_script = 'inference_server.py'
    panel_script = 'painel.py'

    try:
        logging.info("Verificando a integridade dos arquivos de dados necessários...")
        for par in PARES:
            data_file = os.path.join("data", f"dados_historicos_{par}_5m.csv")
            if not os.path.exists(data_file):
                logging.critical(f"ARQUIVO DE DADOS ESSENCIAL NÃO ENCONTRADO: '{data_file}'")
                logging.critical(
                    "Por favor, forneça os arquivos CSV de dados históricos na pasta 'data' antes de iniciar o bot.")
                exit(1)
        logging.info("✅ Todos os arquivos de dados necessários foram encontrados.")

        logging.info("Verificando a integridade dos arquivos de modelo...")
        todos_modelos_ok = True

        from model_trainer import PARES_PARA_TREINAR

        for par in PARES_PARA_TREINAR:
            model_file = f"model_{par}.pkl"
            if not os.path.exists(model_file):
                logging.warning(f"Arquivo de modelo para {par} ('{model_file}') não encontrado.")
                todos_modelos_ok = False
                break

        if not todos_modelos_ok:
            logging.info("Nem todos os arquivos de modelo foram encontrados. Executando o treinador...")
            subprocess.run([sys.executable, trainer_script], check=True, text=True, encoding='utf-8', errors='replace')
            logging.info("Treinador de modelo concluído com sucesso.")
        else:
            logging.info("✅ Todos os modelos de ML necessários foram encontrados.")

        background_processes = []


        def cleanup_processes():
            logging.info("Encerrando processos em segundo plano...")
            for p in background_processes:
                if p.poll() is None:
                    try:
                        p.terminate()
                        p.wait(timeout=5)
                        logging.info(f"Processo {p.pid} encerrado.")
                    except Exception as e:
                        logging.warning(f"Não foi possível encerrar o processo {p.pid}: {e}")
            logging.info("Limpeza de processos concluída.")


        atexit.register(cleanup_processes)

        logging.info("Iniciando o servidor de inferência (Cérebro) em segundo plano...")
        server_process = subprocess.Popen([sys.executable, server_script])
        background_processes.append(server_process)
        logging.info(f"Servidor de inferência iniciado com sucesso. PID: {server_process.pid}")

        logging.info("Aguardando 7 segundos para o servidor de ML aquecer...")
        time.sleep(7)

        logging.info("Iniciando painel NiceGUI em processo separado...")
        panel_process = subprocess.Popen([sys.executable, panel_script])
        background_processes.append(panel_process)
        logging.info(f"Painel NiceGUI iniciado com sucesso. PID: {panel_process.pid}")

        logging.info("=" * 60)
        logging.info(">>> SISTEMA PRONTO E ONLINE. INICIANDO LÓGICA DE TRADING. <<<")
        logging.info("=" * 60)
        iniciar_logica_bot()

    except subprocess.CalledProcessError as e:
        logging.critical(f"Falha CRÍTICA em um script auxiliar: '{' '.join(e.cmd)}'. O bot não pode iniciar.")
        logging.critical(f"Saída do script:\n--- SCRIPT STDOUT ---\n{e.stdout}\n--- SCRIPT STDERR ---\n{e.stderr}")
        logging.critical("Verifique os logs do script que falhou para mais detalhes.")
    except FileNotFoundError as e:
        logging.critical(f"Arquivo de script não encontrado: {e.filename}. Verifique se todos os .py estão na pasta.")
    except Exception as e:
        logging.critical(f"Ocorreu um erro inesperado durante a orquestração: {e}", exc_info=True)
    finally:
        logging.info("--- SCRIPT PRINCIPAL FINALIZADO ---")