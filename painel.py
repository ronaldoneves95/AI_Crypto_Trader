from nicegui import ui, app
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime, timedelta
import threading
import math
import plotly.graph_objects as go
from pybit.unified_trading import HTTP
import asyncio

DB_PATH = "banco.db"
LOG_FILE = "bot.log"
BOT_FILE = "bot_bybit.py"
BOT_STATUS_FILE = "bot_status.flag"
PAUSE_STATUS_FILE = "pause_new_trades.flag"

API_KEY = "API KEY"
API_SECRET = "API KEY"
USE_TESTNET = True

app_state = {
    'trades_df': pd.DataFrame(),
    'open_positions': [],
    'pnl_liquido_total': 0.0,
    'total_trades': 0,
    'trades_hoje': 0,
    'pnl_hoje': 0.0,
    'saldo_estimado': 100.0,
}

try:
    bybit_session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=USE_TESTNET)
    bybit_session.get_server_time()
    ui.notify('Sessão Bybit conectada com sucesso!', type='positive')
except Exception as e:
    bybit_session = None
    ui.notify(f'Falha ao conectar com a Bybit: {e}', type='negative')

REFRESH_SECONDS = 3

def carregar_dados_trades_do_db(limit=1000):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}", conn,
                                   parse_dates=['timestamp'])
        except pd.io.sql.DatabaseError:
            return pd.DataFrame()

    cols_essenciais = ['pnl_liquido', 'pnl_bruto', 'taxas', 'preco_entrada_executado', 'preco_saida', 'qtd', 'tp', 'sl',
                       'estrategia', 'justificativa', 'resultado', 'par', 'acao', 'timestamp', 'id']
    for col in cols_essenciais:
        if col not in df.columns:
            if df.empty:
                df[col] = pd.Series(dtype='object')
            else:
                df[col] = '' if col in ['estrategia', 'justificativa', 'resultado', 'par', 'acao'] else 0.0

    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df


def ler_log_bot(n_lines=200):
    if not os.path.exists(LOG_FILE):
        return "Arquivo 'bot.log' não encontrado."
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-n_lines:]
            return "".join(lines)
    except Exception as e:
        return f"Erro ao ler logs: {e}"


def bot_running():
    return os.path.exists(BOT_STATUS_FILE)


def bot_paused():
    return os.path.exists(PAUSE_STATUS_FILE)


def start_bot_flag():
    open(BOT_STATUS_FILE, 'w').write("run")
    ui.notify('Sinal de START enviado ao bot.', type='positive')


def stop_bot_flag():
    try:
        if os.path.exists(BOT_STATUS_FILE): os.remove(BOT_STATUS_FILE)
        if os.path.exists(PAUSE_STATUS_FILE): os.remove(PAUSE_STATUS_FILE)
        ui.notify('Sinal de STOP enviado ao bot.', color='negative')
    except Exception as e:
        ui.notify(f'Erro ao parar bot: {e}', type='negative')


def pause_entries_flag():
    open(PAUSE_STATUS_FILE, 'w').write("pause")
    ui.notify('Sinal de PAUSAR NOVAS ENTRADAS enviado.', color='warning')


def resume_entries_flag():
    try:
        if os.path.exists(PAUSE_STATUS_FILE): os.remove(PAUSE_STATUS_FILE)
        ui.notify('Sinal de RESUMIR ENTRADAS enviado.', type='positive')
    except Exception as e:
        ui.notify(f'Erro ao resumir entradas: {e}', type='negative')


def fechar_posicao_market(par, qtd):
    if not bybit_session:
        raise RuntimeError("Sessão Bybit não está inicializada no painel.")

    side_para_fechar = None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT acao FROM trades WHERE par = ? AND resultado = 'aberto' ORDER BY id DESC LIMIT 1",
                       (par,))
        resultado = cursor.fetchone()
        if not resultado:
            raise ValueError(f"Nenhuma posição aberta encontrada no DB para o par {par}.")

        acao_original = resultado[0].lower()
        if acao_original == 'buy':
            side_para_fechar = "Sell"
        elif acao_original == 'sell':
            side_para_fechar = "Buy"
        else:
            raise ValueError(f"Ação desconhecida ('{acao_original}') no DB para {par}.")

    try:
        resp = bybit_session.place_order(
            category="linear", symbol=par,
            side=side_para_fechar, order_type="Market",
            qty=str(qtd), reduce_only=True
        )
        return resp
    except Exception as e:
        raise


def atualizar_estado_app():
    df = carregar_dados_trades_do_db(limit=2000)

    if 'pnl_liquido' in df.columns:
        df['pnl_liquido'] = pd.to_numeric(df['pnl_liquido'], errors='coerce')

    app_state['trades_df'] = df
    if not df.empty:
        app_state['pnl_liquido_total'] = df['pnl_liquido'].fillna(0).sum()
        app_state['total_trades'] = len(df)
        hoje = datetime.now().date()
        df_timestamp_safe = pd.to_datetime(df['timestamp'], errors='coerce')
        df_hoje = df[df_timestamp_safe.dt.date == hoje]

        app_state['trades_hoje'] = len(df_hoje)
        app_state['pnl_hoje'] = df_hoje['pnl_liquido'].fillna(0).sum()
        capital_inicial = 100.0
        app_state['saldo_estimado'] = capital_inicial + app_state['pnl_liquido_total']
        abertas = df[df['resultado'] == 'aberto'].copy()
        posicoes = []
        if not abertas.empty and bybit_session:
            try:
                tickers = bybit_session.get_tickers(category="linear")["result"]["list"]
                precos = {t['symbol']: float(t.get("lastPrice", 0)) for t in tickers}
            except Exception:
                precos = {}
            for _, r in abertas.iterrows():
                pos = r.to_dict()
                preco_atual = precos.get(r['par'], 0)
                pnl_flutuante = 0.0
                try:
                    entrada = float(r.get('preco_entrada_executado', 0))
                    qtd = float(r.get('qtd', 0))
                    if entrada > 0 and qtd > 0 and preco_atual > 0:
                        if r['acao'].lower() == 'buy':
                            pnl_flutuante = (preco_atual - entrada) * qtd
                        else:
                            pnl_flutuante = (entrada - preco_atual) * qtd
                except (ValueError, TypeError):
                    pnl_flutuante = 0.0
                pos['pnl_flutuante'] = pnl_flutuante
                posicoes.append(pos)
        app_state['open_positions'] = posicoes


ui.colors(primary='#ff9800', secondary='#607d8b', accent='#ff5722')

with ui.header().classes('items-center justify-between'):
    ui.label('🔥 Painel Bybit — Bot Multi-Estratégia').classes('text-h6')
    with ui.row().classes('items-center'):
        ui.label().bind_text_from(app.storage, 'bot_status_label').classes('text-sm')
        ui.label().bind_text_from(app.storage, 'pause_status_label').classes('text-sm')

    with ui.row():
        ui.button('Start', on_click=start_bot_flag)
        ui.button('Stop', color='negative', on_click=stop_bot_flag)
        ui.button('Pause Entradas', color='warning', on_click=pause_entries_flag)
        ui.button('Resume Entradas', color='primary', on_click=resume_entries_flag)


def update_status_labels():
    app.storage.bot_status_label = "🟢 ATIVO" if bot_running() else "🔴 PARADO"
    app.storage.pause_status_label = "⏸️ Entradas PAUSADAS" if bot_paused() else "▶️ Entradas ATIVAS"


with ui.tabs().classes('w-full') as tabs:
    ui.tab('Dashboard')
    ui.tab('Posições (DB)')
    ui.tab('Gestão Manual (API)')
    ui.tab('Histórico')
    ui.tab('Charts')
    ui.tab('Console')
    ui.tab('Config & Logs')

with ui.tab_panels(tabs, value='Dashboard').classes('w-full'):
    with ui.tab_panel('Dashboard'):
        with ui.row().classes('w-full items-start'):
            with ui.column().classes('w-2/3 pr-4'):
                ui.label('Resumo de Performance').classes('text-h6')
                with ui.card().classes('w-full'):
                    with ui.row():
                        ui.label().bind_text(lambda: f"PnL Líquido Total: ${app_state['pnl_liquido_total']:.2f} USDT")
                        ui.label().bind_text(lambda: f"Trades Registrados: {app_state['total_trades']}")
                ui.separator().classes('my-4')
                ui.label('PnL Diário').classes('text-subtitle1')
                fig_pnl = go.Figure()
                fig_pnl.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=40, b=10))
                pnl_chart = ui.plotly(fig_pnl)
            with ui.column().classes('w-1/3'):
                ui.label('KPIs Rápidos').classes('text-h6')
                with ui.card().classes('w-full'):
                    ui.label().bind_text(lambda: f"Trades Hoje: {app_state['trades_hoje']}")
                    ui.label().bind_text(lambda: f"PnL Hoje: ${app_state['pnl_hoje']:.2f}")
                    ui.label().bind_text(lambda: f"Saldo Estimado: ${app_state['saldo_estimado']:.2f}")


        def atualizar_pnl_chart():
            df = app_state['trades_df']
            if df.empty or 'resultado' not in df.columns: return
            df2 = df[df['resultado'].isin(['WIN', 'LOSS'])].copy()
            if df2.empty: return

            df2['dia'] = pd.to_datetime(df2['timestamp'], errors='coerce').dt.date
            df2.dropna(subset=['dia'], inplace=True)

            daily = df2.groupby('dia')['pnl_liquido'].sum().reset_index()
            pnl_chart.figure.data = []
            pnl_chart.figure.add_trace(go.Bar(x=daily['dia'].astype(str), y=daily['pnl_liquido'], name='PnL Diário'))
            pnl_chart.figure.update_layout(title='Performance Diária (Líquida)', height=350)
            pnl_chart.update()


        ui.timer(10.0, atualizar_pnl_chart, active=True)

    with ui.tab_panel('Posições (DB)'):
        ui.label('Monitor de Posições (Registradas no Banco de Dados)').classes('text-h6')
        with ui.row().classes('items-center gap-2'):
            par_input = ui.input('Par p/ fechar (ex: BTCUSDT)').style('width:200px')
            qtd_input = ui.number('Qtd', format='%.6f').style('width:150px')
            fechar_btn = ui.button('Fechar Market', color='negative')
        pos_info = ui.label()


        def fechar_acao():
            par = par_input.value;
            qtd = qtd_input.value
            if not par or not qtd or qtd <= 0:
                pos_info.set_text('ERRO: Informe o par e a quantidade válida.').classes('text-red-500');
                return
            try:
                with fechar_btn.add_slot('loading'):
                    ui.spinner(size='sm').classes('mr-2')
                resp = fechar_posicao_market(par.strip().upper(), float(qtd))
                if resp.get('retCode') == 0 and resp.get('result', {}).get('orderId'):
                    pos_info.set_text(f"SUCESSO: Ordem enviada (ID: {resp['result']['orderId']}).").classes(
                        'text-green-500')
                    time.sleep(REFRESH_SECONDS)
                    update_pos_table()
                else:
                    raise RuntimeError(resp.get('retMsg', 'Erro desconhecido da API.'))
            except Exception as e:
                pos_info.set_text(f'FALHA: {e}').classes('text-red-500')


        fechar_btn.on('click', fechar_acao)

        columns_pos = [{'name': 'id', 'label': 'ID', 'field': 'id'}, {'name': 'par', 'label': 'Par', 'field': 'par'},
                       {'name': 'acao', 'label': 'Ação', 'field': 'acao'},
                       {'name': 'preco_entrada_executado', 'label': 'Entrada', 'field': 'preco_entrada_executado'},
                       {'name': 'qtd', 'label': 'Qtd', 'field': 'qtd'}, {'name': 'tp', 'label': 'TP', 'field': 'tp'},
                       {'name': 'sl', 'label': 'SL', 'field': 'sl'},
                       {'name': 'pnl_flutuante', 'label': 'PnL (Flut.)', 'field': 'pnl_flutuante'}]
        pos_table = ui.table(columns=columns_pos, rows=[], row_key='id').classes('w-full')


        def update_pos_table():
            rows = app_state['open_positions']
            formatted_rows = [row.copy() for row in rows]
            for row in formatted_rows:
                row['preco_entrada_executado'] = f"{float(row.get('preco_entrada_executado', 0)):.4f}"
                row['tp'] = f"{float(row.get('tp', 0)):.4f}";
                row['sl'] = f"{float(row.get('sl', 0)):.4f}"
                pnl_val = row.get('pnl_flutuante', 0.0)
                row['pnl_flutuante'] = f'${pnl_val:,.4f}'
            pos_table.update_rows(formatted_rows)


        ui.timer(REFRESH_SECONDS, update_pos_table, active=True)


        with ui.tab_panel('Gestão Manual (API)'):
            ui.label('Gerenciar Posições Abertas (Direto da Bybit)').classes('text-h6')
            ui.markdown(
                'Use esta aba para ver a situação real da sua conta e fechar posições que o bot possa não estar rastreando.')

            with ui.dialog() as dialog, ui.card():
                ui.label('Análise Detalhada da Operação').classes('text-h6')
                info_par = ui.markdown()
                info_valor_total = ui.markdown()
                info_situacao_atual = ui.markdown()
                ui.button('Fechar', on_click=dialog.close)


            def carregar_posicoes_da_api():
                if not bybit_session: return []
                try:
                    response = bybit_session.get_positions(category="linear", settleCoin="USDT")
                    positions_raw = response.get("result", {}).get("list", [])
                    return [p for p in positions_raw if float(p.get('size', 0)) > 0]
                except Exception as e:
                    ui.notify(f"Erro ao buscar posições da API: {e}", type='negative')
                    return []


            async def fechar_posicao_api(symbol: str, size: str, side: str):
                side_para_fechar = "Sell" if side == "Buy" else "Buy"
                try:
                    ui.notify(f"Enviando ordem a mercado para fechar {size} {symbol}...", color='info')
                    resp = bybit_session.place_order(
                        category="linear", symbol=symbol, side=side_para_fechar,
                        order_type="Market", qty=str(size), reduce_only=True
                    )
                    if resp.get('retCode') == 0:
                        ui.notify(f"SUCESSO: Ordem de fechamento para {symbol} enviada!", type='positive')
                        await asyncio.sleep(2)
                        update_manual_pos_table.refresh()
                    else:
                        raise RuntimeError(resp.get('retMsg', 'Erro da API.'))
                except Exception as e:
                    ui.notify(f"FALHA ao fechar {symbol}: {e}", type='negative')


            def mostrar_detalhes(e):
                row = e.args
                try:
                    lado = row['side']
                    tamanho = float(row['size'])
                    preco_medio = float(row['avgPrice'])
                    pnl = float(row['unrealisedPnl'])
                    valor_total = tamanho * preco_medio

                    acao_verbo = "comprou" if lado == "Buy" else "vendeu (apostou na baixa)"
                    objetivo = "subir" if lado == "Buy" else "cair"
                    resultado_texto = f"lucro de ${pnl:,.2f}" if pnl > 0 else f"prejuízo de ${-pnl:,.2f}"

                    info_par.set_content(f"**Par:** `{row['symbol']}`")
                    info_valor_total.set_content(
                        f"Você **{acao_verbo} `{tamanho}` moedas** a um preço médio de **`${preco_medio:,.4f}`**. O valor total da sua entrada foi de **`${valor_total:,.2f}`**.")
                    info_situacao_atual.set_content(
                        f"Sua aposta era que o preço fosse **{objetivo}**. Atualmente, você está com um **{resultado_texto}** flutuante.")

                    dialog.open()
                except (ValueError, KeyError) as err:
                    ui.notify(f"Não foi possível gerar os detalhes. Dados incompletos. Erro: {err}", type='negative')


            ui.button('Atualizar Lista da API', on_click=lambda: update_manual_pos_table.refresh())

            columns_manual = [
                {'name': 'symbol', 'label': 'Par', 'field': 'symbol', 'sortable': True},
                {'name': 'side', 'label': 'Lado', 'field': 'side'},
                {'name': 'size', 'label': 'Tamanho', 'field': 'size'},
                {'name': 'avgPrice', 'label': 'Preço Médio', 'field': 'avgPrice'},
                {'name': 'unrealisedPnl', 'label': 'PnL Flutuante', 'field': 'unrealisedPnl_fmt'},
                {'name': 'actions', 'label': 'Ações'},
            ]
            manual_pos_table = ui.table(columns=columns_manual, rows=[], row_key='symbol').classes('w-full')

            manual_pos_table.add_slot('body-cell-actions', '''
                <q-td :props="props">
                    <q-btn @click="$parent.$emit('details', props.row)" color="primary" dense class="q-mr-sm">Detalhes</q-btn>
                    <q-btn @click="$parent.$emit('close', props.row)" color="negative" dense>Fechar a Mercado</q-btn>
                </q-td>
            ''')

            manual_pos_table.on('close', lambda e: fechar_posicao_api(e.args['symbol'], e.args['size'], e.args['side']))
            manual_pos_table.on('details', mostrar_detalhes)


            @ui.refreshable
            def update_manual_pos_table():
                posicoes = carregar_posicoes_da_api()
                for pos in posicoes:
                    pnl_float = float(pos.get('unrealisedPnl', 0))
                    pos['unrealisedPnl_fmt'] = f"${pnl_float:,.4f}"
                    pos['unrealisedPnl'] = pnl_float
                manual_pos_table.update_rows(posicoes)


            ui.timer(30.0, update_manual_pos_table, active=True)


    with ui.tab_panel('Histórico'):
        ui.label('Histórico de Trades').classes('text-h6')
        hist_table = ui.table(columns=[], rows=[], row_key='id', pagination=15).classes('w-full')


        def atualizar_historico():
            df = app_state['trades_df']
            if df.empty:
                hist_table.update_rows([])
                return

            display_df = df.copy()

            for col in ['preco_entrada_executado', 'preco_saida', 'pnl_liquido', 'taxas', 'qtd']:
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0)

            display_df['acao_desc'] = display_df.apply(
                lambda row: f"COMPRA de {row['qtd']}" if row['acao'].lower() == 'buy' else f"VENDA de {row['qtd']}",
                axis=1
            )
            display_df['preco_entrada_fmt'] = display_df['preco_entrada_executado'].apply(lambda x: f"${x:,.4f}")
            display_df['preco_saida_fmt'] = display_df.apply(
                lambda row: f"${row['preco_saida']:,.4f}" if row['resultado'] in ['WIN', 'LOSS'] else "Em aberto",
                axis=1
            )
            display_df['resultado_fmt'] = display_df.apply(
                lambda row: f"✅ Lucro de ${row['pnl_liquido']:,.2f}" if row['resultado'] == 'WIN'
                else (f"❌ Prejuízo de ${row['pnl_liquido']:,.2f}" if row['resultado'] == 'LOSS'
                      else row['resultado'].capitalize()),
                axis=1
            )
            display_df['taxas_fmt'] = display_df['taxas'].apply(lambda x: f"${x:,.4f}")

            columns_hist_didatico = [
                {'name': 'timestamp', 'label': 'Data Abertura', 'field': 'timestamp', 'sortable': True},
                {'name': 'par', 'label': 'Par', 'field': 'par', 'sortable': True},
                {'name': 'acao_desc', 'label': 'Operação', 'field': 'acao_desc'},
                {'name': 'preco_entrada_fmt', 'label': 'Preço Entrada', 'field': 'preco_entrada_fmt'},
                {'name': 'preco_saida_fmt', 'label': 'Preço Saída', 'field': 'preco_saida_fmt'},
                {'name': 'resultado_fmt', 'label': 'Resultado Líquido', 'field': 'resultado_fmt', 'sortable': True},
                {'name': 'taxas_fmt', 'label': 'Taxas', 'field': 'taxas_fmt'},
                {'name': 'justificativa', 'label': 'Justificativa', 'field': 'justificativa',
                 'style': 'max-width: 250px; white-space: normal;'}
            ]

            hist_table.columns = columns_hist_didatico
            hist_table.update_rows(display_df.to_dict('records'))


        ui.timer(5.0, atualizar_historico, active=True)

    with ui.tab_panel('Charts'):
        ui.label('Gráficos Interativos').classes('text-h6')
        with ui.row():
            par_select = ui.select(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'], value='BTCUSDT')
            intervalo = ui.select(['1', '5', '15', '60', '240'], value='60')
            ui.button('Atualizar Gráfico', on_click=lambda: atualizar_candle(par_select.value, intervalo.value))
        fig_candle = go.Figure()
        fig_candle.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=40, b=10))
        candle_plot = ui.plotly(fig_candle)


        def atualizar_candle(par, interval_val):
            if not bybit_session: ui.notify("Sessão Bybit não disponível.", type='warning'); return
            try:
                resp = bybit_session.get_kline(category="linear", symbol=par, interval=interval_val, limit=200)
                dados = resp.get("result", {}).get("list", [])
                if not dados: ui.notify(f"Nenhum dado de candle para {par}.", type='warning'); return
                df = pd.DataFrame(dados, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
                candle_plot.figure.data = []
                candle_plot.figure.add_trace(
                    go.Candlestick(x=df['timestamp'], open=df['open'].astype(float), high=df['high'].astype(float),
                                   low=df['low'].astype(float), close=df['close'].astype(float)))
                candle_plot.figure.update_layout(title=f'Gráfico de {par} - {interval_val}min', height=450)
                candle_plot.update()
            except Exception as e:
                ui.notify(f'Erro ao buscar gráfico: {e}', type='negative')


        ui.timer(0.1, lambda: atualizar_candle(par_select.value, intervalo.value), once=True)

    with ui.tab_panel('Console'):
        ui.label('Logs em tempo real').classes('text-h6')
        log_area = ui.textarea().props('readonly outlined').classes('w-full h-screen')


        def atualizar_logs(): log_area.set_value(ler_log_bot(500))


        ui.timer(2.0, atualizar_logs, active=True)

    with ui.tab_panel('Config & Logs'):
        ui.label('Arquivos e Configurações').classes('text-h6')
        ui.markdown(f"- **Banco de Dados:** `{os.path.abspath(DB_PATH)}`")
        ui.markdown(f"- **Arquivo de Log:** `{os.path.abspath(LOG_FILE)}`")
        ui.markdown(f"- **Script do Bot:** `{os.path.abspath(BOT_FILE)}`")
        if os.name == 'nt':
            ui.button('Abrir pasta do projeto', on_click=lambda: os.startfile(os.getcwd()))

ui.timer(REFRESH_SECONDS, atualizar_estado_app, active=True)
ui.timer(1.0, update_status_labels, active=True)

ui.run(title='Painel Bybit - NiceGUI', host='0.0.0.0', port=8080, reload=False,
       storage_secret='uma_chave_secreta_para_o_storage')