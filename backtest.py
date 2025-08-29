# -*- coding: utf-8 -*-
import os
import math
import yaml
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from feature_engineering import criar_features
import traceback

# -----------------------
# Config
# -----------------------
DEFAULT_CONFIG = {
    'risk_perc': 0.01,
    'ml_thresholds': {'BTCUSDT': 0.65, 'ETHUSDT': 0.65},
    'min_volatility_perc': 0.15,
    'adx_threshold': 20,
    'volume_ma_window': 20,
    'taker_fee': 0.00055,
    'maker_fee': 0.00020,
    'slippage_perc': 0.0005,
    'cooldown_minutes': 120,
    'post_loss_cooldown_minutes': 360,
    'rr_ratio': 1.7,
    'atr_sl_multiplier': 1.5,
    'funding_column_names': ['funding_rate', 'funding'],
    'start_date': '2023-01-01',
    'output_dir': 'research_results_confluence',
    'walk_forward': {'enabled': False}
}

# Carrega config.yaml
CONFIG = DEFAULT_CONFIG.copy()
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        user_cfg = yaml.safe_load(f) or {}
        CONFIG.update(user_cfg)
    print("✅ config.yaml carregado.")
except FileNotFoundError:
    print("⚠️ config.yaml não encontrado — usando defaults embutidos.")
except Exception as e:
    print(f"⚠️ Erro ao ler config.yaml: {e}. Usando defaults.")

# Carrega as configurações em variáveis globais
RISK_PERC = CONFIG.get('risk_perc', 0.01)
ML_THRESHOLDS = CONFIG.get('ml_thresholds', DEFAULT_CONFIG['ml_thresholds'])
MIN_VOL_PERC = CONFIG.get('min_volatility_perc', 0.15)
ADX_THRESHOLD = CONFIG.get('adx_threshold', 20)
TAKER_FEE = CONFIG.get('taker_fee', 0.00055)
MAKER_FEE = CONFIG.get('maker_fee', 0.00020)
SLIPPAGE_PERC = CONFIG.get('slippage_perc', 0.0005)
COOLDOWN_MIN = CONFIG.get('cooldown_minutes', 120)
POST_LOSS_COOLDOWN_MIN = CONFIG.get('post_loss_cooldown_minutes', 360)
RR_RATIO = CONFIG.get('rr_ratio', 1.7)
ATR_SL_MULT = CONFIG.get('atr_sl_multiplier', 1.5)
OUTPUT_DIR_BASE = CONFIG.get('output_dir', 'research_results_confluence')
START_DATE = CONFIG.get('start_date', '2023-01-01')
PARES_PARA_TESTAR = list(ML_THRESHOLDS.keys())
np.random.seed(42)


# -----------------------
# Helpers
# -----------------------
def load_model_artifact(par: str):
    model_file = f"model_{par}.pkl"
    if not os.path.exists(model_file):
        return None
    try:
        artefato = joblib.load(model_file)
        if not isinstance(artefato, dict) or 'model' not in artefato:
            print(f"⚠️ Artefato {model_file} carregado mas formato inesperado.")
            return None
        return artefato
    except Exception as e:
        print(f"⚠️ Falha ao carregar artefato {model_file}: {e}")
        return None


def simulate_execution(entry_price_ideal, exit_price_ideal, side='long'):
    if side == 'long':
        executed_entry = entry_price_ideal * (1 + SLIPPAGE_PERC)
        executed_exit = exit_price_ideal * (1 - SLIPPAGE_PERC)
    else:
        executed_entry = entry_price_ideal * (1 - SLIPPAGE_PERC)
        executed_exit = exit_price_ideal * (1 + SLIPPAGE_PERC)
    return executed_entry, executed_exit


# =================================================================================
# === SEÇÃO CORRIGIDA E OTIMIZADA =================================================
# =================================================================================
def regime_allows_trade(row: pd.Series, adx_threshold: float, min_volatility_perc: float) -> (bool, str):
    last_adx = row.get('adx', 0)
    vol_perc = row.get('normalized_volatility', 0.0) * 100
    last_vol = row.get('volume', 0)
    last_vol_ma = row.get('volume_ma_20', 0)

    # Validações (agora muito mais simples e rápidas)
    if last_adx < adx_threshold:
        return False, f"ADX baixo ({last_adx:.2f} < {adx_threshold}) - mercado lateral."
    if vol_perc < min_volatility_perc:
        return False, f"Volatilidade baixa (ATR%={vol_perc:.3f}% < {min_volatility_perc}%)."
    if last_vol_ma > 0 and last_vol < 0.5 * last_vol_ma:
        volume_ma_window = CONFIG.get('volume_ma_window', 20)
        return False, f"Volume baixo (last {last_vol:.0f} < 0.5 * MA{volume_ma_window}={last_vol_ma:.0f})."

    return True, f"Regime ok (ADX={last_adx:.2f}, ATR%={vol_perc:.3f}%, vol={last_vol:.0f})."


# -----------------------
# Main backtest por par
# -----------------------
def run_backtest(par: str):
    print("\n" + "#" * 80)
    print(f"🔬 EXECUTANDO BACKTEST (ML-PRIMÁRIO): Ativo = {par}")
    print("#" * 80)

    output_dir = os.path.join(OUTPUT_DIR_BASE, par)
    os.makedirs(output_dir, exist_ok=True)

    arquivo = os.path.join("data", f"dados_historicos_{par}_5m.csv")
    if not os.path.exists(arquivo):
        print(f"❌ Arquivo de dados {arquivo} não encontrado. Pulei {par}.")
        return

    df_raw = pd.read_csv(arquivo, parse_dates=['timestamp'])
    df_raw = df_raw[df_raw['timestamp'] >= pd.to_datetime(START_DATE)].reset_index(drop=True)

    if df_raw.empty:
        print(f"❌ Sem candles após {START_DATE} para {par}.")
        return

    funding_col = next((c for c in CONFIG['funding_column_names'] if c in df_raw.columns), None)
    if funding_col:
        print(f"ℹ️ Modelo de funding usará a coluna '{funding_col}' do CSV.")
    else:
        print("⚠️ Coluna de funding não encontrada no CSV. Funding será assumido como zero.")

    df_feat = criar_features(df_raw.copy(), drop_na=True)
    print(f"📊 Candles preparados: {len(df_feat)}")

    artefato = load_model_artifact(par)
    model, features_list, classes = None, None, None
    if artefato:
        model = artefato.get('model')
        features_list = artefato.get('features_list')
        classes = artefato.get('classes')
        print(f"✅ Artefato ML carregado para {par}.")
    else:
        print("⚠️ Modelo ML não encontrado — fallback para QQE-RSI (regra técnica).")

    capital = 1000.0
    equity_curve = [capital]
    trades = []
    cooldown_until = None

    for i in range(1, len(df_feat)):
        row = df_feat.iloc[i]
        timestamp = row['timestamp']

        if cooldown_until and timestamp < cooldown_until:
            equity_curve.append(capital)
            continue

        allows, reason = regime_allows_trade(
            row,
            adx_threshold=ADX_THRESHOLD,
            min_volatility_perc=MIN_VOL_PERC
        )
        if not allows:
            equity_curve.append(capital)
            continue

        signal, tp, sl, justification = None, None, None, ""
        entry_price = row['close']

        if model and features_list:
            try:
                X = row[features_list].values.reshape(1, -1)
                probs = model.predict_proba(X)[0]

                prob_map = {int(c): float(p) for c, p in zip(classes, probs)} if classes else {}
                prob_alta = prob_map.get(1, 0.0)
                prob_baixa = prob_map.get(0, 0.0)

            except Exception as e:
                print(f"Erro na predição do modelo no timestamp {timestamp}: {e}")
                prob_alta, prob_baixa = 0.0, 0.0

            threshold = ML_THRESHOLDS.get(par, 0.65)
            if prob_alta >= threshold:
                signal = 'buy'
                justification = f"ML primário (p={prob_alta:.3f})"
            elif prob_baixa >= threshold:
                signal = 'sell'
                justification = f"ML primário (p={prob_baixa:.3f})"
            else:
                equity_curve.append(capital)
                continue

            last_atr = row.get('atr')
            if pd.isna(last_atr) or last_atr <= 0:
                equity_curve.append(capital)
                continue

            dist_sl = ATR_SL_MULT * last_atr
            sl = entry_price - dist_sl if signal == 'buy' else entry_price + dist_sl
            tp = entry_price + dist_sl * RR_RATIO if signal == 'buy' else entry_price - dist_sl * RR_RATIO

        else:
            prev_row = df_feat.iloc[i - 1]
            qqe_cross_up = row['qqe_line'] > row['qqe_signal'] and prev_row['qqe_line'] <= prev_row['qqe_signal']
            qqe_cross_down = row['qqe_line'] < row['qqe_signal'] and prev_row['qqe_line'] >= prev_row['qqe_signal']

            if qqe_cross_up and row['rsi'] > 50:
                signal = 'buy'
            elif qqe_cross_down and row['rsi'] < 50:
                signal = 'sell'
            else:
                equity_curve.append(capital)
                continue

            dist_sl = ATR_SL_MULT * row['atr']
            sl = entry_price - dist_sl if signal == 'buy' else entry_price + dist_sl
            tp = entry_price + dist_sl * RR_RATIO if signal == 'buy' else entry_price - dist_sl * RR_RATIO
            justification = "QQE-RSI confluência (fallback)"

        if signal is None:
            equity_curve.append(capital)
            continue

        distancia_sl = abs(entry_price - sl)
        if distancia_sl <= 1e-12:
            equity_curve.append(capital)
            continue

        risk_amount = capital * RISK_PERC
        qty = risk_amount / distancia_sl

        notional = qty * entry_price
        max_notional = capital * CONFIG.get('max_notional_perc', 0.08)
        if notional > max_notional:
            qty *= (max_notional / notional)
            notional = qty * entry_price

        executed_entry, _ = simulate_execution(entry_price, tp, side='long' if signal == 'buy' else 'short')

        open_pos = {
            'par': par, 'direcao': signal, 'preco_entrada_ideal': entry_price,
            'preco_entrada_executado': executed_entry, 'tp': tp, 'sl': sl,
            'qtd': qty, 'timestamp_entrada': timestamp, 'justificativa': justification
        }

        closed = False
        funding_total = 0.0

        for j in range(i + 1, len(df_feat)):
            fut = df_feat.iloc[j]
            hit_tp, hit_sl = False, False
            exit_price_ideal = 0.0

            if signal == 'buy':
                if fut['high'] >= open_pos['tp']:
                    exit_price_ideal, hit_tp = open_pos['tp'], True
                elif fut['low'] <= open_pos['sl']:
                    exit_price_ideal, hit_sl = open_pos['sl'], True
            else:  # sell
                if fut['low'] <= open_pos['tp']:
                    exit_price_ideal, hit_tp = open_pos['tp'], True
                elif fut['high'] >= open_pos['sl']:
                    exit_price_ideal, hit_sl = open_pos['sl'], True

            if funding_col:
                fr = float(fut.get(funding_col, 0.0) or 0.0)
                funding_total += notional * fr * (1 if signal == 'buy' else -1)

            if hit_tp or hit_sl:
                _, executed_exit = simulate_execution(open_pos['preco_entrada_executado'], exit_price_ideal,
                                                      side=open_pos['direcao'])

                pnl_grosso = (executed_exit - open_pos['preco_entrada_executado']) * qty if signal == 'buy' else (
                                                                                                                             open_pos[
                                                                                                                                 'preco_entrada_executado'] - executed_exit) * qty
                fees = (open_pos['preco_entrada_executado'] * qty + executed_exit * qty) * TAKER_FEE
                pnl_liq = pnl_grosso - fees - funding_total  # Funding é um custo, então subtrai

                trades.append({
                    'par': par, 'timestamp_entrada': open_pos['timestamp_entrada'], 'timestamp_saida': fut['timestamp'],
                    'direcao': signal, 'preco_entrada_ideal': open_pos['preco_entrada_ideal'],
                    'preco_entrada_executado': open_pos['preco_entrada_executado'],
                    'preco_saida_ideal': exit_price_ideal, 'preco_saida_executado': executed_exit,
                    'tp': open_pos['tp'], 'sl': open_pos['sl'], 'qtd': qty, 'notional': notional,
                    'pnl_bruto': pnl_grosso, 'fees': fees, 'funding': funding_total,
                    'pnl_liquido': pnl_liq, 'resultado': 'WIN' if pnl_liq > 0 else 'LOSS',
                    'justificativa': justification
                })

                capital += pnl_liq
                cooldown_duration = POST_LOSS_COOLDOWN_MIN if pnl_liq <= 0 else COOLDOWN_MIN
                cooldown_until = fut['timestamp'] + pd.Timedelta(minutes=cooldown_duration)
                closed = True
                break

        if not closed:
            trades.append({**open_pos, 'resultado': 'TIMEOUT', 'pnl_liquido': 0.0})

        equity_curve.append(capital)

    gerar_relatorio_e_grafico(par, trades, equity_curve, initial_capital=1000.0, output_path=output_dir)


# -----------------------
# Relatórios e métricas (sem alterações)
# -----------------------
def gerar_relatorio_e_grafico(par, trades, equity_curve, initial_capital, output_path):
    os.makedirs(output_path, exist_ok=True)
    report_lines = [f" RELATÓRIO DE PERFORMANCE: {par} ".center(80, "=")]

    if not trades:
        report_lines.append("\nNenhuma operação realizada.")
    else:
        df_trades = pd.DataFrame(trades).dropna(subset=['pnl_liquido'])
        total_trades = len(df_trades)
        if total_trades == 0:
            report_lines.append("\nNenhuma operação concluída (pode haver timeouts).")
        else:
            wins = df_trades[df_trades['pnl_liquido'] > 0]
            losses = df_trades[df_trades['pnl_liquido'] <= 0]
            win_rate = len(wins) / total_trades * 100
            pnl_total = df_trades['pnl_liquido'].sum()

            capital_series = pd.Series(equity_curve).ffill()
            retorno_total_pct = (capital_series.iloc[-1] / initial_capital - 1) * 100

            gross_wins = wins['pnl_liquido'].sum()
            gross_losses = abs(losses['pnl_liquido'].sum())
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

            avg_win = wins['pnl_liquido'].mean() if not wins.empty else 0.0
            avg_loss = abs(losses['pnl_liquido'].mean()) if not losses.empty else 0.0
            payoff = avg_win / avg_loss if avg_loss > 0 else float('inf')

            peak = capital_series.cummax()
            dd = (capital_series - peak) / peak
            max_dd = abs(dd.min()) * 100

            rets = capital_series.pct_change().dropna()
            sharpe = (rets.mean() / rets.std()) * math.sqrt(288 * 252) if rets.std() > 0 else 0.0

            report_lines.extend([
                f"Capital Final.............: ${capital_series.iloc[-1]:,.2f}",
                f"PnL Líquido Total.........: ${pnl_total:,.2f} ({retorno_total_pct:.2f}%)",
                f"Total de Operações........: {total_trades}",
                f"Taxa de Acerto............: {win_rate:.2f}%",
                f"Fator de Lucro............: {profit_factor:.2f}",
                f"Payoff Ratio..............: {payoff:.2f}",
                f"Drawdown Máximo...........: {max_dd:.2f}%",
                f"Sharpe Ratio (Anualizado).: {sharpe:.2f}",
            ])

            df_trades.to_csv(os.path.join(output_path, "trades_history.csv"), index=False)

            fig = go.Figure(
                data=go.Scatter(x=list(range(len(capital_series))), y=capital_series, mode='lines', name='Capital'))
            fig.update_layout(title=f'Curva de Capital - {par}', template='plotly_dark')
            fig.write_html(os.path.join(output_path, "equity_curve.html"))

    full_report = "\n".join(report_lines)
    print(full_report)
    with open(os.path.join(output_path, "performance_report.txt"), "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"📄 Relatório e arquivos salvos em: {os.path.abspath(output_path)}")


# -----------------------
# Entrypoint
# -----------------------
if __name__ == '__main__':
    for par in PARES_PARA_TESTAR:
        try:
            run_backtest(par)
        except Exception as e:
            print(f"Erro fatal ao rodar backtest para {par}: {e}")
            traceback.print_exc()
    print("\nBacktests concluídos. Verifique a pasta:", OUTPUT_DIR_BASE)