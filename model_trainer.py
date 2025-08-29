# -*- coding: utf-8 -*-
import argparse
import os
from datetime import datetime, timezone
import platform
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
from feature_engineering import criar_features
from types import SimpleNamespace


ATR_LOOKBACK = 14
ATR_MULTIPLIER_TP = 5.0
ATR_MULTIPLIER_SL = 1.8
MAX_HOLD_PERIOD = 24

try:
    import xgboost as xgb

    XGBOOST_VERSION = xgb.__version__
    from packaging import version

    xgb_version = version.parse(XGBOOST_VERSION)
    SUPPORTS_CONSTRUCTOR_EARLY_STOPPING = xgb_version >= version.parse("2.0.0")
except ImportError:
    XGBOOST_VERSION = "unknown"
    SUPPORTS_CONSTRUCTOR_EARLY_STOPPING = False

try:
    import sklearn

    SKLEARN_VERSION = sklearn.__version__
except ImportError:
    SKLEARN_VERSION = "unknown"

PARES_PARA_TREINAR = ["BTCUSDT", "ETHUSDT"]


def criar_alvo_r_r(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out['alvo'] = np.nan

    atr_series = df_out['atr']
    high_series = df_out['high']
    low_series = df_out['low']
    close_series = df_out['close']

    for i in range(len(df_out) - MAX_HOLD_PERIOD):
        distancia_sl = atr_series.iloc[i] * ATR_MULTIPLIER_SL
        distancia_tp = atr_series.iloc[i] * ATR_MULTIPLIER_TP

        if distancia_sl == 0: continue

        sl_price_long = close_series.iloc[i] - distancia_sl
        tp_price_long = close_series.iloc[i] + distancia_tp

        sl_price_short = close_series.iloc[i] + distancia_sl
        tp_price_short = close_series.iloc[i] - distancia_tp

        atingiu_tp_long = False
        atingiu_sl_long = False
        atingiu_tp_short = False
        atingiu_sl_short = False

        for j in range(1, MAX_HOLD_PERIOD + 1):
            future_high = high_series.iloc[i + j]
            future_low = low_series.iloc[i + j]

            if not atingiu_tp_long and not atingiu_sl_long:
                if future_high >= tp_price_long: atingiu_tp_long = True
                if future_low <= sl_price_long: atingiu_sl_long = True

            if not atingiu_tp_short and not atingiu_sl_short:
                if future_low <= tp_price_short: atingiu_tp_short = True
                if future_high >= sl_price_short: atingiu_sl_short = True

            if atingiu_tp_long and not atingiu_sl_long:
                df_out.loc[df_out.index[i], 'alvo'] = 1
                break
            if atingiu_sl_long:
                if atingiu_tp_short and not atingiu_sl_short:
                    df_out.loc[df_out.index[i], 'alvo'] = 0
                break

    df_out = df_out.dropna(subset=['alvo']).reset_index(drop=True)
    df_out['alvo'] = df_out['alvo'].astype(int)
    return df_out


def treinar_um_modelo(par: str, args):
    arquivo_dados = os.path.join("data", f"dados_historicos_{par}_5m.csv")
    modelo_saida = f'model_{par}.pkl'
    modelo_temp = f'{modelo_saida}.tmp'

    print(f"\n--- Iniciando Treinamento para o par: {par} ---")
    print(f"XGBoost versão: {XGBOOST_VERSION}")

    if not os.path.exists(arquivo_dados):
        print(f"[AVISO] Arquivo de dados '{arquivo_dados}' não encontrado. Pulando.")
        return

    print(f"Carregando dados de: {arquivo_dados}")
    df_raw = pd.read_csv(arquivo_dados)

    df_features = criar_features(df_raw, drop_na=False)

    print("Criando alvo baseado em Risco/Retorno (isso pode levar alguns minutos)...")
    df_final = criar_alvo_r_r(df_features)
    print("... Alvo criado.")

    # --- LISTA DE FEATURES ---
    features = [
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'ema_9_slope', 'ema_21_slope',
        'adx',
        'rsi', 'macd', 'macd_signal', 'stoch_k',
        'return_1', 'return_3', 'return_5', 'return_15',
        'atr', 'bb_pband',
        'normalized_volatility',
        'close_pos_in_range',
        'range_to_atr',
        'volume_ma_20', 'volume_ratio',
        'rsi_trend_adjusted',
        'atr_acceleration',
        'volume_climax',
    ]

    for feature in features:
        if feature not in df_final.columns:
            print(f"[ERRO CRÍTICO] A feature '{feature}' não foi encontrada. Verifique 'feature_engineering.py'.")
            return

    X = df_final[features].values
    y = df_final['alvo'].values

    if len(X) < 500:
        print(f"[AVISO] Dados de alta qualidade insuficientes para treinar {par} ({len(X)} eventos). Pulando.")
        return

    split_index = int(len(df_final) * (1 - args.test_frac))
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    print(f"Total de eventos significativos: {len(df_final)}")
    print(f"Tamanho treino: {len(X_train)} | teste: {len(X_test)}")
    print(f"Distribuição do Alvo (Treino): \n{pd.Series(y_train).value_counts(normalize=True)}")

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    scale_pos_weight = neg / pos

    max_depth_eff = 4
    min_child_weight_eff = 1.0
    subsample_eff = 0.75
    colsample_eff = 0.75

    if par == "SOLUSDT":
        min_child_weight_eff = 1.5
        subsample_eff = 0.70
        colsample_eff = 0.70

    print("Iniciando treinamento com Early Stopping e balanceamento...")
    model = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', n_estimators=600,
        learning_rate=0.06, max_depth=max_depth_eff, subsample=subsample_eff,
        colsample_bytree=colsample_eff, min_child_weight=min_child_weight_eff,
        reg_lambda=1.0, scale_pos_weight=scale_pos_weight, random_state=42,
        verbosity=0, early_stopping_rounds=args.early_stop if SUPPORTS_CONSTRUCTOR_EARLY_STOPPING else None
    )

    fit_params = {'X': X_train, 'y': y_train, 'eval_set': [(X_test, y_test)], 'verbose': False}
    if not SUPPORTS_CONSTRUCTOR_EARLY_STOPPING:
        fit_params['early_stopping_rounds'] = args.early_stop
    model.fit(**fit_params)
    print("Treinamento concluído.")

    y_pred = model.predict(X_test)
    print(f"\n=== Avaliação para {par} (Classificação Binária) ===")
    print(classification_report(y_test, y_pred, target_names=['Baixa', 'Alta'], digits=4, zero_division=0))

    artefato = {
        'model': model, 'features_list': features,
        'feature_creation_function': criar_features, 'classes': model.classes_.tolist(),
        'training_timestamp': datetime.now(timezone.utc).isoformat(),
        'env': {'python': platform.python_version(), 'xgboost': XGBOOST_VERSION, 'sklearn': SKLEARN_VERSION}
    }

    try:
        joblib.dump(artefato, modelo_temp)
        os.rename(modelo_temp, modelo_saida)
        print(f"Modelo especialista para {par} salvo com sucesso em: {modelo_saida}")
    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha ao salvar o artefato do modelo para {par}: {e}")
        if os.path.exists(modelo_temp): os.remove(modelo_temp)


def obter_args_padrao():
    return SimpleNamespace(test_frac=0.2, early_stop=25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina modelos XGBoost especialistas para cada par.')
    parser.add_argument('--test-frac', default=0.2, type=float, help='Fração dos dados para o conjunto de teste.')
    parser.add_argument('--early-stop', default=25, type=int, help='Rodadas para early stopping.')
    args = parser.parse_args()

    print("--- INICIANDO TREINADOR DE MODELOS ESPECIALISTAS (V3.2 - ALVO CORRIGIDO) ---")
    print(f"Ambiente: XGBoost {XGBOOST_VERSION} | Sklearn {SKLEARN_VERSION}")

    for par_alvo in PARES_PARA_TREINAR:
        try:
            treinar_um_modelo(par_alvo, args)
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha ao treinar o modelo para {par_alvo}: {e}")
            import traceback

            traceback.print_exc()
    print("\n--- Processo de treinamento de todos os modelos concluído. ---")