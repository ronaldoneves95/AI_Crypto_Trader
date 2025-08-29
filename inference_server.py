# -*- coding: utf-8 -*-
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import logging
import os
from feature_engineering import criar_features

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


PARES_SUPORTADOS = ["BTCUSDT", "ETHUSDT"]
modelos = {}

print("[SERVER] Carregando modelos especialistas...")
for par in PARES_SUPORTADOS:
    caminho_modelo = f'model_{par}.pkl'
    if os.path.exists(caminho_modelo):
        try:
            modelos[par] = joblib.load(caminho_modelo)
            print(f"  > Modelo para {par} carregado com sucesso.")
        except Exception as e:
            print(f"  > ERRO ao carregar o modelo para {par}: {e}")
    else:
        print(f"  > AVISO: Modelo '{caminho_modelo}' não encontrado. O par {par} não será suportado.")

if not modelos:
    print("[SERVER] ERRO CRÍTICO: Nenhum modelo foi carregado. Execute o model_trainer.py primeiro.")
    raise SystemExit(1)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        dados_json = request.json
        par = dados_json.get('par')
        candles = dados_json.get('candles')

        if not par or not candles:
            return jsonify({'success': False, 'error': "Payload inválido: 'par' e 'candles' são obrigatórios."})

        if par not in modelos:
            return jsonify({'success': False, 'error': f"Nenhum modelo treinado disponível para o par {par}."})

        artefato = modelos[par]
        modelo_preditivo = artefato['model']
        FEATURES_LIST = artefato['features_list']
        MODEL_CLASSES = artefato['classes']
        criar_features_func = artefato['feature_creation_function']

        df = pd.DataFrame(candles)

        features_df = criar_features_func(df, drop_na=False)

        latest_features = features_df.iloc[-1:]

        if latest_features[FEATURES_LIST].isnull().values.any():
            return jsonify({'success': False,
                            'error': f'Dados de candle insuficientes para gerar uma linha de features completa. Necessário mais histórico.'})

        X = latest_features[FEATURES_LIST].values
        probabilidades = modelo_preditivo.predict_proba(X)[0]

        prob_map = {int(MODEL_CLASSES[i]): float(probabilidades[i]) for i in range(len(MODEL_CLASSES))}

        resposta = {
            'success': True,
            'prob_baixa': prob_map.get(0, 0.0),
            'prob_alta': prob_map.get(1, 0.0),
            'prob_neutra': 0.0
        }
        return jsonify(resposta)

    except Exception as e:
        print(f"[SERVER-ERROR] Ocorreu uma exceção no endpoint /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print(f"\n[SERVER] Servidor de inferência Multi-Modelo iniciado. Modelos carregados para: {list(modelos.keys())}")
    print("[SERVER] Lógica de features agora é carregada dinamicamente de cada modelo.")
    app.run(host='0.0.0.0', port=5000)