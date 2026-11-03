# -*- coding: utf-8 -*-
"""
Atividade da Aula 2 -- Mineração de Séries Temporais (PUCPR)
Forecasting da série UNRATE (taxa mensal de desemprego nos EUA, FRED) com ETS e LSTM
Aluno: César Willian Pacheco

Tarefa: prever as últimas 12 observações com ETS e com LSTM; qual tem o menor RMSE?

Protocolo:
  * holdout temporal -- as 12 últimas observações são o teste; tudo o que vem antes é treino;
    nada é embaralhado e nenhum parâmetro/escala é ajustado com dados de teste;
  * ETS: candidatos ETS(A,N,N), ETS(A,A,N), ETS(A,Ad,N) e, por completude, versões com
    sazonalidade aditiva (m=12); a configuração final é escolhida pelo AICc no TREINO;
  * LSTM: mesma arquitetura do notebook do professor (janela w -> LSTM(u) -> Dense(12),
    abordagem direta multi-output), MinMax ajustado só no treino, early stopping em 20 %
    finais das janelas de treino; a configuração (w, u) é escolhida pelo menor erro de
    VALIDAÇÃO (não pelo teste) e o resultado é reportado em 5 sementes.
  * baselines naive (mean, naive, drift) como referência de dificuldade.

Dados: UNRATE.csv do FRED (https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE).
Aceita o formato do professor (colunas date,UNRATE) ou do FRED (observation_date,UNRATE).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
FIG = "figuras"; os.makedirs(FIG, exist_ok=True)
C_TR, C_TE, C_ETS, C_LSTM, C_B = "black", "black", "#1f3a5f", "#c0392b", "gray"

# ----------------------------------------------------------------------------
# 1. Dados e holdout temporal
# ----------------------------------------------------------------------------
df = pd.read_csv("UNRATE.csv", na_values=["."])
df.columns = ["date", "UNRATE"]
df["date"] = pd.to_datetime(df["date"])
n_faltantes = int(df["UNRATE"].isna().sum())
df = df.dropna().reset_index(drop=True)      # mesma convenção do notebook do professor
series = df["UNRATE"].values.astype(float)
datas = df["date"]
h = 12
train, test = series[:-h], series[-h:]
datas_test = datas.iloc[-h:]
res = {"n_total": int(len(series)), "n_treino": int(len(train)), "n_teste": h,
       "periodo": [str(datas.iloc[0].date()), str(datas.iloc[-1].date())],
       "periodo_teste": [str(datas_test.iloc[0].date()), str(datas_test.iloc[-1].date())],
       "obs_faltantes_removidas": n_faltantes}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def mape(y, p): return float(100 * np.mean(np.abs((np.asarray(y) - np.asarray(p)) / np.asarray(y))))

# ----------------------------------------------------------------------------
# 2. Baselines naive (referência)
# ----------------------------------------------------------------------------
base = {"mean": np.full(h, train.mean()), "naive": np.full(h, train[-1]),
        "drift": train[-1] + (train[-1] - train[0]) / (len(train) - 1) * np.arange(1, h + 1)}
res["baselines"] = {k: {"rmse": rmse(test, v), "mae": mae(test, v), "mape": mape(test, v)} for k, v in base.items()}

# ----------------------------------------------------------------------------
# 3. ETS -- seleção por AICc no treino
# ----------------------------------------------------------------------------
candidatos = [
    ("ETS(A,N,N)", dict(error="add", trend=None, seasonal=None)),
    ("ETS(A,A,N)", dict(error="add", trend="add", seasonal=None)),
    ("ETS(A,Ad,N)", dict(error="add", trend="add", damped_trend=True, seasonal=None)),
    ("ETS(A,N,A)", dict(error="add", trend=None, seasonal="add", seasonal_periods=12)),
    ("ETS(A,A,A)", dict(error="add", trend="add", seasonal="add", seasonal_periods=12)),
    ("ETS(A,Ad,A)", dict(error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=12)),
]
ets_tab = []
ets_fits = {}
for nome, kw in candidatos:
    fit = ETSModel(train, **kw).fit(disp=False)
    pred = fit.forecast(h)
    ets_fits[nome] = (fit, pred)
    ets_tab.append({"modelo": nome, "aicc": float(fit.aicc), "aic": float(fit.aic),
                    "k_parametros": int(len(fit.params)),
                    "rmse_teste": rmse(test, pred), "mae_teste": mae(test, pred)})
res["ets_candidatos"] = ets_tab
melhor = min(ets_tab, key=lambda r: r["aicc"])["modelo"]
ets_fit, ets_pred = ets_fits[melhor]
params = dict(zip(ets_fit.param_names, [float(v) for v in ets_fit.params]))
res["ets_escolhido"] = {"modelo": melhor, "parametros": params, "rmse": rmse(test, ets_pred),
                        "mae": mae(test, ets_pred), "mape": mape(test, ets_pred),
                        "previsoes": [float(x) for x in ets_pred]}
# intervalo de previsão 95% (simulação) para a figura
np.random.seed(0); sim = ets_fit.simulate(nsimulations=h, repetitions=500, anchor="end")
sim = np.asarray(sim)
ic_low, ic_high = np.percentile(sim, 2.5, axis=1), np.percentile(sim, 97.5, axis=1)

# ----------------------------------------------------------------------------
# 4. LSTM -- abordagem direta (janela w -> 12 saídas), como no notebook do professor
# ----------------------------------------------------------------------------
def janelas_diretas(s, w, hz):
    X, y = [], []
    for i in range(w, len(s) - hz + 1):
        X.append(s[i - w:i]); y.append(s[i:i + hz])
    X = np.array(X, dtype=np.float32).reshape(-1, w, 1)
    return X, np.array(y, dtype=np.float32)


def treinar_lstm(train, w, u, seed, epochs=300, batch=8, patience=20):
    tf.keras.utils.set_random_seed(seed)
    scaler = MinMaxScaler().fit(train.reshape(-1, 1))          # escala ajustada SÓ no treino
    tr = scaler.transform(train.reshape(-1, 1)).ravel()
    X, y = janelas_diretas(tr, w, h)
    model = Sequential([Input(shape=(w, 1)), LSTM(u), Dense(h)])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(X, y, epochs=epochs, batch_size=batch, validation_split=0.2,
                     shuffle=False, verbose=0, callbacks=[es])
    X_last = tr[-w:].reshape(1, w, 1)
    pred = scaler.inverse_transform(model.predict(X_last, verbose=0).reshape(-1, 1)).ravel()
    n_par = int(model.count_params())
    val_best = float(min(hist.history["val_loss"]))
    return pred, val_best, len(hist.history["loss"]), n_par, X.shape[0]


# 4a. seleção de (w, u) pela perda de validação (média de 3 sementes) -- sem olhar o teste
grade = [(12, 8), (24, 8), (24, 16), (36, 8)]
sel = []
for w, u in grade:
    vals = []
    for seed in range(3):
        _, v, _, n_par, n_jan = treinar_lstm(train, w, u, seed)
        vals.append(v)
    sel.append({"janela": w, "unidades": u, "val_loss_media": float(np.mean(vals)),
                "parametros": n_par, "janelas_treino": n_jan})
    print(f"LSTM w={w} u={u}: val_loss={np.mean(vals):.5f} params={n_par} janelas={n_jan}")
res["lstm_selecao"] = sel
best = min(sel, key=lambda r: r["val_loss_media"])
w, u = best["janela"], best["unidades"]

# 4b. avaliação da configuração escolhida em 5 sementes
runs = []
for seed in range(5):
    pred, v, ep, n_par, n_jan = treinar_lstm(train, w, u, seed)
    runs.append({"seed": seed, "rmse": rmse(test, pred), "mae": mae(test, pred), "mape": mape(test, pred),
                 "val_loss": v, "epocas": ep, "previsoes": [float(x) for x in pred]})
    print(f"LSTM seed={seed}: RMSE={runs[-1]['rmse']:.4f} épocas={ep}")
lstm_pred = np.array(runs[0]["previsoes"])          # execução principal (seed 0)
lstm_media = np.mean([r["previsoes"] for r in runs], axis=0)
res["lstm"] = {"janela": w, "unidades": u, "parametros": n_par, "janelas_treino": n_jan,
               "execucoes": runs,
               "rmse_medio": float(np.mean([r["rmse"] for r in runs])),
               "rmse_desvio": float(np.std([r["rmse"] for r in runs])),
               "rmse_seed0": runs[0]["rmse"], "rmse_media_das_previsoes": rmse(test, lstm_media),
               "previsoes_media": [float(x) for x in lstm_media]}

# ----------------------------------------------------------------------------
# 5. Figuras
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(datas, series, color=C_TR, lw=0.9)
ax.axvline(datas_test.iloc[0], color=C_LSTM, ls="--")
ax.set_title("UNRATE -- taxa de desemprego mensal nos EUA (dessazonalizada); teste = últimos 12 meses")
ax.set_ylabel("%"); fig.tight_layout(); fig.savefig(f"{FIG}/serie.png"); plt.close(fig)

k = 60
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(datas.iloc[-h-k:-h], train[-k:], color=C_TR, label="Treino (últimos 5 anos)")
ax.plot(datas_test, test, color=C_TE, ls=":", lw=2.2, marker="o", ms=3.5, label="Teste (real)")
ax.fill_between(datas_test, ic_low, ic_high, color=C_ETS, alpha=0.12, label="ETS -- IC 95%")
ax.plot(datas_test, ets_pred, color=C_ETS, lw=2, marker="s", ms=3.5, label=f"{melhor}")
ax.plot(datas_test, lstm_pred, color=C_LSTM, lw=2, marker="^", ms=3.5, label=f"LSTM (w={w}, {u} unid., seed 0)")
ax.plot(datas_test, base["naive"], color=C_B, lw=1, ls="--", label="Naive")
ax.set_ylabel("%"); ax.set_title("Previsão das 12 últimas observações")
ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(f"{FIG}/previsoes.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(datas_test, test, color=C_TE, ls=":", lw=2.2, marker="o", ms=4, label="Teste (real)")
ax.plot(datas_test, ets_pred, color=C_ETS, lw=2, marker="s", ms=4, label=melhor)
for r in runs:
    ax.plot(datas_test, r["previsoes"], color=C_LSTM, lw=0.8, alpha=0.5)
ax.plot(datas_test, lstm_media, color=C_LSTM, lw=2, marker="^", ms=4, label="LSTM -- média de 5 sementes (linhas finas: cada semente)")
ax.set_ylabel("%"); ax.set_title("Detalhe do horizonte de teste")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/detalhe.png"); plt.close(fig)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("\n=== Resultados (RMSE nas 12 observações de teste) ===")
print(f"Série: {res['n_total']} obs ({res['periodo'][0]} a {res['periodo'][1]}), teste {res['periodo_teste']}")
for k_, v in res["baselines"].items(): print(f"baseline {k_:6s}: RMSE={v['rmse']:.4f}")
for r in ets_tab: print(f"{r['modelo']:12s} AICc={r['aicc']:.2f}  RMSE={r['rmse_teste']:.4f}")
print(f"ETS escolhido (AICc): {melhor}  params={params}")
print(f"LSTM w={w} u={u}: RMSE seed0={res['lstm']['rmse_seed0']:.4f}  "
      f"média±dp 5 sementes={res['lstm']['rmse_medio']:.4f}±{res['lstm']['rmse_desvio']:.4f}  "
      f"RMSE da previsão média={res['lstm']['rmse_media_das_previsoes']:.4f}")
print("teste real :", np.round(test, 2).tolist())
print("ETS        :", np.round(ets_pred, 2).tolist())
print("LSTM seed0 :", np.round(lstm_pred, 2).tolist())
