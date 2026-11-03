# -*- coding: utf-8 -*-
"""
Atividade da Aula 1 -- Mineração de Séries Temporais (PUCPR)
Análise da série temporal da ação PETR4 (Petrobras PN, B3)
Aluno: César Willian Pacheco

Responde às perguntas do slide "Atividade" (Aula 1):
  1. A série é univariada ou multivariada?
  2. Qual a taxa de amostragem?
  3. Dimensionalidade (variáveis x observações)?
  4. Data e valor do maior fechamento (Close)?
  5. Para Close e Volume: tendência, ciclo, sazonalidade, autocorrelação e estacionariedade?

Execução:  python analise_petr4.py        (gera figuras em ./figuras e resultados.json)
Dados: PETR4.csv (colunas Date,Open,High,Low,Close,Volume). No Colab do professor o
arquivo é obtido com  !gdown 1vUh6gxcfFeKreamBk3JKS7QsAmRS8ME5 ; aqui, se o arquivo
não existir localmente, tenta-se o mesmo download.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.0})
COR = "#1f3a5f"      # série
COR2 = "#c0392b"     # destaque (tendência, picos)
FIG = "figuras"
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Carregamento
# ----------------------------------------------------------------------------
ARQ = "PETR4.csv"
if not os.path.exists(ARQ):
    try:
        import gdown  # noqa
        os.system("gdown 1vUh6gxcfFeKreamBk3JKS7QsAmRS8ME5")
    except Exception:
        raise SystemExit("PETR4.csv não encontrado. Coloque o arquivo na pasta do script.")

data = pd.read_csv(ARQ, parse_dates=["Date"]).dropna()
data = data.sort_values("Date").reset_index(drop=True)
data["Date"] = pd.to_datetime(data["Date"]).dt.normalize()   # remove a hora (16:56)
data = data.set_index("Date")
res = {}

# ----------------------------------------------------------------------------
# 2. Univariada x multivariada, dimensionalidade
# ----------------------------------------------------------------------------
res["variaveis"] = list(data.columns)
res["n_variaveis"] = int(data.shape[1])
res["n_observacoes"] = int(data.shape[0])
res["periodo"] = [str(data.index.min().date()), str(data.index.max().date())]
res["tipo"] = "multivariada" if data.shape[1] > 1 else "univariada"

# ----------------------------------------------------------------------------
# 3. Taxa / intervalo de amostragem
# ----------------------------------------------------------------------------
delta = pd.Series(data.index).diff().dt.days.dropna()
res["intervalo_dias"] = delta.value_counts().sort_index().to_dict()
res["intervalo_dias"] = {int(k): int(v) for k, v in res["intervalo_dias"].items()}
dias_corridos = (data.index.max() - data.index.min()).days
res["obs_por_ano_civil"] = round(len(data) / (dias_corridos / 365.25), 1)
res["obs_por_ano_calendario"] = {int(a): int(n) for a, n in data.groupby(data.index.year).size().items()}
dias_uteis = pd.bdate_range(data.index.min(), data.index.max())
res["dias_uteis_sem_pregao"] = int(len(dias_uteis) - len(data))
res["obs_fim_de_semana"] = int((data.index.dayofweek >= 5).sum())

# ----------------------------------------------------------------------------
# 4. Maior fechamento
# ----------------------------------------------------------------------------
imax = data["Close"].idxmax()
res["maior_close"] = {"data": str(imax.date()), "valor": float(data.loc[imax, "Close"])}
imin = data["Close"].idxmin()
res["menor_close"] = {"data": str(imin.date()), "valor": float(data.loc[imin, "Close"])}

# ----------------------------------------------------------------------------
# 5. Características de Close e Volume
# ----------------------------------------------------------------------------
def analisar(serie, nome, janela_tend=126, periodo_sazonal=252, lags=260):
    """Tendência, ciclo, sazonalidade, autocorrelação e estacionariedade de uma série diária."""
    out = {}
    s = serie.astype(float)

    # --- tendência: média móvel de ~6 meses de pregão e regressão linear global
    tend = s.rolling(janela_tend, center=True).mean()
    t = np.arange(len(s))
    coef = np.polyfit(t, s.values, 1)
    out["inclinacao_por_obs"] = float(coef[0])
    out["variacao_total_pct"] = float((s.iloc[-1] / s.iloc[0] - 1) * 100)
    medias_anuais = s.groupby(s.index.year).mean()
    out["media_por_ano"] = {int(a): float(round(v, 2)) for a, v in medias_anuais.items()}

    # --- ciclos: picos relevantes da série suavizada (mesma técnica do notebook do professor)
    suave = s.rolling(63, center=True).mean()
    prom = suave.std() * 0.8
    picos, _ = find_peaks(suave.bfill().ffill(), prominence=prom)
    datas_picos = s.index[picos]
    out["picos"] = [str(d.date()) for d in datas_picos]
    out["duracao_ciclos_obs"] = [int(x) for x in np.diff(picos)]

    # --- sazonalidade: decomposição aditiva anual (252 pregões) e ACF no lag anual
    dec = seasonal_decompose(s, model="additive", period=periodo_sazonal)
    amp_saz = float(dec.seasonal.max() - dec.seasonal.min())
    out["amplitude_sazonal"] = amp_saz
    out["amplitude_sazonal_relativa_pct"] = float(amp_saz / s.mean() * 100)
    out["desvio_residuo"] = float(dec.resid.std())
    out["forca_sazonal"] = float(max(0, 1 - dec.resid.var() / (dec.resid + dec.seasonal).var()))
    # sazonalidade semanal (dia da semana)
    out["media_por_dia_semana"] = {int(k): float(round(v, 3)) for k, v in
                                   s.groupby(s.index.dayofweek).mean().items()}

    # --- autocorrelação
    ac = acf(s, nlags=lags, fft=True)
    out["acf"] = {"lag1": float(ac[1]), "lag5": float(ac[5]), "lag21": float(ac[21]),
                  "lag63": float(ac[63]), "lag126": float(ac[126]), "lag252": float(ac[252])}
    d1 = s.diff().dropna()
    ac_d = acf(d1, nlags=20, fft=True)
    out["acf_diferenca"] = {"lag1": float(ac_d[1]), "lag2": float(ac_d[2]), "lag5": float(ac_d[5])}
    out["limite_significancia"] = float(1.96 / np.sqrt(len(s)))

    # --- estacionariedade: ADF (H0: raiz unitária) e KPSS (H0: estacionária)
    adf = adfuller(s, autolag="AIC")
    kp = kpss(s, regression="c", nlags="auto")
    adf_d = adfuller(d1, autolag="AIC")
    kp_d = kpss(d1, regression="c", nlags="auto")
    out["adf"] = {"estatistica": float(adf[0]), "p_valor": float(adf[1]), "lags": int(adf[2])}
    out["kpss"] = {"estatistica": float(kp[0]), "p_valor": float(kp[1])}
    out["adf_diferenca"] = {"estatistica": float(adf_d[0]), "p_valor": float(adf_d[1])}
    out["kpss_diferenca"] = {"estatistica": float(kp_d[0]), "p_valor": float(kp_d[1])}
    # estatísticas em 5 janelas (como no notebook do professor)
    w = len(s) // 5
    out["janelas"] = [{"media": float(s.iloc[i*w:(i+1)*w].mean()),
                       "desvio": float(s.iloc[i*w:(i+1)*w].std())} for i in range(5)]

    # ---------------- figuras ----------------
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s.index, s.values, color=COR, label=nome)
    ax.plot(tend.index, tend.values, color=COR2, lw=2, label=f"Tendência (média móvel {janela_tend} pregões)")
    ax.plot(datas_picos, s.loc[datas_picos], "o", color=COR2, ms=5, label="Picos (ciclos)")
    ax.set_title(f"PETR4 -- {nome}: série, tendência e picos")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/{nome.lower()}_tendencia.png"); plt.close(fig)

    fig = dec.plot(); fig.set_size_inches(8, 6)
    fig.suptitle(f"PETR4 -- {nome}: decomposição aditiva (período = {periodo_sazonal} pregões)", y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/{nome.lower()}_decomposicao.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    plot_acf(s, lags=lags, zero=False, ax=axs[0], color=COR, vlines_kwargs={"colors": COR})
    axs[0].set_title(f"ACF -- {nome} (nível)")
    plot_pacf(s, lags=40, zero=False, ax=axs[1], color=COR, vlines_kwargs={"colors": COR}, method="ywm")
    axs[1].set_title(f"PACF -- {nome} (nível)")
    fig.tight_layout(); fig.savefig(f"{FIG}/{nome.lower()}_acf_pacf.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    axs[0].plot(d1.index, d1.values, color=COR, lw=0.6); axs[0].set_title(f"Primeira diferença -- {nome}")
    plot_acf(d1, lags=40, zero=False, ax=axs[1], color=COR, vlines_kwargs={"colors": COR})
    axs[1].set_title(f"ACF -- primeira diferença de {nome}")
    fig.tight_layout(); fig.savefig(f"{FIG}/{nome.lower()}_diferenca.png"); plt.close(fig)
    return out


res["Close"] = analisar(data["Close"], "Close")
res["Volume"] = analisar(data["Volume"], "Volume")

# figura de visão geral (as duas séries, eixos separados -- nunca eixo duplo)
fig, axs = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
axs[0].plot(data.index, data["Close"], color=COR); axs[0].set_ylabel("Close (R$)")
axs[0].plot(imax, data.loc[imax, "Close"], "o", color=COR2)
axs[0].annotate(f"máx. {data.loc[imax,'Close']:.2f} em {imax.date()}", (imax, data.loc[imax, "Close"]),
                xytext=(-160, -30), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color=COR2))
axs[1].plot(data.index, data["Volume"] / 1e6, color=COR); axs[1].set_ylabel("Volume (milhões)")
axs[0].set_title("PETR4 -- fechamento diário e volume negociado (2020--2025)")
fig.tight_layout(); fig.savefig(f"{FIG}/visao_geral.png"); plt.close(fig)

# figura: intervalos de amostragem
fig, ax = plt.subplots(figsize=(5, 2.4))
vc = delta.value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color=COR)
ax.set_xlabel("Dias entre observações consecutivas"); ax.set_ylabel("Frequência")
ax.set_title("Intervalo de amostragem observado")
fig.tight_layout(); fig.savefig(f"{FIG}/intervalo_amostragem.png"); plt.close(fig)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------------
# Resumo em texto
# ----------------------------------------------------------------------------
print("=== PETR4 -- respostas ===")
print(f"Variáveis ({res['n_variaveis']}): {res['variaveis']}  -> série {res['tipo']}")
print(f"Observações: {res['n_observacoes']}  ({res['periodo'][0]} a {res['periodo'][1]})")
print(f"Intervalos (dias) entre observações: {res['intervalo_dias']}")
print(f"Observações por ano civil: {res['obs_por_ano_civil']}  | por ano: {res['obs_por_ano_calendario']}")
print(f"Maior Close: {res['maior_close']}")
for nome in ["Close", "Volume"]:
    r = res[nome]
    print(f"\n--- {nome} ---")
    print(f"variação total: {r['variacao_total_pct']:.1f}% | médias anuais: {r['media_por_ano']}")
    print(f"picos: {r['picos']} | duração dos ciclos (pregões): {r['duracao_ciclos_obs']}")
    print(f"amplitude sazonal: {r['amplitude_sazonal']:.3g} ({r['amplitude_sazonal_relativa_pct']:.1f}% da média); "
          f"força sazonal: {r['forca_sazonal']:.2f}; desvio resíduo: {r['desvio_residuo']:.3g}")
    print(f"ACF: {r['acf']}  | ACF da diferença: {r['acf_diferenca']} | limite ±{r['limite_significancia']:.3f}")
    print(f"ADF p={r['adf']['p_valor']:.4f}  KPSS p={r['kpss']['p_valor']:.3f}  | "
          f"diferença: ADF p={r['adf_diferenca']['p_valor']:.2e}  KPSS p={r['kpss_diferenca']['p_valor']:.3f}")
    print("janelas:", [(round(j['media'], 2), round(j['desvio'], 2)) for j in r['janelas']])
