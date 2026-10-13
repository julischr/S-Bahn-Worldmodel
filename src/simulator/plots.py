"""Stufe 4: Plots (Ein-Schritt, Pruefsteine, Szenarien). Aufruf: python -m src.simulator.plots [abschnitt]"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import OUT

PL = OUT / "plots"
PL.mkdir(exist_ok=True, parents=True)
C1, C2, C3, C4 = "#2a6f97", "#d1495b", "#555555", "#edae49"


def reliability_plots():
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for lab, ls in (("roh", "-"), ("kalibriert", "--")):
        t = pd.read_csv(OUT / f"reliability_ausfall_{lab}.csv")
        ax[0].errorbar(t.mean_pred, t.obs_rate, yerr=[t.obs_rate - t.obs_lo, t.obs_hi - t.obs_rate], marker="o", ls=ls, lw=1, label=lab, ms=4)
    ax[0].plot([1e-4, 0.3], [1e-4, 0.3], color="k", lw=0.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_title("Ausfall an k+1: vorhergesagt vs beobachtet (Test)", fontsize=9); ax[0].set_xlabel("mittlere vorhergesagte Wahrscheinlichkeit"); ax[0].set_ylabel("beobachtete Häufigkeit"); ax[0].legend(fontsize=8)
    tz = {lab: pd.read_csv(OUT / f"reliability_zuwachs_{lab}.csv") for lab in ("roh", "kalibriert")}
    for i, ev in enumerate(["Delta<=0", "Delta>=+2", "Delta>=+5"]):
        for lab, ls in (("roh", "-"), ("kalibriert", "--")):
            t = tz[lab][tz[lab].ereignis == ev]
            ax[i + 1].errorbar(t.mean_pred, t.obs_rate, yerr=[t.obs_rate - t.obs_lo, t.obs_hi - t.obs_rate], marker="o", ls=ls, lw=1, label=lab, ms=4)
        ax[i + 1].plot([0, 1], [0, 1], color="k", lw=0.7)
        ax[i + 1].set_title(f"Zuwachs {ev} (Test)", fontsize=9); ax[i + 1].set_xlabel("vorhergesagt"); ax[i + 1].set_ylabel("beobachtet")
    fig.tight_layout(); fig.savefig(PL / "01_reliability.png", dpi=130); plt.close(fig)


def shap_plot():
    a = pd.read_csv(OUT / "shap_ausfall.csv").head(15)[::-1]
    b = pd.read_csv(OUT / "shap_zuwachs.csv").head(15)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].barh(a.merkmal, a.mean_abs_shap, color=C1); ax[0].set_title("SHAP Ausfallmodell: mittleres |SHAP| (log-Odds)", fontsize=9)
    ax[1].barh(b.merkmal, b.mean_abs_shap_summe_klassen, color=C2); ax[1].set_title("SHAP Zuwachsmodell: Summe über 19 Klassen des mittleren |SHAP|", fontsize=9)
    fig.tight_layout(); fig.savefig(PL / "02_shap.png", dpi=130); plt.close(fig)


def stone_plots(tag="basis"):
    A = pd.read_parquet(OUT / f"pruefstein_a_rohdaten_{tag}.parquet")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    s = A.sample(min(len(A), 6000), random_state=1)
    ax[0].scatter(s.real_end, s.sim_q50, s=4, alpha=0.3, color=C1, label="Sim-Median")
    ax[0].vlines(s.real_end[:300], s.sim_q05[:300], s.sim_q95[:300], color=C2, alpha=0.2, lw=0.6)
    ax[0].plot([-5, 40], [-5, 40], color="k", lw=0.7); ax[0].set_xlim(-5, 40); ax[0].set_ylim(-5, 40)
    ax[0].set_xlabel("echte Verspätung am Pfadende (min)"); ax[0].set_ylabel("simuliert Median (min); rot: 90-%-Band (300 Züge)"); ax[0].set_title("(a) Ankunft am Pfadende", fontsize=9)
    ax[1].hist(A.err, bins=np.arange(-15.5, 16.5, 1), color=C1); ax[1].set_title("(a) Median(Sim) − Echt (min)", fontsize=9); ax[1].set_xlabel("min")
    B = pd.read_parquet(OUT / f"pruefstein_b_rohdaten_{tag}.parquet")
    bins = np.arange(-10, 20, 1)
    ax[2].hist(B.real_build, bins=bins, alpha=0.6, color=C3, label=f"echt (Mittel {B.real_build.mean():.2f})", density=True)
    ax[2].hist(B.sim_build_mean, bins=bins, alpha=0.6, color=C2, label=f"Sim, Lauf-Mittel je Zug ({B.sim_build_mean.mean():.2f})", density=True)
    ax[2].set_title("(b) Aufbau Eintritt→Ende (Rang 0↔11)", fontsize=9); ax[2].legend(fontsize=8); ax[2].set_xlabel("min")
    fig.tight_layout(); fig.savefig(PL / "03_pruefstein_a_b.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, (f, col_e, col_s, ttl, fmt) in enumerate([("pruefstein_c", "echt_mittel", "sim_mittel", "(c) mittlerer Zuwachs Folgezug (min)", "{:.1f}"),
                                                     ("pruefstein_d", "echt_rate", "sim_rate", "(d) Ausfallquote Folgezug (%)", "{:.1f}")]):
        t = pd.read_csv(OUT / f"{f}_{tag}.csv")
        labs = list(dict.fromkeys(t.vorgaenger)); rows = list(dict.fromkeys(t.abstand))
        sc = 100 if k == 1 else 1
        M_e = np.array([[t[(t.abstand == r) & (t.vorgaenger == c)][col_e].iloc[0] * sc for c in labs] for r in rows])
        M_s = np.array([[t[(t.abstand == r) & (t.vorgaenger == c)][col_s].iloc[0] * sc for c in labs] for r in rows])
        w = 0.38
        x = np.arange(len(rows) * len(labs))
        ax[k].bar(x - w / 2, M_e.ravel(), w, color=C3, label="echt")
        ax[k].bar(x + w / 2, M_s.ravel(), w, color=C2, label="Simulation")
        ax[k].set_xticks(x); ax[k].set_xticklabels([f"{r}\n{c}" for r in rows for c in labs], fontsize=5, rotation=90)
        ax[k].set_title(ttl, fontsize=9); ax[k].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PL / "04_pruefstein_c_d.png", dpi=130); plt.close(fig)

    t = pd.read_csv(OUT / f"pruefstein_e_{tag}.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.median_echt.replace(np.inf, np.nan), 0.4, color=C3, label="echt (Testtage)")
    ax.bar(x + 0.2, t.median_sim_lauf_mittel.replace(np.inf, np.nan), 0.4, color=C2, label="Simulation (Mittel der Lauf-Mediane)")
    ax.errorbar(x + 0.2, t.median_sim_lauf_mittel, yerr=[t.median_sim_lauf_mittel - t.median_sim_q025, t.median_sim_q975 - t.median_sim_lauf_mittel], fmt="none", color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(t.klasse); ax.set_xlabel("Schwere in den ersten 15 min (min)"); ax.set_ylabel("Median Erholungszeit ab t0+15 (min)")
    ax.legend(fontsize=8); ax.set_title("(e) Erholungszeit je Schwereklasse", fontsize=9)
    fig.tight_layout(); fig.savefig(PL / "05_pruefstein_e.png", dpi=130); plt.close(fig)

    p = pd.read_csv(OUT / f"profil_verspaetung_{tag}.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in enumerate(["Ost", "West"]):
        q = p[p.richtung == r]
        ax[i].plot(q.station_rang, q.echt_mittel_verspaetung, "o-", color=C3, label="echt")
        ax[i].plot(q.station_rang, q.sim_mittel_verspaetung, "s--", color=C2, label="Simulation")
        ax[i].set_title(f"mittlere Verspätung je Station, {r}", fontsize=9); ax[i].set_xlabel("Station (0 Pasing … 11 Leuchtenbergring)"); ax[i].set_ylabel("min"); ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PL / "06_profil.png", dpi=130); plt.close(fig)


def scenario_plots(tag="szenarien"):
    c = pd.read_csv(OUT / f"szenario_kurven_{tag}.csv")
    for art in c["art"].unique():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for k, hvz in enumerate([True, False]):
            q = c[(c.art == art) & (c.hvz == hvz)]
            for j, (n, g) in enumerate(q.groupby("minuten")):
                m = g.groupby("minuten_nach_eingriff").diff_mittel.mean()
                ax[k].plot(m.index, m.values, label=f"N={int(n)} min" if art == "verspaetung" else "Ausfall", color=[C1, C2, C4][j % 3])
            ax[k].axhline(0, color="k", lw=0.6)
            ax[k].set_title(f"{'HVZ' if hvz else 'außerhalb HVZ'}", fontsize=9); ax[k].set_xlabel("Minuten nach Eingriff")
            ax[k].legend(fontsize=8)
        ax[0].set_ylabel("mittlere Verspätungsdifferenz Szenario − Basis, andere Züge (min)")
        fig.suptitle(f"Szenario {art}", fontsize=10)
        fig.tight_layout(); fig.savefig(PL / f"07_szenario_{art}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "alles"
    if which in ("alles", "einschritt"):
        reliability_plots(); shap_plot()
    if which in ("alles", "pruefsteine"):
        stone_plots()
    if which in ("alles", "szenarien"):
        scenario_plots()
