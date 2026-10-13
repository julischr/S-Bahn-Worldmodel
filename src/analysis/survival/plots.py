"""Plots 1-7 der Survival-Analyse. Aufruf: uv run python -m src.analysis.survival.plots"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D
from . import km as K
from .groups import code_table, gruppe_map
from .run_cox import SEV_BINS, SEV_LABELS

PL = D.OUT / "plots"
C_EV, C_PL, C_PL2, C_HVZ, C_NHVZ = "#C4471C", "#6B7280", "#9CA3AF", "#C4471C", "#2A6F97"
GRID = np.arange(0, 241, 2.0)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "font.size": 9})


def km_grid_plot(curves, curves2, title, fname, names):
    groups = list(curves.keys())
    n = len(groups)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), sharex=True, sharey=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, g in zip(axes.ravel(), groups):
        c = curves[g]
        ax.fill_between(GRID, c["p_lo"], c["p_hi"], color=C_PL, alpha=.2, lw=0)
        ax.step(GRID, c["p_point"], where="post", color=C_PL, lw=1.4)
        if g in curves2:
            ax.step(GRID, curves2[g]["p_point"], where="post", color=C_PL2, lw=1.2, ls="--")
        ax.fill_between(GRID, c["x_lo"], c["x_hi"], color=C_EV, alpha=.2, lw=0, step="post")
        ax.step(GRID, c["x_point"], where="post", color=C_EV, lw=1.6)
        lab = "alle Auslöser" if g == "alle" else ("selten (<50)" if g == "selten" else f"Code {g}: {names.get(int(g), '')[:26]}")
        ax.set_title(f"{lab}\nn={c['n']:,}".replace(",", "."), fontsize=8, loc="left")
        ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("min seit t0")
    for ax in axes[:, 0]:
        ax.set_ylabel("noch nicht erholt")
    fig.suptitle(title + "\nrot = Störung, grau = Placebo P1 (Plan), grau gestrichelt = Placebo P2 (zugverankert); Band = 95 %-KI, tageweiser Bootstrap",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(PL / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    PL.mkdir(parents=True, exist_ok=True)
    ep = pd.read_parquet(D.ZW / "episoden.parquet")
    tab = code_table(ep)
    names = dict(zip(tab["code"], tab["bedeutung"]))
    cv = pd.read_pickle(D.ZW / "km_kurven.pkl")
    res = pd.read_csv(D.OUT / "km_ergebnisse.csv")

    km_grid_plot(cv[("placebo", "st", 180, "std")], cv[("placebo2", "st", 180, "std")],
                 "Plot 1: Nicht-erholt-Anteil an der Meldestation (Schwelle 180 s), je Auslösercode",
                 "01_km_je_code_station_180.png", names)
    km_grid_plot(cv[("placebo", "ln", 180, "std")], cv[("placebo2", "ln", 180, "std")],
                 "Plot 2: Nicht-erholt-Anteil auf der ganzen Stammstrecke (Schwelle 180 s), je Auslösercode",
                 "02_km_je_code_strecke_180.png", names)

    # Plot 3: Schwellen
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, lv, nm in zip(axes, ("st", "ln"), ("Meldestation", "ganze Strecke")):
        for thr, ls in ((180, "-"), (360, ":")):
            c = cv[("placebo", lv, thr, "std")]["alle"]
            ax.step(GRID, c["x_point"], where="post", color=C_EV, ls=ls, lw=1.8, label=f"Störung, <{thr} s")
            ax.step(GRID, c["p_point"], where="post", color=C_PL, ls=ls, lw=1.6, label=f"Placebo P1, <{thr} s")
        ax.set_title(nm, loc="left"); ax.set_xlabel("min seit t0"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("noch nicht erholt"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Plot 3: Schwelle 180 s vs. 360 s (alle Auslöser gepoolt)", fontsize=10)
    fig.tight_layout(); fig.savefig(PL / "03_schwellenvergleich.png", dpi=130); plt.close(fig)

    # Plot 4: Histogramm
    pl = pd.read_parquet(D.ZW / "placebo.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    bins = np.arange(0, 245, 5)
    for ax, (lbl, d, col) in zip(axes, (("Störung", ep, C_EV), ("Placebo P1", pl, C_PL))):
        x = d.dropna(subset=["t_R0_180_st_std"])
        rec = x[x["e_R0_180_st_std"] == 1]["t_R0_180_st_std"].clip(upper=239)
        w = np.ones(len(rec)) / len(x)
        ax.hist(rec, bins=bins, weights=w, color=col, alpha=.85)
        ax.set_title(f"{lbl} (n={len(x):,})".replace(",", "."), loc="left")
        ax.text(.98, .95, f"ohne Wirkung (erste 3 Züge pünktlich): {x['noeff_R0_180'].mean():.0%}\n"
                f"zensiert (nicht erholt beobachtet): {1 - x['e_R0_180_st_std'].mean():.0%}\n"
                f"bis 240 min erholt: {(x['e_R0_180_st_std'] == 1).mean():.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_xlabel("Erholungszeit (min), nur erholte Fälle; letzter Balken = 235+")
    axes[0].set_ylabel("Anteil aller Fälle je 5-min-Klasse")
    fig.suptitle("Plot 4: Verteilung der Erholungszeit an der Meldestation (180 s)", fontsize=10)
    fig.tight_layout(); fig.savefig(PL / "04_histogramm_erholungszeit.png", dpi=130); plt.close(fig)

    # Plot 5: Forest
    ct = pd.read_csv(D.OUT / "cox_koeffizienten.csv")
    ct = ct[ct["modell"] == "haupt_R15_180_station"]
    groups = [("Auslösercode (Ref. 36)", ct[ct.term.str.startswith("code_")]),
              ("Meldestation", ct[ct.term.str.startswith("st_")]),
              ("Übrige", ct[~ct.term.str.startswith(("code_", "st_"))])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 8), gridspec_kw=dict(width_ratios=[1.1, 1, .8]))
    for ax, (ttl, g) in zip(axes, groups):
        g = g.iloc[::-1]
        y = np.arange(len(g))
        ax.errorbar(g.hr, y, xerr=[g.hr - g.hr_lo, g.hr_hi - g.hr], fmt="o", color=C_EV, ms=4, lw=1.2)
        ax.axvline(1, color="k", lw=.8)
        ax.set_yticks(y); ax.set_yticklabels(g.term.str.replace("code_", "Code ").str.replace("st_", ""), fontsize=8)
        ax.set_xscale("log"); ax.set_title(ttl, loc="left"); ax.set_xlabel("Hazard Ratio (>1 = schnellere Erholung)")
    fig.suptitle("Plot 5: Cox-Modell, Erholung ab t0+15 min (Meldestation, 180 s); 95 %-KI, clusterrobust nach Betriebstag", fontsize=10)
    fig.tight_layout(); fig.savefig(PL / "05_forest_cox.png", dpi=130); plt.close(fig)

    # Plot 6: Kipppunkt
    kl = pd.read_csv(D.OUT / "kipppunkt_klassen_median.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    num = lambda s: pd.to_numeric(s, errors="coerce")
    for ax, pn, ttl in zip(axes, ("placebo", "placebo2"), ("Vergleich mit Placebo P1", "Vergleich mit Placebo P2")):
        for hv, col, nm, off in ((0, C_NHVZ, "Störung, außerhalb HVZ", -.12), (1, C_HVZ, "Störung, HVZ", .12)):
            s = kl[(kl.quelle == "ereignis") & (kl.hvz.astype(str) == str(hv))].set_index("klasse").reindex(SEV_LABELS)
            x = np.arange(len(SEV_LABELS)) + off
            ax.errorbar(x, num(s["median"]), yerr=[num(s["median"]) - num(s["median_lo"]), num(s["median_hi"]) - num(s["median"])],
                        fmt="o-", color=col, label=nm, capsize=2, lw=1.2, ms=5)
        for hv, mk, nm, off in ((0, "s", "Placebo, außerhalb HVZ", -.12), (1, "D", "Placebo, HVZ", .12)):
            s = kl[(kl.quelle == pn) & (kl.hvz.astype(str) == str(hv))].set_index("klasse").reindex(SEV_LABELS)
            x = np.arange(len(SEV_LABELS)) + off
            ax.errorbar(x, num(s["median"]), yerr=[num(s["median"]) - num(s["median_lo"]), num(s["median_hi"]) - num(s["median"])],
                        fmt=mk + "--", color=C_PL, label=nm, capsize=2, lw=1, ms=4, mfc="white")
        ax.set_xticks(range(len(SEV_LABELS))); ax.set_xticklabels(SEV_LABELS)
        ax.set_xlabel("Schwere = max. Verspätung in den ersten 15 min (min)"); ax.set_title(ttl, loc="left")
    axes[0].set_ylabel("Median-Erholungszeit ab t0+15 (min)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Plot 6: Erholungszeit je Schwereklasse (Meldestation, 180 s; Klassen mit < 20 Fällen fehlen; fehlender Punkt = Median nicht erreicht)", fontsize=9)
    fig.tight_layout(); fig.savefig(PL / "06_kipppunkt_schwereklassen.png", dpi=130); plt.close(fig)

    # Plot 7: Ausfallquote je Schwereklasse (Ereignis vs Placebo)
    days = np.array(sorted(ep["opday_nr"].unique()))
    Wd = K.make_day_weights(len(days), 1000, 13).astype(float)
    rows = []
    for who, d, dc in (("Störung", ep, "opday_nr"),
                       ("Placebo P1", pl.assign(day_nr=np.array(pl["parent_opday"], dtype="datetime64[D]").astype("int64")), "day_nr"),
                       ("Placebo P2", pd.read_parquet(D.ZW / "placebo2.parquet").pipe(lambda p: p.assign(day_nr=np.array(p["parent_opday"], dtype="datetime64[D]").astype("int64"))), "day_nr")):
        d = d.dropna(subset=["schwere_min", "ausfallquote"]).copy()
        d["klasse"] = pd.cut(d["schwere_min"], SEV_BINS, labels=SEV_LABELS, right=False)
        for kls in SEV_LABELS:
            s = d[d.klasse == kls]
            if len(s) < 20:
                continue
            di = np.searchsorted(days, s[dc])
            w = Wd[:, di]
            bm = (w * s["ausfallquote"].to_numpy()).sum(1) / w.sum(1)
            rows.append(dict(quelle=who, klasse=kls, n=len(s), ausfallquote=s["ausfallquote"].mean(),
                             lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5)))
    aq = pd.DataFrame(rows); aq.to_csv(D.OUT / "ausfallquote_schwereklassen.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for who, col, mk, off in (("Störung", C_EV, "o", -.1), ("Placebo P1", C_PL, "s", 0), ("Placebo P2", C_PL2, "^", .1)):
        s = aq[aq.quelle == who].set_index("klasse").reindex(SEV_LABELS)
        ax.errorbar(np.arange(5) + off, s.ausfallquote * 100, yerr=[(s.ausfallquote - s.lo) * 100, (s.hi - s.ausfallquote) * 100],
                    fmt=mk + "-", color=col, label=who, capsize=2, lw=1.2, ms=5)
    ax.set_xticks(range(5)); ax.set_xticklabels(SEV_LABELS)
    ax.set_xlabel("Schwere-Klasse (max. Verspätung in den ersten 15 min, min)")
    ax.set_ylabel("ausgefallene Halte in [t0+15, t0+135) (%)"); ax.legend(frameon=False)
    ax.set_title("Plot 7: Ausfallquote nach t0 je Schwereklasse (Meldestation, 95 %-KI tageweiser Bootstrap)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(PL / "07_ausfallquote_schwereklassen.png", dpi=130); plt.close(fig)
    print(aq.round(4).to_string())


if __name__ == "__main__":
    main()
