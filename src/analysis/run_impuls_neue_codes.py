"""
Ausführung der Impulsantwort für Code 5 (Hauptfall), Code 8 (Vergleich),
Code 2 (zusätzlich). Zwei Artefakt-Tests zuerst, dann 11 Plots für den
Hauptfall, reduzierter Satz für die Vergleichscodes, plus Direktvergleich
mit Code 34.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.impulse_response import load_state_vector
from src.analysis.impulse_response_diagnostics import STATION_NAMES
from src.analysis.impulse_response_neue_codes import (
    CODES, N_BOOTSTRAP, PRE_MIN, POST_MIN, REL_MINUTES, RNG_SEED,
    bootstrap_ci, duration_for_events, run_pipeline,
)
from src.utils.constants import OUTPUTS_DIR, PLOTS_DIR

HVZ_HOURS = list(range(6, 10)) + list(range(15, 20))


def save_and_print(name, df):
    df.to_csv(OUTPUTS_DIR / f"{name}.csv", index=False)


# --- Artefakt-Tests -----------------------------------------------------

def artefakt_test_verschobene_terzile(diff_df, code, label) -> None:
    """Terzile nach Wert bei -35min statt bei t0."""
    val_m35 = diff_df[-35]
    terzil = pd.qcut(val_m35, 3, labels=["niedrig", "mittel", "hoch"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"niedrig": "#55A868", "mittel": "#DD8452", "hoch": "#C44E52"}
    for t in ["niedrig", "mittel", "hoch"]:
        idx = terzil[terzil == t].index
        curve = diff_df.loc[idx].mean(axis=0, skipna=True)
        ax.plot(REL_MINUTES, curve, color=colors[t], linewidth=2,
                label=f"{t} (n={len(idx)}, Wert@-35={val_m35.loc[idx].mean():.0f}s)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(-35, color="black", linestyle=":", linewidth=1, label="Sortierpunkt -35min")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minuten relativ zu t0"); ax.set_ylabel("Δ delay_mean (s)")
    ax.set_title(f"Code {code} ({label}): Artefakt-Test -- Terzile nach Wert bei -35min\n"
                 "Wandert der Ausschlag mit dem Sortierpunkt? -> Regression zur Mitte")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_4_terzil_m35.png", dpi=150)
    plt.close(fig)


def artefakt_test_zufallsteilung(diff_df, code, label) -> None:
    rng = np.random.default_rng(RNG_SEED + 99)
    groups = rng.integers(0, 3, size=len(diff_df))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g in range(3):
        idx = np.where(groups == g)[0]
        curve = diff_df.iloc[idx].mean(axis=0, skipna=True)
        ax.plot(REL_MINUTES, curve, linewidth=2, label=f"Zufallsgruppe {g+1} (n={len(idx)})")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minuten relativ zu t0"); ax.set_ylabel("Δ delay_mean (s)")
    ax.set_title(f"Code {code} ({label}): Null-Test -- zufällige Dreiteilung\n"
                 "Muss deckungsgleich sein (kein Sortierkriterium)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_5_zufallsteilung.png", dpi=150)
    plt.close(fig)


# --- Plots 1, 2 -----------------------------------------------------------

def plot_1_impulsantwort(diff_df, meta_df, code, label) -> pd.DataFrame:
    mean_curve, ci_low, ci_high, n_days = bootstrap_ci(diff_df, meta_df)
    n_je_zeitpunkt = diff_df.notna().sum(axis=0).values
    result = pd.DataFrame({"minute_relativ_t0": REL_MINUTES, "impulsantwort_delay_s": mean_curve,
                            "ci_low": ci_low, "ci_high": ci_high, "n_ereignisse": n_je_zeitpunkt})
    save_and_print(f"impulsantwort_code{code}", result)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, height_ratios=[2, 1])
    axes[0].fill_between(REL_MINUTES, ci_low, ci_high, color="#C44E52", alpha=0.25)
    axes[0].plot(REL_MINUTES, mean_curve, color="#C44E52", linewidth=2)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Δ delay_mean (s)")
    axes[0].set_title(f"Impulsantwort Code {code} ({label}), n={len(diff_df)} Ereignisse, {n_days} Tage")
    axes[1].plot(REL_MINUTES, n_je_zeitpunkt, color="#8172B2")
    axes[1].set_xlabel("Minuten relativ zu t0"); axes[1].set_ylabel("n Ereignisse")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_1_impulsantwort.png", dpi=150)
    plt.close(fig)
    return result


def plot_2_rohkurven(event_df, control_df, diff_df, code, label) -> None:
    me = event_df.mean(axis=0, skipna=True); mc = control_df.mean(axis=0, skipna=True)
    md = diff_df.mean(axis=0, skipna=True)
    n = diff_df.notna().sum(axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, height_ratios=[2, 1.3, 1])
    axes[0].plot(REL_MINUTES, me, color="#C44E52", label="Ereignisfenster (absolut)")
    axes[0].plot(REL_MINUTES, mc, color="#4C72B0", label="Kontrollfenster (absolut)")
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("delay_mean (s)"); axes[0].legend(); axes[0].set_title(f"Code {code} ({label}): Rohkurven")
    axes[1].plot(REL_MINUTES, md, color="#55A868")
    axes[1].axhline(0, color="gray", linewidth=0.8); axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Δ delay_mean (s)")
    axes[2].plot(REL_MINUTES, n, color="#8172B2")
    axes[2].set_xlabel("Minuten relativ zu t0"); axes[2].set_ylabel("n Ereignisse")
    axes[2].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_2_rohkurven.png", dpi=150)
    plt.close(fig)


def plot_terzile(diff_df, values, title, fname, caveat="") -> pd.Series:
    terzil = pd.qcut(values, 3, labels=["niedrig", "mittel", "hoch"], duplicates="drop")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"niedrig": "#55A868", "mittel": "#DD8452", "hoch": "#C44E52"}
    for t in terzil.cat.categories:
        idx = terzil[terzil == t].index
        curve = diff_df.loc[idx].mean(axis=0, skipna=True)
        ax.plot(REL_MINUTES, curve, color=colors.get(t), linewidth=2, label=f"{t} (n={len(idx)})")
    ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minuten relativ zu t0"); ax.set_ylabel("Δ delay_mean (s)")
    ax.set_title(title + (f"\n{caveat}" if caveat else ""))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    return terzil


def plot_9_je_station(diff_df, meta_df, code, label) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
    for ax, (stop_id, name) in zip(axes.flat, STATION_NAMES.items()):
        idx = meta_df[meta_df["stop_id"] == stop_id].index
        if len(idx) == 0:
            ax.set_title(f"{name} (n=0)"); continue
        curve = diff_df.loc[idx].mean(axis=0, skipna=True)
        ax.plot(REL_MINUTES, curve, color="#C44E52")
        ax.axhline(0, color="gray", linewidth=0.6); ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title(f"{name} (n={len(idx)})", fontsize=9)
    fig.suptitle(f"Code {code} ({label}): je Station")
    fig.supxlabel("Minuten relativ zu t0"); fig.supylabel("Δ delay_mean (s)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_9_je_station.png", dpi=150)
    plt.close(fig)


def plot_11_histogramm(diff_df, code, label) -> None:
    vals = diff_df[0].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(vals.mean(), color="#C44E52", linestyle="--", label=f"Mittel={vals.mean():.0f}s")
    ax.axvline(vals.median(), color="#55A868", linestyle="--", label=f"Median={vals.median():.0f}s")
    ax.set_xlabel("Δ delay_mean bei t0 (s), Einzelereignisse"); ax.set_ylabel("Anzahl")
    ax.set_title(f"Code {code} ({label}): Verteilung der Einzelwerte bei t0 (n={len(vals)})\n"
                 f"Std={vals.std():.0f}s, Min={vals.min():.0f}s, Max={vals.max():.0f}s")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_11_histogramm.png", dpi=150)
    plt.close(fig)
    return vals.describe()


if __name__ == "__main__":
    print("Lade Zustandsvektor...")
    sv = load_state_vector()

    results = {}
    for code in [5, 8, 2]:
        print(f"\n{'='*60}\nCode {code}: {CODES[code]}\n{'='*60}")
        results[code] = run_pipeline(code, sv)

    # --- Artefakt-Tests für Hauptfall (5) und Vergleich (8) ---
    for code in [5, 8]:
        r = results[code]
        print(f"\n--- Artefakt-Tests Code {code} ---")
        artefakt_test_verschobene_terzile(r["diff_df"], code, CODES[code])
        artefakt_test_zufallsteilung(r["diff_df"], code, CODES[code])
        print(f"Plots: qc_neucode_{code}_4_terzil_m35.png, qc_neucode_{code}_5_zufallsteilung.png")

    # --- Hauptfall (5): volle 11-Plot-Suite ---
    code = 5
    r = results[code]
    diff_df, event_df, control_df, meta_df, hits = r["diff_df"], r["event_df"], r["control_df"], r["meta_df"], r["hits"]

    print(f"\n--- Volle Plot-Suite Code {code} ---")
    curve1 = plot_1_impulsantwort(diff_df, meta_df, code, CODES[code])
    plot_2_rohkurven(event_df, control_df, diff_df, code, CODES[code])

    peak_t0 = diff_df[0]
    terzil_peak = plot_terzile(diff_df, peak_t0, f"Code {code}: Terzile nach Peak-Höhe bei t0",
                                f"qc_neucode_{code}_3_terzil_peak.png",
                                "Achtung: Regression zur Mitte -- aussagekräftig ist die FORM, nicht die Höhe")

    meta_df["dauer_min"] = duration_for_events(hits, meta_df)
    terzil_dauer = plot_terzile(diff_df, meta_df["dauer_min"],
                                 f"Code {code}: Terzile nach Ereignisdauer", f"qc_neucode_{code}_7_terzil_dauer.png")

    terzil_trips = plot_terzile(diff_df, meta_df["n_trips"], f"Code {code}: Terzile nach Zahl betroffener Fahrten",
                                 f"qc_neucode_{code}_6_terzil_fahrten.png")

    meta_df["hvz"] = meta_df["t0"].dt.hour.isin(HVZ_HOURS)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for hvz_val, farbe, lbl in [(True, "#C44E52", "HVZ (6-9,15-19 Uhr)"), (False, "#4C72B0", "Rest")]:
        idx = meta_df[meta_df["hvz"] == hvz_val].index
        curve = diff_df.loc[idx].mean(axis=0, skipna=True)
        ax.plot(REL_MINUTES, curve, color=farbe, linewidth=2, label=f"{lbl} (n={len(idx)})")
    ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minuten relativ zu t0"); ax.set_ylabel("Δ delay_mean (s)")
    ax.set_title(f"Code {code}: HVZ vs. Rest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qc_neucode_{code}_8_hvz.png", dpi=150)
    plt.close(fig)

    plot_9_je_station(diff_df, meta_df, code, CODES[code])
    hist_stats = plot_11_histogramm(diff_df, code, CODES[code])
    print(hist_stats)

    meta_df.to_csv(OUTPUTS_DIR / f"qc_neucode_{code}_metadaten.csv", index=False)

    # --- Vergleichscode 8: Plots 1,2 + Terzile Peak (für Formvergleich) ---
    code = 8
    r = results[code]
    diff_df8, event_df8, control_df8, meta_df8 = r["diff_df"], r["event_df"], r["control_df"], r["meta_df"]
    curve8 = plot_1_impulsantwort(diff_df8, meta_df8, code, CODES[code])
    plot_2_rohkurven(event_df8, control_df8, diff_df8, code, CODES[code])
    plot_terzile(diff_df8, diff_df8[0], f"Code {code}: Terzile nach Peak-Höhe bei t0",
                 f"qc_neucode_{code}_3_terzil_peak.png",
                 "Achtung: Regression zur Mitte -- aussagekräftig ist die FORM, nicht die Höhe")

    # --- Code 2 (Polizei): nur Basiskurve ---
    code = 2
    r = results[code]
    curve2 = plot_1_impulsantwort(r["diff_df"], r["meta_df"], code, CODES[code])
    plot_2_rohkurven(r["event_df"], r["control_df"], r["diff_df"], code, CODES[code])

    # --- Plot 10: Direktvergleich mit Code 34 ---
    print("\n--- Plot 10: Direktvergleich mit Code 34 ---")
    code34 = pd.read_csv(OUTPUTS_DIR / "impulsantwort_code34.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(code34["minute_relativ_t0"], code34["impulsantwort_delay_s"], color="gray", linewidth=2,
            label="Code 34 (Signaldefekt, Original, Fenster -60/+240)")
    for code, curve, color in [(5, curve1, "#C44E52"), (8, curve8, "#DD8452"), (2, curve2, "#4C72B0")]:
        mask = curve["minute_relativ_t0"] <= 120
        ax.plot(curve.loc[mask, "minute_relativ_t0"], curve.loc[mask, "impulsantwort_delay_s"],
                color=color, linewidth=2, label=f"Code {code} ({CODES[code].split('(')[0].strip()})")
    ax.axhline(0, color="gray", linewidth=0.8); ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minuten relativ zu t0"); ax.set_ylabel("Δ delay_mean (s)")
    ax.set_title("Direktvergleich: Code 34 vs. Notfall-/Personenschadencodes\n"
                  "(Code 34 hat längeres Fenster, hier auf 0-120min beschnitten für Vergleichbarkeit)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qc_neucode_10_vergleich_code34.png", dpi=150)
    plt.close(fig)

    print("\nFertig. Alle Dateien unter outputs/ bzw. outputs/plots/qc_neucode_*.")
