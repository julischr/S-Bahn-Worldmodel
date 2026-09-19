"""
Teil 3 der Impulsantwort-Nachdiagnose: komplette Neuberechnung mit
Fenster [-60,+60] statt [-60,+240], um zu prüfen, wie viele der 1.902
durch die Überlappungsbereinigung verworfenen Ereignisse bei einem
kürzeren Fenster erhalten blieben.

Setzt module-level PRE_MIN/POST_MIN in impulse_response.py zur Laufzeit
um (kein Datei-Edit, nur dieser Prozess) -- cluster_events, drop_overlapping,
find_controls und curve_for lesen diese Konstanten als Modul-Globals zur
Aufrufzeit, daher wirkt die Umsetzung korrekt auf alle vier Funktionen.
Original-Datei und Original-Ergebnisse bleiben unverändert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.analysis.impulse_response as ir
from src.utils.constants import OUTPUTS_DIR, PLOTS_DIR

ir.POST_MIN = 60  # PRE_MIN bleibt -60

REL_MINUTES = np.arange(ir.PRE_MIN, ir.POST_MIN + 1, 5)


def run() -> None:
    print(f"Fenster: [{ir.PRE_MIN},{ir.POST_MIN}]min")
    hits = ir.load_code_hits()
    all_events = ir.cluster_events(hits)
    print(f"{len(all_events)} Ereignisse (Clustering unabhängig vom Antwortfenster, "
          f"sollte 5478 sein)")

    clean_events, n_dropped = ir.drop_overlapping(all_events)
    print(f"{n_dropped} verworfen wegen Überlappung im Fenster [{ir.PRE_MIN},{ir.POST_MIN}] "
          f"(Original bei [-60,240]: 1902 verworfen)")
    print(f"{len(clean_events)} saubere Ereignisse übrig")

    by_key: dict[tuple[int, str], list] = {}
    for e in all_events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    rng = np.random.default_rng(ir.RNG_SEED)
    events_with_controls = []
    for e in clean_events:
        controls = ir.find_controls(e, by_key, rng)
        if len(controls) >= ir.K_CONTROLS:
            events_with_controls.append((e, controls))
    print(f"{len(events_with_controls)} Ereignisse mit {ir.K_CONTROLS} Kontrollen gefunden")

    print("Lade Zustandsvektor...")
    sv = ir.load_state_vector()

    diff_rows, event_days = [], []
    for e, controls in events_with_controls:
        ec = ir.curve_for(sv, e.stop_id, e.richtung, e.t0)
        cc = pd.concat([ir.curve_for(sv, e.stop_id, e.richtung, c) for c in controls], axis=1).mean(axis=1)
        diff_rows.append(ec - cc)
        event_days.append(e.t0.date())
    diff_df = pd.DataFrame(diff_rows)
    diff_df.columns = REL_MINUTES

    mean_curve = diff_df.mean(axis=0, skipna=True)
    n_je_zeitpunkt = diff_df.notna().sum(axis=0)

    days_arr = np.array(event_days)
    unique_days = np.unique(days_arr)
    rng2 = np.random.default_rng(ir.RNG_SEED + 1)
    day_to_rows = {d: np.where(days_arr == d)[0] for d in unique_days}
    boot_means = np.zeros((1000, len(REL_MINUTES)))
    for b in range(1000):
        sampled_days = rng2.choice(unique_days, size=len(unique_days), replace=True)
        rows = np.concatenate([day_to_rows[d] for d in sampled_days])
        boot_means[b] = diff_df.iloc[rows].mean(axis=0, skipna=True).values
    ci_low = np.nanpercentile(boot_means, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_means, 97.5, axis=0)

    result = pd.DataFrame({
        "minute_relativ_t0": REL_MINUTES,
        "impulsantwort_delay_s": mean_curve.values,
        "n_ereignisse_je_zeitpunkt": n_je_zeitpunkt.values,
        "ci_low": ci_low, "ci_high": ci_high,
    })
    result.to_csv(OUTPUTS_DIR / "qc_impuls_3_kurzfenster_60min.csv", index=False)
    print(f"\nn={len(events_with_controls)} Ereignisse, {len(unique_days)} Tage")
    print(result.round(1).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, height_ratios=[2, 1])
    axes[0].fill_between(result["minute_relativ_t0"], result["ci_low"], result["ci_high"],
                          color="#C44E52", alpha=0.25)
    axes[0].plot(result["minute_relativ_t0"], result["impulsantwort_delay_s"], color="#C44E52", linewidth=2)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Δ delay_mean (s)")
    axes[0].set_title(f"Impulsantwort Code 34, Fenster [-60,+60]min (n={len(events_with_controls)})")

    axes[1].plot(result["minute_relativ_t0"], result["n_ereignisse_je_zeitpunkt"], color="#8172B2")
    axes[1].set_xlabel("Minuten relativ zu t0")
    axes[1].set_ylabel("n Ereignisse")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qc_impuls_3_kurzfenster_60min.png", dpi=150)
    plt.close(fig)
    print("Plot: outputs/plots/qc_impuls_3_kurzfenster_60min.png")


if __name__ == "__main__":
    run()
