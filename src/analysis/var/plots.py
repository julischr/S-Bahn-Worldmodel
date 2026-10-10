"""Alle Plots fuer die VAR-Analyse (Stufe 2). PNG, deutsche Beschriftung."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .stations import STATION_POSITION

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 130})

COLOR_MAIN = "#2A6F97"
COLOR_ACCENT = "#C4471C"
CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_periodogram(raw: pd.DataFrame, resid: pd.DataFrame, stations: list[str], out_path, richtung: str = ""):
    """Plot 1: Lomb-Scargle-Periodogramm auf den tatsaechlich beobachteten (nicht-NaN) Punkten,
    OHNE Interpolation der Nachtluecken (die wuerde bei linearer Fuellung ein kuenstliches
    24h-Artefakt erzeugen) -- vor/nach Taktentfernung."""
    from scipy.signal import lombscargle

    fig, axes = plt.subplots(len(stations), 2, figsize=(11, 3.0 * len(stations)))
    if len(stations) == 1:
        axes = axes.reshape(1, 2)
    idx = raw.index
    t_hours = (idx - idx[0]).total_seconds().to_numpy() / 3600.0
    periods = np.geomspace(0.3, 48, 1500)
    ang_freqs = 2 * np.pi / periods
    for row, st in enumerate(stations):
        for col, (df, title) in enumerate([(raw, "vor Taktentfernung"), (resid, "nach Taktentfernung (Residuen)")]):
            y = df[st].to_numpy()
            mask = ~np.isnan(y)
            t = t_hours[mask]
            yv = y[mask] - np.nanmean(y[mask])
            power = lombscargle(t, yv, ang_freqs, normalize=True)
            ax = axes[row, col]
            ax.plot(periods, power, color=COLOR_MAIN, lw=0.8)
            for p in [24, 12, 8, 6]:
                ax.axvline(p, color=COLOR_ACCENT, lw=0.6, ls="--", alpha=0.6)
            ax.set_xscale("log")
            suffix = f" [Richtung {richtung}]" if richtung else ""
            ax.set_title(f"{st}: {title}{suffix}", fontsize=9)
            ax.set_xlabel("Periode (Stunden)")
            ax.set_ylabel("Lomb-Scargle-Leistung (norm.)")
    _save(fig, out_path)


def plot_aic_bic(tab: pd.DataFrame, chosen_p: int, out_path, richtung: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tab["p"], tab["aic"], marker="o", ms=3, color=COLOR_MAIN, label="AIC")
    ax.plot(tab["p"], tab["bic"], marker="o", ms=3, color=COLOR_ACCENT, label="BIC")
    ax.axvline(chosen_p, color="grey", ls="--", lw=1, label=f"gewaehlt: p={chosen_p}")
    ax.set_xlabel("Lag-Ordnung p (je 5 Minuten)")
    ax.set_ylabel("Informationskriterium")
    suffix = f" [Richtung {richtung}]" if richtung else ""
    ax.set_title(f"Lag-Wahl: AIC und BIC ueber Lag-Ordnung{suffix}")
    ax.legend()
    _save(fig, out_path)


def plot_eigenvalues(eig: np.ndarray, out_path, richtung: str = ""):
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=1)
    ax.scatter(eig.real, eig.imag, color=COLOR_MAIN, s=25, zorder=3)
    ax.axhline(0, color="lightgrey", lw=0.5)
    ax.axvline(0, color="lightgrey", lw=0.5)
    ax.set_xlabel("Realteil")
    ax.set_ylabel("Imaginaerteil")
    suffix = f" [Richtung {richtung}]" if richtung else ""
    ax.set_title(f"Eigenwerte der Begleitmatrix (max |lambda| = {np.max(np.abs(eig)):.3f}){suffix}")
    ax.set_aspect("equal")
    _save(fig, out_path)


def plot_irf_grid(theta: np.ndarray, columns: list[str], step_minutes: float, out_path, max_minutes=120, richtung: str = ""):
    K = len(columns)
    steps = int(max_minutes / step_minutes)
    t = np.arange(theta.shape[0]) * step_minutes
    fig, axes = plt.subplots(K, K, figsize=(1.55 * K, 1.35 * K), sharex=True, sharey=False)
    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            ax.plot(t[:steps], theta[:steps, i, j], color=COLOR_MAIN, lw=0.8)
            ax.axhline(0, color="lightgrey", lw=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(columns[j][:4], fontsize=6)
            if j == 0:
                ax.set_ylabel(columns[i][:4], fontsize=6, rotation=0, ha="right", va="center")
    suffix = f" -- Zuege Richtung {richtung}" if richtung else ""
    fig.suptitle(f"IRF-Gitter: Schock Spalte j -> Antwort Zeile i (Cholesky, geografische Ordnung){suffix}", fontsize=10)
    _save(fig, out_path)


def plot_selected_shocks(theta: np.ndarray, columns: list[str], shock_stations: list[str], step_minutes: float, out_path, max_minutes=90, boot_theta: np.ndarray | None = None, richtung: str = ""):
    steps = int(max_minutes / step_minutes)
    t = np.arange(theta.shape[0]) * step_minutes
    fig, axes = plt.subplots(1, len(shock_stations), figsize=(5.5 * len(shock_stations), 4.2), sharey=False)
    if len(shock_stations) == 1:
        axes = [axes]
    max_pos = max(STATION_POSITION.values())
    cmap = plt.get_cmap("plasma")
    if boot_theta is not None:
        lo = np.nanpercentile(boot_theta, 2.5, axis=0)
        hi = np.nanpercentile(boot_theta, 97.5, axis=0)
    for ax, shock in zip(axes, shock_stations):
        j = columns.index(shock)
        for i, col in enumerate(columns):
            color = cmap(STATION_POSITION[col] / max_pos)
            ax.plot(t[:steps], theta[:steps, i, j], color=color, lw=1.3, label=col)
            if boot_theta is not None:
                ax.fill_between(t[:steps], lo[:steps, i, j], hi[:steps, i, j], color=color, alpha=0.12, lw=0)
        ax.axhline(0, color="lightgrey", lw=0.5)
        suffix = f" [Richtung {richtung}]" if richtung else ""
        ax.set_title(f"Schock an {shock}{suffix}")
        ax.set_xlabel("Minuten nach Schock")
    axes[0].set_ylabel("Antwort (delay_mean, s)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_pos))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.15, aspect=40)
    cbar.set_label("Position entlang Strecke (0=Pasing, 11=Leuchtenbergring)")
    _save(fig, out_path)


def _heatmap(mat: np.ndarray, columns: list[str], title: str, cbar_label: str, out_path, cmap=CMAP_SEQ, fmt="{:.0f}", richtung: str = ""):
    if richtung:
        title = f"{title} [Richtung {richtung}]"
    K = len(columns)
    fig, ax = plt.subplots(figsize=(0.62 * K + 2.5, 0.62 * K + 2))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels(columns, rotation=90, fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(columns, fontsize=8)
    ax.set_xlabel("Schock an Station j")
    ax.set_ylabel("Antwort an Station i")
    ax.set_title(title, fontsize=10)
    for i in range(K):
        for j in range(K):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5, color="white" if im.norm(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    _save(fig, out_path)


def plot_time_to_peak_heatmap(time_to_peak: np.ndarray, columns: list[str], out_path, richtung: str = ""):
    _heatmap(time_to_peak, columns, "Zeit bis Antwortmaximum (Minuten)", "Minuten", out_path, cmap="viridis", fmt="{:.0f}", richtung=richtung)


def plot_amplitude_heatmap(amplitude: np.ndarray, columns: list[str], out_path, richtung: str = ""):
    vmax = np.nanmax(np.abs(amplitude))
    fig_cmap = "RdBu_r"
    _heatmap(amplitude, columns, "Amplitude der Antwort (delay_mean, s)", "s", out_path, cmap=fig_cmap, fmt="{:.0f}", richtung=richtung)


def plot_halflife_heatmap(hl: np.ndarray, columns: list[str], out_path, richtung: str = ""):
    _heatmap(hl, columns, "Abkling-Halbwertszeit ab Peak (Minuten)", "Minuten", out_path, cmap="viridis", fmt="{:.0f}", richtung=richtung)


def plot_fevd(fevd_mat: np.ndarray, columns: list[str], horizon_minutes: float, out_path, richtung: str = ""):
    K = len(columns)
    fig, ax = plt.subplots(figsize=(0.75 * K + 3, 5.5))
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(K)
    for j in range(K):
        vals = fevd_mat[:, j]
        ax.bar(range(K), vals, bottom=bottom, color=cmap(j % 20), label=columns[j], width=0.7)
        bottom += vals
    ax.set_xticks(range(K))
    ax.set_xticklabels(columns, rotation=90, fontsize=8)
    ax.set_ylabel("Anteil erklaerter Prognosefehlervarianz")
    suffix = f" [Richtung {richtung}]" if richtung else ""
    ax.set_title(f"Varianzzerlegung (FEVD) bei Horizont {horizon_minutes:.0f} min{suffix}")
    ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left", title="Schockquelle")
    _save(fig, out_path)


def plot_net_coupling(net: pd.Series, out_path_bar, out_path_line, richtung: str = ""):
    order = list(STATION_POSITION.keys())
    net_ordered = net.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLOR_ACCENT if v < 0 else COLOR_MAIN for v in net_ordered.values]
    ax.barh(range(len(net_ordered)), net_ordered.values, color=colors)
    ax.set_yticks(range(len(net_ordered)))
    ax.set_yticklabels(net_ordered.index, fontsize=8)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Netto-Kopplung (Export - Import, FEVD-Anteil)")
    suffix = f" [Richtung {richtung}]" if richtung else ""
    ax.set_title(f"Netto-Quelle (positiv) vs. Netto-Senke (negativ){suffix}")
    _save(fig, out_path_bar)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    xs = [STATION_POSITION[s] for s in net_ordered.index]
    ax2.plot(xs, net_ordered.values, marker="o", color=COLOR_MAIN)
    ax2.axhline(0, color="grey", lw=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(net_ordered.index, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Netto-Kopplung")
    ax2.set_title(f"Netto-Kopplung entlang der Strecke (West -> Ost){suffix}")
    _save(fig2, out_path_line)


def plot_direction_comparison(curves: dict, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (t, y) in curves.items():
        ax.plot(t, y, label=label, lw=1.3)
    ax.set_xlabel("Minuten nach Schock")
    ax.set_ylabel("Antwort (delay_mean, s)")
    ax.set_title("Richtungsvergleich Ost vs. West")
    ax.legend()
    _save(fig, out_path)


def plot_cholesky_robustness(theta_a: np.ndarray, theta_b: np.ndarray, columns: list[str], shock_stations: list[str], label_a: str, label_b: str, step_minutes: float, out_path, max_minutes=90, richtung: str = ""):
    steps = int(max_minutes / step_minutes)
    t = np.arange(theta_a.shape[0]) * step_minutes
    fig, axes = plt.subplots(1, len(shock_stations), figsize=(5.5 * len(shock_stations), 4.2))
    if len(shock_stations) == 1:
        axes = [axes]
    for ax, shock in zip(axes, shock_stations):
        j = columns.index(shock)
        i = j
        ax.plot(t[:steps], theta_a[:steps, i, j], color=COLOR_MAIN, label=label_a, lw=1.4)
        ax.plot(t[:steps], theta_b[:steps, i, j], color=COLOR_ACCENT, label=label_b, lw=1.4, ls="--")
        suffix = f" [Zuege Richtung {richtung}]" if richtung else ""
        ax.set_title(f"Eigenantwort {shock}{suffix}")
        ax.set_xlabel("Minuten")
        ax.axhline(0, color="lightgrey", lw=0.5)
    axes[0].set_ylabel("Antwort (s)")
    axes[0].legend(fontsize=8)
    _save(fig, out_path)


def plot_lag_crosscheck(curves: dict, summary: pd.DataFrame, step_minutes: float, out_path, max_minutes=90):
    """Abschnitt 10.1: Vergleich der Pasing<->Leuchtenbergring-Eigenantwort und der mittleren
    FEVD-Fremdvarianz ueber die vier Kombinationen (Ost p=5/p=17, West p=5/p=17).
    curves: dict label -> 1D-array der (kausal richtigen) Kreuzantwort-Kurve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    steps = int(max_minutes / step_minutes)
    t = np.arange(steps) * step_minutes
    styles = {
        "Ost p=5 (Hauptlauf)": (COLOR_MAIN, "-"),
        "Ost p=17 (Kreuzvergleich)": (COLOR_MAIN, "--"),
        "West p=17 (Hauptlauf)": (COLOR_ACCENT, "-"),
        "West p=5 (Kreuzvergleich)": (COLOR_ACCENT, "--"),
    }
    for label, curve in curves.items():
        color, ls = styles.get(label, ("grey", "-"))
        ax1.plot(t, curve[:steps], color=color, ls=ls, lw=1.6, label=label)
    ax1.axhline(0, color="lightgrey", lw=0.5)
    ax1.axvline(25, color="grey", lw=0.7, ls=":")
    ax1.set_xlabel("Minuten nach Schock")
    ax1.set_ylabel("Antwort (delay_mean, s)")
    ax1.set_title("Pasing <-> Leuchtenbergring: 4 Lag-Varianten\n(jeweils kausal richtige Cholesky-Ordnung)", fontsize=10)
    ax1.legend(fontsize=7.5)

    labels_short = ["Ost\np=5", "Ost\np=17", "West\np=17", "West\np=5"]
    colors = [COLOR_MAIN, COLOR_MAIN, COLOR_ACCENT, COLOR_ACCENT]
    ax2.bar(labels_short, summary["mean_foreign_fevd"].to_numpy() * 100, color=colors, alpha=0.85)
    for i, v in enumerate(summary["mean_foreign_fevd"].to_numpy() * 100):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax2.set_ylabel("Mittlere FEVD-Fremdvarianz (%)")
    ax2.set_title("Systemkopplung ueber die vier Varianten", fontsize=10)
    ax2.set_ylim(0, max(summary["mean_foreign_fevd"]) * 100 * 1.25)
    _save(fig, out_path)


def plot_acf_west_vs_ost(acf_west: pd.DataFrame, acf_ost: pd.DataFrame, stations: list[str], step_minutes: float, out_path, highlight_min=(70, 90)):
    """Abschnitt 10.2: ACF West vs. Ost fuer ausgewaehlte Stationen, mit markiertem
    70-90-Minuten-Fenster (Lags 14-18), in dem der Nutzer eine Struktur vermutet hat."""
    fig, axes = plt.subplots(1, len(stations), figsize=(4.3 * len(stations), 4.0), sharey=True)
    if len(stations) == 1:
        axes = [axes]
    lags = acf_west.columns.to_numpy()
    t = lags * step_minutes
    for ax, st in zip(axes, stations):
        ax.axvspan(highlight_min[0], highlight_min[1], color="grey", alpha=0.15, lw=0, label="70-90 min (Lag 14-18)")
        ax.plot(t, acf_west.loc[st], color=COLOR_ACCENT, marker="o", ms=3, lw=1.2, label="West")
        ax.plot(t, acf_ost.loc[st], color=COLOR_MAIN, marker="o", ms=3, lw=1.2, label="Ost")
        ax.axhline(0, color="lightgrey", lw=0.5)
        ax.axvline(20, color="black", lw=0.6, ls=":", alpha=0.6)
        ax.axvline(80, color="black", lw=0.6, ls="--", alpha=0.6)
        ax.set_title(st, fontsize=10)
        ax.set_xlabel("Lag (Minuten)")
    axes[0].set_ylabel("Autokorrelation (episodengewichtet gepoolt)")
    axes[0].legend(fontsize=8)
    fig.suptitle("ACF der Fourier-Residuen: West vs. Ost (gestrichelt: 20min-Schwingung, gepunktet: 80min)", fontsize=10)
    _save(fig, out_path)


def plot_acf_bic_step(aic_bic_west: pd.DataFrame, out_path):
    """Zeigt die logdet(Sigma_u)-Kurve fuer West mit Markierung der Stufe bei p=15-17."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(aic_bic_west["p"], aic_bic_west["logdet_Sigma_u"], marker="o", ms=4, color=COLOR_MAIN)
    ax.axvspan(14.5, 17.5, color=COLOR_ACCENT, alpha=0.15, label="sichtbare Stufe p=15-17 (75-85min)")
    ax.set_xlabel("Lag-Ordnung p (je 5 Minuten)")
    ax.set_ylabel("log det(Sigma_u)")
    ax.set_title("West: log det(Sigma_u) je Lag -- sichtbare Stufe bei p=15-17")
    ax.legend()
    _save(fig, out_path)


def plot_complex_oscillation(theta_component: np.ndarray, columns_pair: tuple[str, str], eig_info: dict, step_minutes: float, out_path, richtung: str = ""):
    t = np.arange(len(theta_component)) * step_minutes
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, theta_component, color=COLOR_MAIN, lw=1.2)
    ax.axhline(0, color="lightgrey", lw=0.5)
    ax.set_xlabel("Minuten nach Schock")
    ax.set_ylabel("Antwort (s)")
    suffix = f" [Richtung {richtung}]" if richtung else ""
    ttl = f"Gedaempfte Schwingung {columns_pair[0]}<-{columns_pair[1]}: |lambda|={eig_info['mod']:.3f}, Periode={eig_info['period_min']:.1f} min{suffix}"
    ax.set_title(ttl, fontsize=10)
    _save(fig, out_path)
