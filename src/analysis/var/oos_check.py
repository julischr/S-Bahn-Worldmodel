"""Validierung 9.7: Out-of-Sample-Pruefung des Ost-VAR(5).

Trainiert das Hauptmodell (VAR(5), Ost) nur auf Episoden aus Jan-Sep 2025, prueft
1-Schritt- und 5-Schritt-Vorhersagen auf Episoden aus Okt-Dez 2025 gegen zwei
naive Baselines (Random Walk, historischer Mittelwert je Station/Wochentag/Stunde
aus den Trainingsdaten). Zusaetzlich: getrennte Schaetzung H1 (Jan-Jun) vs. H2
(Jul-Dez) zum Vergleich der Kernbefunde.
"""

import numpy as np
import pandas as pd

from .prepare import load_state_vector, pivot_direction
from .detrend import detrend_frame
from .episodes import interpolate_short_gaps, build_daily_episodes
from .var_model import fit_var, VARResult
from .irf import structural_irf
from .irf_metrics import fevd, peak_amplitude_time
from .stations import STATION_NAMES_WEST_OST
from .run_all import EXCLUDED_DATES

P_FIXED = 5


def prepare_episodes():
    df = load_state_vector()
    wide = pivot_direction(df, "Ost")[STATION_NAMES_WEST_OST]
    resid, _ = detrend_frame(wide, 4, 3)
    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]
    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    episodes, ep_meta = build_daily_episodes(interp, min_length=24)
    return episodes, ep_meta


def split_episodes(episodes, ep_meta, cutoff_start, cutoff_end):
    """cutoff_start/cutoff_end: pd.Timestamp bounds (inclusive) on episode date."""
    idx = ep_meta.index[(ep_meta["date"] >= cutoff_start) & (ep_meta["date"] <= cutoff_end)].tolist()
    return [episodes[i] for i in idx], ep_meta.iloc[idx]


def historical_mean_lookup(train_episodes):
    """Mittelwert je (station, weekday, hour) ueber alle Trainings-Episoden."""
    frames = []
    for ep in train_episodes:
        f = ep.copy()
        f["weekday"] = ep.index.weekday
        f["hour"] = ep.index.hour
        frames.append(f)
    allf = pd.concat(frames)
    cols = [c for c in allf.columns if c not in ("weekday", "hour")]
    lookup = {}
    for col in cols:
        lookup[col] = allf.groupby(["weekday", "hour"])[col].mean()
    return lookup


def forecast_episode(ep: pd.DataFrame, res: VARResult, p: int, horizons=(1, 5)):
    """Rekursive Mehrschritt-Prognose je Zeitpunkt t (mit genuegend Lags/Horizont) in ep.
    Gibt DataFrame mit Fehlern (Beobachtung - Prognose) je Station/Horizont zurueck."""
    arr = ep.to_numpy()
    T = arr.shape[0]
    K = arr.shape[1]
    As = res.A_matrices()
    c = res.const()
    max_h = max(horizons)
    rows = []
    for t in range(p - 1, T - max_h - 1):
        # history window y_{t-p+1..t}, most recent last
        hist = [arr[t - l] for l in range(p)]  # hist[0]=y_t, hist[1]=y_{t-1}, ...
        preds = {}
        cur_hist = list(hist)
        for h in range(1, max_h + 1):
            yhat = c.copy()
            for lag in range(1, p + 1):
                yhat = yhat + As[lag - 1] @ cur_hist[lag - 1]
            preds[h] = yhat
            cur_hist = [yhat] + cur_hist[:-1]
        for h in horizons:
            actual = arr[t + h]
            err = actual - preds[h]
            rw_pred = arr[t]  # random walk: last observed value
            rw_err = actual - rw_pred
            rows.append({"h": h, "t_idx": t, "err": err, "rw_err": rw_err, "actual_idx": t + h})
    return rows


def evaluate_oos(train_episodes, test_episodes, cols, p=P_FIXED, horizons=(1, 5)):
    res = fit_var(train_episodes, p)
    hist_lookup = historical_mean_lookup(train_episodes)

    var_errs = {h: [] for h in horizons}
    rw_errs = {h: [] for h in horizons}
    hist_errs = {h: [] for h in horizons}

    for ep in test_episodes:
        rows = forecast_episode(ep, res, p, horizons)
        arr = ep.to_numpy()
        weekday = ep.index.weekday
        hour = ep.index.hour
        for row in rows:
            h = row["h"]
            var_errs[h].append(row["err"])
            rw_errs[h].append(row["rw_err"])
            t_target = row["actual_idx"]
            wd, hr = weekday[t_target], hour[t_target]
            hist_pred = np.array([
                hist_lookup[col].get((wd, hr), np.nan) for col in cols
            ])
            hist_errs[h].append(arr[t_target] - hist_pred)

    rows_out = []
    for h in horizons:
        var_arr = np.array(var_errs[h])
        rw_arr = np.array(rw_errs[h])
        hist_arr = np.array(hist_errs[h])
        for i, col in enumerate(cols):
            rows_out.append({
                "station": col, "horizon_steps": h, "horizon_min": h * 5,
                "rmse_var": np.sqrt(np.nanmean(var_arr[:, i] ** 2)),
                "mae_var": np.nanmean(np.abs(var_arr[:, i])),
                "rmse_rw": np.sqrt(np.nanmean(rw_arr[:, i] ** 2)),
                "mae_rw": np.nanmean(np.abs(rw_arr[:, i])),
                "rmse_hist": np.sqrt(np.nanmean(hist_arr[:, i] ** 2)),
                "mae_hist": np.nanmean(np.abs(hist_arr[:, i])),
                "n": var_arr.shape[0],
            })
    return pd.DataFrame(rows_out), res


def half_comparison(episodes, ep_meta, cols, order, horizon=48, p=P_FIXED):
    """Fit separat auf H1 (Jan-Jun) und H2 (Jul-Dez), Kernkennzahlen vergleichen."""
    h1_start, h1_end = pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-06-30").date()
    h2_start, h2_end = pd.Timestamp("2025-07-01").date(), pd.Timestamp("2025-12-31").date()
    ep_h1, meta_h1 = split_episodes(episodes, ep_meta, h1_start, h1_end)
    ep_h2, meta_h2 = split_episodes(episodes, ep_meta, h2_start, h2_end)

    out = {}
    for label, eps in [("H1_Jan_Jun", ep_h1), ("H2_Jul_Dez", ep_h2)]:
        res = fit_var(eps, p)
        As = res.A_matrices()
        theta = structural_irf(As, res.Sigma_u, order, horizon)
        fevd_h = fevd(theta, horizon)
        time_to_peak, amplitude = peak_amplitude_time(theta, step_minutes=5.0)
        pi, li = cols.index("Pasing"), cols.index("Leuchtenbergring")
        K = len(cols)
        export = np.zeros(K); import_ = np.zeros(K)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                export[j] += fevd_h[i, j]
                import_[i] += fevd_h[i, j]
        net = pd.Series(export - import_, index=cols)
        out[label] = dict(
            n_episodes=len(eps), T=res.T,
            peak_zeit_pasing_leuchtenbergring_min=time_to_peak[li, pi],
            amplitude_pasing_leuchtenbergring_s=amplitude[li, pi],
            mean_fevd_fremdvarianz=float(np.mean([1 - fevd_h[i, i] for i in range(K)])),
            net_pasing=float(net["Pasing"]), net_laim=float(net["Laim"]),
        )
    return out
