"""Validierung 9.4: Versuch eines gemeinsamen VAR mit 24 Reihen (12 Stationen x 2 Richtungen).

Nutzt dieselbe Datenaufbereitung wie run_all.py (Fourier-Detrending, Ausschluss der
auffaelligen Betriebstage), aber baut EINE gemeinsame Episodenbildung ueber alle 24 Spalten
(Ost + West nebeneinander), prueft dann ob ein VAR(p) auf dieser Basis sinnvoll schaetzbar ist.

Neue Datei, dupliziert keine bestehende Logik ausser durch Wiederverwendung der
importierten Funktionen aus stations/prepare/detrend/episodes/var_model.
"""

import numpy as np
import pandas as pd

from .prepare import load_state_vector, pivot_direction
from .detrend import detrend_frame
from .episodes import interpolate_short_gaps, build_daily_episodes
from .var_model import fit_var, aic_bic_for_p, stability_eigenvalues
from .stations import STATION_NAMES_WEST_OST
from .run_all import EXCLUDED_DATES


def build_joint_wide():
    df = load_state_vector()
    wide_ost = pivot_direction(df, "Ost")[STATION_NAMES_WEST_OST]
    wide_west = pivot_direction(df, "West")[STATION_NAMES_WEST_OST]
    wide_ost = wide_ost.add_suffix("_Ost")
    wide_west = wide_west.add_suffix("_West")
    joint = wide_ost.join(wide_west, how="inner")
    return joint, wide_ost.columns.tolist(), wide_west.columns.tolist()


def run_joint24(p_range=range(1, 11)):
    joint, cols_ost, cols_west = build_joint_wide()
    n_daily, n_weekly = 4, 3
    resid, fourier_info = detrend_frame(joint, n_daily, n_weekly)

    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]

    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    episodes, ep_meta = build_daily_episodes(interp, min_length=24)

    n_episodes = len(episodes)
    total_obs = sum(len(e) for e in episodes)

    result = {
        "n_episodes": n_episodes,
        "total_obs_raw": total_obs,
        "ep_meta": ep_meta,
        "cols_ost": cols_ost,
        "cols_west": cols_west,
        "episodes": episodes,
        "fourier_info": fourier_info,
    }

    if n_episodes < 5 or total_obs < 500:
        result["feasible"] = False
        result["reason"] = f"Nur {n_episodes} Episoden / {total_obs} Beobachtungen gemeinsam -- zu wenig fuer K=24."
        return result

    aic_bic = aic_bic_for_p(episodes, p_range)
    result["aic_bic"] = aic_bic

    K = 24
    obs_per_param_rows = []
    for p in p_range:
        n_params_per_eq = K * p + 1
        total_params = K * n_params_per_eq
        # usable T at this p (rough: episodes with length > p)
        T_p = sum(max(len(e) - p, 0) for e in episodes)
        obs_per_param_rows.append({"p": p, "total_params": total_params, "T_usable": T_p,
                                    "obs_per_param": T_p / total_params if total_params else np.nan})
    obs_tab = pd.DataFrame(obs_per_param_rows)
    result["obs_per_param"] = obs_tab

    # Choose p via BIC only among feasible (obs_per_param > ~5) to avoid numerically degenerate fits
    feasible_p = obs_tab.loc[obs_tab["obs_per_param"] > 3, "p"].tolist()
    if not feasible_p:
        result["feasible"] = False
        result["reason"] = "Selbst bei p=1 zu wenige Beobachtungen pro Parameter (<3) -- K=24 VAR nicht sinnvoll schaetzbar."
        return result

    bic_sub = aic_bic[aic_bic["p"].isin(feasible_p)]
    chosen_p = int(bic_sub.loc[bic_sub["bic"].idxmin(), "p"])
    result["chosen_p"] = chosen_p

    res = fit_var(episodes, chosen_p)
    As = res.A_matrices()
    eig = stability_eigenvalues(As)
    result["var_result"] = res
    result["As"] = As
    result["eig"] = eig
    result["max_abs_eig"] = float(np.max(np.abs(eig)))
    result["feasible"] = res.T > K * (K * chosen_p + 1) * 2  # heuristic: need >=2x obs/param overall
    result["reason"] = (
        f"p={chosen_p} gewaehlt, T={res.T} Beobachtungen, "
        f"{K*(K*chosen_p+1)} Parameter, Verhaeltnis={res.T/(K*(K*chosen_p+1)):.1f}."
    )
    return result


if __name__ == "__main__":
    r = run_joint24()
    print(r["reason"])
    print("feasible:", r.get("feasible"))
