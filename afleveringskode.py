# %%
from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# %%

# 1. STRUCTURAL MODEL
@dataclass(frozen=True)
class Params:

    # Worker and firm types.
    x_L: float = 1.0
    x_H: float = 2.0
    y_L: float = 1.0
    y_H: float = 2.0

    # Mechanisms.
    gamma: float = 0.5     # production complementarity (supermodularity)
    sigma: float = 1.0     # strength of assortative meetings

    # Search / bargaining.
    r: float = 0.05        # discount rate
    delta: float = 0.05    # job-destruction rate
    lam0: float = 0.5      # contact rate while unemployed
    lam1: float = 0.2      # contact rate while employed (OJS)
    b: float = 0.5         # flow value of unemployment
    beta: float = 0.5      # worker bargaining power

    # Population composition.
    muH_pop: float = 0.5   # population share of high-type firms
    pH_workers: float = 0.5  # population share of high-type workers


def production(x: float, y: float, p: Params) -> float:
    """Match output f(x, y) = x + y + gamma * x * y."""
    return x + y + p.gamma * x * y


def meeting_probs(x: float, p: Params) -> dict[str, float]:
    """Conditional meeting probabilities {"L": mu(y_L|x), "H": mu(y_H|x)}."""
    s = 1.0 if x == p.x_H else -1.0
    muL_pop = 1.0 - p.muH_pop
    weight_L = muL_pop * np.exp(p.sigma * s * (-1.0))  # t(y_L) = -1
    weight_H = p.muH_pop * np.exp(p.sigma * s * (1.0))  # t(y_H) = +1
    z = weight_L + weight_H
    return {"L": weight_L / z, "H": weight_H / z}


def unemployment_flow_value(x: float, p: Params) -> float:
    # Closed form
    mu = meeting_probs(x, p)
    muL, muH = mu["L"], mu["H"]
    f_xL = production(x, p.y_L, p)
    f_xH = production(x, p.y_H, p)

    A = p.r + p.delta
    B = p.r + p.delta + p.lam1 * muH

    num = A * B * p.b + p.lam0 * p.beta * (
        muL * A * f_xL + p.beta * p.lam1 * muL * muH * f_xH + muH * B * f_xH
    )
    den = A * B + p.lam0 * p.beta * (
        muL * A + p.beta * p.lam1 * muL * muH + muH * B
    )
    return num / den


def wages(x: float, p: Params) -> dict[str, float]:
    """Flow wages in levels {"L": w(x, y_L), "H": w(x, y_H)}."""
    mu = meeting_probs(x, p)
    muH = mu["H"]
    f_xL = production(x, p.y_L, p)
    f_xH = production(x, p.y_H, p)
    rUx = unemployment_flow_value(x, p)

    w_xH = p.beta * f_xH + (1.0 - p.beta) * rUx
    ojs = (p.beta * (1.0 - p.beta) * p.lam1 * muH) / (p.r + p.delta)
    w_xL = p.beta * f_xL + (1.0 - p.beta) * rUx - ojs * (f_xH - rUx)
    return {"L": w_xL, "H": w_xH}


def stationary_shares(x: float, p: Params) -> dict[str, float]:
    """Stationary shares {"u": u, "eL": e_L(x), "eH": e_H(x)} for worker x."""
    mu = meeting_probs(x, p)
    muL, muH = mu["L"], mu["H"]
    u = p.delta / (p.lam0 + p.delta)
    eL = (p.lam0 * p.delta * muL) / (
        (p.lam0 + p.delta) * (p.delta + p.lam1 * muH)
    )
    eH = (p.lam0 * muH) / (p.lam0 + p.delta) + (
        p.lam1 * muH * p.lam0 * muL
    ) / ((p.lam0 + p.delta) * (p.delta + p.lam1 * muH))
    return {"u": u, "eL": eL, "eH": eH}


def prob_high_firm_given_employed(x: float, p: Params) -> float:
    """Conditional probability P(y_H | x, employed) in steady state."""
    shares = stationary_shares(x, p)
    return shares["eH"] / (shares["eL"] + shares["eH"])


def model_cells(p: Params) -> pd.DataFrame:
    """Analytical 2x2 cell table with wages and the joint employed distribution."""
    rows = []
    for x_label, x in (("L", p.x_L), ("H", p.x_H)):
        pop = p.pH_workers if x == p.x_H else (1.0 - p.pH_workers)
        shares = stationary_shares(x, p)
        w = wages(x, p)
        rows.append(dict(x_label=x_label, x_val=x, y_label="L", y_val=p.y_L,
                         wage=w["L"], mass=pop * shares["eL"]))
        rows.append(dict(x_label=x_label, x_val=x, y_label="H", y_val=p.y_H,
                         wage=w["H"], mass=pop * shares["eH"]))
    df = pd.DataFrame(rows)
    df["pi"] = df["mass"] / df["mass"].sum()
    return df


# %%
# 2. ESTIMATION (analytic type-level benchmark + worker-firm AKM)

def type_level_projection(cells: pd.DataFrame) -> dict[str, float]:
    """Weighted additive projection of the 2x2 cell wages onto type effects."""
    w = {(r.x_label, r.y_label): r.wage for r in cells.itertuples()}
    pi = {(r.x_label, r.y_label): r.pi for r in cells.itertuples()}

    g_L = w[("L", "H")] - w[("L", "L")]
    g_H = w[("H", "H")] - w[("H", "L")]
    interaction = g_H - g_L

    rows, weights, target = [], [], []
    for (xl, yl), wage in w.items():
        rows.append([1.0, 1.0 if xl == "H" else 0.0, 1.0 if yl == "H" else 0.0])
        weights.append(pi[(xl, yl)])
        target.append(wage)
    X = np.asarray(rows)
    omega = np.diag(weights)
    y = np.asarray(target)
    beta = np.linalg.solve(X.T @ omega @ X, X.T @ omega @ y)
    firm_gap = beta[2]

    qL = pi[("L", "L")] + pi[("L", "H")]
    qH = pi[("H", "L")] + pi[("H", "H")]
    avg_true_premium = qL * g_L + qH * g_H
    benchmark_gap = firm_gap - avg_true_premium

    alpha = {"L": 0.0, "H": beta[1]}
    psi = {"L": 0.0, "H": beta[2]}
    a = np.array([alpha[xl] for (xl, _yl) in w])
    ps = np.array([psi[yl] for (_xl, yl) in w])
    wts = np.array([pi[k] for k in w])
    a_bar = np.sum(wts * a)
    ps_bar = np.sum(wts * ps)
    cov = np.sum(wts * (a - a_bar) * (ps - ps_bar))

    return {
        "firm_gap": firm_gap,
        "g_L": g_L,
        "g_H": g_H,
        "interaction": interaction,
        "avg_true_premium": avg_true_premium,
        "benchmark_gap": benchmark_gap,
        "two_cov": 2.0 * cov,
    }


def estimate_akm(panel: pd.DataFrame, tol: float = 1e-10,
                 max_iter: int = 5000) -> float:
    """AKM firm-effect gap psi(H) - psi(L) from a full two-way FE regression.

    Fits the high-dimensional AKM model w_it = alpha_i + psi_j + e_it with one
    fixed effect per worker and per firm, estimated by alternating least squares
    (Gauss-Seidel): given the firm effects, each worker effect is the mean
    residual over that worker's spells, and vice versa. Identification relies on
    a connected worker-firm set (verified separately, see connected_set_share).
    The reported gap is the employment-weighted mean firm effect of high-type
    firms minus that of low-type firms.
    """
    w_codes, _ = pd.factorize(panel["worker_id"])
    f_codes, _ = pd.factorize(panel["firm_id"])
    y = panel["wage"].to_numpy(float)
    n_w = int(w_codes.max()) + 1
    n_f = int(f_codes.max()) + 1
    w_count = np.bincount(w_codes, minlength=n_w)
    f_count = np.bincount(f_codes, minlength=n_f)

    alpha = np.zeros(n_w)
    psi = np.zeros(n_f)
    for _ in range(max_iter):
        alpha = np.bincount(w_codes, weights=y - psi[f_codes],
                            minlength=n_w) / w_count
        psi_new = np.bincount(f_codes, weights=y - alpha[w_codes],
                              minlength=n_f) / f_count
        psi_new -= psi_new[0]  # fix one firm effect for identification
        if np.max(np.abs(psi_new - psi)) < tol:
            psi = psi_new
            break
        psi = psi_new

    is_H_firm = np.zeros(n_f, dtype=bool)
    is_H_firm[f_codes[(panel["firm_type"] == "H").to_numpy()]] = True
    psi_H = np.average(psi[is_H_firm], weights=f_count[is_H_firm])
    psi_L = np.average(psi[~is_H_firm], weights=f_count[~is_H_firm])
    return float(psi_H - psi_L)


# %%
# 3. PANEL SIMULATION
def initialize_states(is_high: np.ndarray, p: Params,
                      rng: np.random.Generator) -> np.ndarray:
    """Draw initial states from the analytical stationary distribution."""
    n = is_high.shape[0]
    state = np.zeros(n, dtype=np.int8)
    for is_h, x in ((True, p.x_H), (False, p.x_L)):
        shares = stationary_shares(x, p)
        probs = np.array([shares["u"], shares["eL"], shares["eH"]])
        probs = probs / probs.sum()
        idx = np.flatnonzero(is_high == is_h)
        state[idx] = rng.choice(3, size=idx.shape[0], p=probs).astype(np.int8)
    return state


def simulate_panel(p: Params, n_workers: int = 8000, n_firms_per_type: int = 200,
                   n_steps: int = 600, burn_in: int = 200, record_every: int = 4,
                   dt: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """Simulate a worker-firm panel and return employed observations.

    States: 0 unemployed, 1 employed at a low-type firm, 2 at a high-type firm.
    Wages attached to each observation are the analytical level wages w(x, y).
    """
    rng = np.random.default_rng(seed)

    is_high = rng.random(n_workers) < p.pH_workers

    muL_H, muH_H = meeting_probs(p.x_H, p).values()
    muL_L, muH_L = meeting_probs(p.x_L, p).values()
    muH = np.where(is_high, muH_H, muH_L)

    F = n_firms_per_type

    state = initialize_states(is_high, p, rng)
    firm = np.full(n_workers, -1, dtype=np.int64)
    emp_L = state == 1
    emp_H = state == 2
    firm[emp_L] = rng.integers(0, F, size=int(emp_L.sum()))
    firm[emp_H] = F + rng.integers(0, F, size=int(emp_H.sum()))

    p_event_U = 1.0 - np.exp(-p.lam0 * dt)
    total_L = p.delta + p.lam1 * muH
    p_event_L = 1.0 - np.exp(-total_L * dt)
    p_to_H_given_event_L = np.where(total_L > 0, (p.lam1 * muH) / total_L, 0.0)
    p_event_H = 1.0 - np.exp(-p.delta * dt)

    cells = model_cells(p)
    wage_map = {(r.x_label, r.y_label): r.wage for r in cells.itertuples()}

    records: list[pd.DataFrame] = []
    for t in range(n_steps):
        u = rng.random(n_workers)
        dest = rng.random(n_workers)
        prev = state.copy()

        # From U.
        mU = prev == 0
        ev = mU & (u < p_event_U)
        go_H = ev & (dest < muH)
        go_L = ev & ~(dest < muH)
        state[go_H] = 2
        firm[go_H] = F + rng.integers(0, F, size=int(go_H.sum()))
        state[go_L] = 1
        firm[go_L] = rng.integers(0, F, size=int(go_L.sum()))

        # From L.
        mL = prev == 1
        ev = mL & (u < p_event_L)
        go_H = ev & (dest < p_to_H_given_event_L)
        go_U = ev & ~(dest < p_to_H_given_event_L)
        state[go_H] = 2
        firm[go_H] = F + rng.integers(0, F, size=int(go_H.sum()))
        state[go_U] = 0
        firm[go_U] = -1

        # From H.
        mH = prev == 2
        ev = mH & (u < p_event_H)
        state[ev] = 0
        firm[ev] = -1

        if t >= burn_in and (t - burn_in) % record_every == 0:
            emp = state > 0
            idx = np.flatnonzero(emp)
            records.append(pd.DataFrame({
                "period": t,
                "worker_id": idx,
                "worker_type": np.where(is_high[idx], "H", "L"),
                "firm_id": firm[idx],
                "firm_type": np.where(state[idx] == 2, "H", "L"),
            }))

    panel = pd.concat(records, ignore_index=True)
    panel["wage"] = [
        wage_map[(xt, yt)]
        for xt, yt in zip(panel["worker_type"], panel["firm_type"])
    ]
    return panel


# %%
# 4. SCENARIO DRIVER + DIAGNOSTICS
def validate_stationary_distribution(panel: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Compare simulated vs analytical P(y_H | x, employed) by worker type."""
    rows = []
    for x_label, x in (("L", p.x_L), ("H", p.x_H)):
        sub = panel[panel["worker_type"] == x_label]
        rows.append({
            "worker_type": x_label,
            "P_yH_emp_analytic": prob_high_firm_given_employed(x, p),
            "P_yH_emp_simulated": float((sub["firm_type"] == "H").mean()),
        })
    return pd.DataFrame(rows)


def mover_share(panel: pd.DataFrame) -> float:
    """Share of workers observed at more than one firm."""
    firms_per_worker = panel.groupby("worker_id")["firm_id"].nunique()
    return float((firms_per_worker > 1).mean())


def connected_set_share(panel: pd.DataFrame) -> float:
    """Share of observations in the largest connected worker-firm set.

    Workers and firms are nodes; each observation links a worker to a firm.
    A simple union-find groups nodes into connected components.
    """
    # Connectedness is checked explicitly because AKM identification requires a
    # connected worker-firm set. The result is 1.0 in every scenario by design
    # (the simulation is built to be fully connected), not by coincidence.
    w_codes, _ = pd.factorize(panel["worker_id"])
    f_codes, _ = pd.factorize(panel["firm_id"])
    n_w = int(w_codes.max()) + 1
    parent = np.arange(n_w + int(f_codes.max()) + 1)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for w, f in zip(w_codes, n_w + f_codes):
        rw, rf = root(w), root(f)
        if rw != rf:
            parent[rw] = rf

    roots = np.array([root(w) for w in w_codes])
    biggest = np.bincount(roots).argmax()
    return float((roots == biggest).mean())


def run_scenario(name: str, p: Params, sim_kwargs: dict) -> dict[str, object]:
    """Run one scenario: simulate, estimate AKM, and store everything the tables need.

    The analytic "truth" comes from the closed-form 2x2 model; the AKM gap comes
    from the simulated panel.
    """
    truth = type_level_projection(model_cells(p))
    panel = simulate_panel(p, **sim_kwargs)
    firm_gap_akm = estimate_akm(panel)
    stat = validate_stationary_distribution(panel, p).set_index("worker_type")

    return {
        "scenario": _short_label(name),
        "params": p,
        # Population benchmark used by the main results table it isolates the
        # structural benchmark gap free of sampling noise. The simulated
        # firm_gap_akm below is reported only in the validation table to show
        # the two coincide.
        # Analytic structural truth (from the closed-form model).
        "g_L": truth["g_L"],
        "g_H": truth["g_H"],
        "avg_true_premium": truth["avg_true_premium"],
        "firm_gap_benchmark": truth["firm_gap"],
        "two_cov": truth["two_cov"],
        # AKM estimate (from the simulated panel).
        "firm_gap_akm": firm_gap_akm,
        # Simulation diagnostics.
        "P_yH_xL_analytic": stat.loc["L", "P_yH_emp_analytic"],
        "P_yH_xL_sim": stat.loc["L", "P_yH_emp_simulated"],
        "P_yH_xH_analytic": stat.loc["H", "P_yH_emp_analytic"],
        "P_yH_xH_sim": stat.loc["H", "P_yH_emp_simulated"],
        "mover_share": mover_share(panel),
        "connected_set_share": connected_set_share(panel),
    }


def default_scenarios() -> list[tuple[str, Params]]:
    """The four main scenarios plus one technical sanity check (lam1 = 0)."""
    base = dict(r=0.05, delta=0.05, lam0=0.5, lam1=0.2, b=0.5, beta=0.5)
    return [
        ("S1: gamma=0, sigma=0", Params(gamma=0.0, sigma=0.0, **base)),
        ("S2: gamma>0, sigma=0", Params(gamma=0.5, sigma=0.0, **base)),
        ("S3: gamma=0, sigma>0", Params(gamma=0.0, sigma=1.0, **base)),
        ("S4: gamma>0, sigma>0", Params(gamma=0.5, sigma=1.0, **base)),
        ("Sanity: gamma=0, sigma=0, lam1=0",
         Params(gamma=0.0, sigma=0.0, lam0=0.5, lam1=0.0, r=0.05, delta=0.05,
                b=0.5, beta=0.5)),
    ]


def run_all_scenarios(sim_kwargs: dict) -> pd.DataFrame:
    """Run every default scenario and return the summary table."""
    return pd.DataFrame(
        [run_scenario(name, p, sim_kwargs) for name, p in default_scenarios()]
    )


# %%
# Configuration
# Single simulation configuration used for every simulated table: the design
# documented in the paper (8000 workers, 200 firms per type = 400 total).
SIM_KWARGS = dict(
    n_workers=8000,
    n_firms_per_type=200,
    n_steps=600,
    burn_in=200,
    record_every=4,
    dt=0.25,
    seed=0,
)

# Baseline structural parameters shared by every scenario.
_BASE_PARAMS = dict(r=0.05, delta=0.05, lam0=0.5, b=0.5, beta=0.5)

# 10x10 grid for the cell-residual figures.
X_GRID = np.linspace(1.0, 2.0, 10)
Y_GRID = np.linspace(1.0, 2.0, 10)
BASELINE_LAM1 = 0.2
GAMMAS_FIG1 = (0.0, 0.25, 0.5, 0.75)


def _short_label(name: str) -> str:
    """Short scenario label (e.g. 'S1' from 'S1: gamma=0...')."""
    return name.split(":")[0].strip()


# %%
# 5. TABLES
def build_validation_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Model and simulation validation, one row per scenario."""
    tbl = scenarios[[
        "scenario", "P_yH_xL_analytic", "P_yH_xL_sim",
        "P_yH_xH_analytic", "P_yH_xH_sim", "mover_share",
        "connected_set_share",
    ]].copy()
    tbl["akm_gap_diff"] = (
        scenarios["firm_gap_akm"] - scenarios["firm_gap_benchmark"]
    ).abs()
    return tbl


def build_scenario_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Structural premia, population AKM gap, and benchmark gap."""
    out = scenarios[[
        "scenario", "g_L", "g_H", "firm_gap_benchmark", "avg_true_premium",
        "two_cov",
    ]].copy()
    out["benchmark_gap"] = out["firm_gap_benchmark"] - out["avg_true_premium"]
    return out[["scenario", "g_L", "g_H", "firm_gap_benchmark",
                "avg_true_premium", "benchmark_gap", "two_cov"]]


def build_weighting_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Appendix G: the weighting mechanism behind the benchmark gap."""
    rows = []
    for r in scenarios.itertuples():
        p = r.params
        cells = model_cells(p)
        pi = {(c.x_label, c.y_label): c.pi for c in cells.itertuples()}
        pLL, pLH = pi[("L", "L")], pi[("L", "H")]
        pHL, pHH = pi[("H", "L")], pi[("H", "H")]
        PH_L = pLH / (pLL + pLH)
        PH_H = pHH / (pHL + pHH)
        qL = pLL + pLH
        qH = pHL + pHH
        omega_L = qL * PH_L * (1.0 - PH_L)
        omega_H = qH * PH_H * (1.0 - PH_H)
        W_L = omega_L / (omega_L + omega_H) if (omega_L + omega_H) > 0 else qL
        proj = type_level_projection(cells)
        rows.append({
            "scenario": _short_label(r.scenario),
            "P_yH_xL": PH_L,
            "P_yH_xH": PH_H,
            "q_L": qL,
            "omega_akm_xL": W_L,
            "weight_gap": qL - W_L,
            "interaction": proj["interaction"],
            "benchmark_gap": proj["benchmark_gap"],
        })
    return pd.DataFrame(rows)


# %%
# Run the scenarios once and print the three tables.
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
scenarios = run_all_scenarios(SIM_KWARGS)
print("\nTable: validation\n"
      + build_validation_table(scenarios).to_string(index=False))
print("\nTable: scenario results\n"
      + build_scenario_table(scenarios).to_string(index=False))
print("\nTable: weighting mechanism\n"
      + build_weighting_table(scenarios).to_string(index=False))


# %%
# 6. MAIN-TEXT FIGURES
def _premia_bars(df: pd.DataFrame, title: str, subtitle: str) -> None:
    """Grouped bar chart of g_L, Delta-psi_hat, g_H with a midpoint tick."""
    labels = [_short_label(s) for s in df["scenario"]]
    x = np.arange(len(df))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars_gl = ax.bar(x - w, df["g_L"], w, color="#1f77b4",
                     label=r"$g(x_L)$  true (low type)")
    bars_akm = ax.bar(x, df["firm_gap_benchmark"], w, color="#d62728",
                      label=r"AKM gap $\Delta\psi$  (population)")
    bars_gh = ax.bar(x + w, df["g_H"], w, color="#2ca02c",
                     label=r"$g(x_H)$  true (high type)")

    for bars in (bars_gl, bars_akm, bars_gh):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, color="#333333")

    for xi, mid in zip(x, df["avg_true_premium"]):
        ax.plot([xi - 1.6 * w, xi + 1.6 * w], [mid, mid],
                color="black", linewidth=2.4, solid_capstyle="butt", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Firm premium (wage level)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)

    handles, lbls = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=2.4))
    lbls.append("Employment-weighted true premium")
    ax.legend(handles, lbls, fontsize=8.5, loc="upper left", framealpha=0.95,
              ncol=2, columnspacing=1.2, handlelength=1.6)

    fig.suptitle("\n".join(textwrap.wrap(title, width=70)),
                 fontsize=13, weight="bold", y=0.985)
    fig.text(0.5, 0.9, "\n".join(textwrap.wrap(subtitle, width=104)),
             ha="center", va="top", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.85))


def figure_interpretation_gap(scenarios: pd.DataFrame) -> None:
    """Interpretation gap and benchmark gap across scenarios."""
    _premia_bars(
        scenarios,
        title=("Interpretation gap: one AKM premium cannot match both "
               "type-specific premia"),
        subtitle=(r"Per scenario: blue $g(x_L)$, red AKM $\Delta\psi$, green "
                  r"$g(x_H)$; black tick = employment-weighted true premium. The "
                  r"blue-green spread is the interpretation gap; the red-to-tick "
                  r"distance is the benchmark gap."),
    )


figure_interpretation_gap(scenarios)
plt.show()


# %%
def _sweep_over_sigma(value_key: str, gammas: list[float]) -> dict:
    """Analytic quantity (benchmark_gap / two_cov) vs sigma for each gamma."""
    sigmas = np.linspace(0.0, 1.5, 31)
    out = {}
    for g in gammas:
        ys = []
        for s in sigmas:
            p = Params(gamma=float(g), sigma=float(s), lam1=0.2, **_BASE_PARAMS)
            ys.append(type_level_projection(model_cells(p))[value_key])
        out[g] = np.array(ys)
    return {"sigmas": sigmas, "curves": out}


def figure_benchmark_gap() -> None:
    """Benchmark gap vs sorting strength, one line per gamma."""
    gammas = [0.0, 0.5, 1.0]
    sw = _sweep_over_sigma("benchmark_gap", gammas)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0)
    for g in gammas:
        ax.plot(sw["sigmas"], sw["curves"][g], linewidth=2.0,
                label=rf"$\gamma={g:g}$")
    ax.set_xlabel(r"Sorting strength  $\sigma$")
    ax.set_ylabel(r"Benchmark gap  $B = \Delta\psi - \sum_x q(x)\,g(x)$")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(title="Complementarity", fontsize=8)
    title = "Benchmark gap of the AKM firm-effect gap versus worker-firm sorting"
    subtitle = ("The benchmark gap is zero without assortative meetings, although "
                "type-specific discrepancies may remain.")
    fig.suptitle("\n".join(textwrap.wrap(title, width=72)),
                 fontsize=12, weight="bold", y=0.985)
    fig.text(0.5, 0.9, "\n".join(textwrap.wrap(subtitle, width=100)),
             ha="center", va="top", fontsize=8.5, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.86))


figure_benchmark_gap()
plt.show()


# %%
def _true_two_cov(p: Params) -> float:
    """2*Cov(x, y) over the employed distribution pi(x, y) -- pure sorting."""
    cells = model_cells(p)
    x = cells["x_val"].to_numpy(float)
    y = cells["y_val"].to_numpy(float)
    pi = cells["pi"].to_numpy(float)
    ex = float((pi * x).sum())
    ey = float((pi * y).sum())
    return 2.0 * float((pi * (x - ex) * (y - ey)).sum())


def figure_covariance() -> None:
    """Measured 2*Cov(alpha,psi) vs true 2*Cov(x,y) against sorting strength."""
    sigmas = np.linspace(0.0, 1.5, 31)
    gammas = [0.0, 0.5, 1.0]
    colors = {0.0: "#1f77b4", 0.5: "#2ca02c", 1.0: "#d62728"}

    measured, true_curves = {}, {}
    for g in gammas:
        meas, tru = [], []
        for s in sigmas:
            p = Params(gamma=float(g), sigma=float(s), lam1=0.2, **_BASE_PARAMS)
            meas.append(type_level_projection(model_cells(p))["two_cov"])
            tru.append(_true_two_cov(p))
        measured[g] = np.array(meas)
        true_curves[g] = np.array(tru)

    true_common = true_curves[gammas[0]]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    for g in gammas:
        ax.plot(sigmas, measured[g], color=colors[g], linewidth=2.0,
                label=rf"Measured $2\,\mathrm{{Cov}}(\alpha,\psi)$, $\gamma={g:g}$")
    ax.plot(sigmas, true_common, color="black", linewidth=2.2,
            linestyle="--",
            label=r"True $2\,\mathrm{Cov}(x,y)$  (identical for all $\gamma$)")

    ax.set_xlabel(r"Sorting strength  $\sigma$")
    ax.set_ylabel(r"$2\,\mathrm{Cov}$  (covariance term)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Covariance and structural sorting",
                 fontsize=12, weight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))


figure_covariance()
plt.show()


# %%
# 7. 10x10 CELL-RESIDUAL MODEL (full job-ladder solver, N worker/firm types)
# The 10x10 grid is only built here, at the end, right before its own figures.
def _meeting_probs_NxN(x_vals: np.ndarray, y_vals: np.ndarray,
                       sigma: float, mu_pop: np.ndarray) -> np.ndarray:
    """Conditional meeting matrix mu(y_j | x_i) for N types."""
    N_x = len(x_vals)
    N_y = len(y_vals)
    s = np.linspace(-1, 1, N_x)
    t = np.linspace(-1, 1, N_y)
    log_mu = np.log(mu_pop)[None, :] + sigma * s[:, None] * t[None, :]
    log_mu -= log_mu.max(axis=1, keepdims=True)
    mu = np.exp(log_mu)
    mu /= mu.sum(axis=1, keepdims=True)
    return mu


def model_cells_NxN(x_vals: np.ndarray, y_vals: np.ndarray,
                    gamma: float, sigma: float,
                    pop_x: np.ndarray | None = None,
                    mu_pop: np.ndarray | None = None,
                    r: float = 0.05, delta: float = 0.05,
                    lam0: float = 0.5, lam1: float = 0.2,
                    b: float = 0.5, beta: float = 0.5) -> pd.DataFrame:
    """Build the NxN cell table from the full structural job-ladder model.

    Solves the search-and-matching model with on-the-job search for N worker and
    N firm types. Setting lam1 = 0 reproduces the additive closed form.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    N_x = len(x_vals)
    N_y = len(y_vals)
    if pop_x is None:
        pop_x = np.ones(N_x) / N_x
    if mu_pop is None:
        mu_pop = np.ones(N_y) / N_y

    F = x_vals[:, None] + y_vals[None, :] + gamma * x_vals[:, None] * y_vals[None, :]
    mu = _meeting_probs_NxN(x_vals, y_vals, sigma, np.asarray(mu_pop, dtype=float))

    A = r + delta
    u = delta / (lam0 + delta)

    W = np.zeros((N_x, N_y))
    E = np.zeros((N_x, N_y))

    for i in range(N_x):
        mu_i = mu[i]
        F_i = F[i]
        tail_meet = np.array([mu_i[j + 1:].sum() for j in range(N_y)])
        D = A + lam1 * tail_meet

        a = np.zeros(N_y)
        c = np.zeros(N_y)
        for j in range(N_y - 1, -1, -1):
            hi = slice(j + 1, N_y)
            a[j] = (F_i[j] + lam1 * beta * np.sum(mu_i[hi] * a[hi])) / D[j]
            c[j] = (-1.0 + lam1 * beta * np.sum(mu_i[hi] * c[hi])) / D[j]
        denom_rU = 1.0 - lam0 * beta * np.sum(mu_i * c)
        rU_i = (b + lam0 * beta * np.sum(mu_i * a)) / denom_rU
        S_i = a + c * rU_i

        W[i] = F_i - (1.0 - beta) * S_i * D

        e = np.zeros(N_y)
        for j in range(N_y):
            below = e[:j].sum()
            e[j] = (lam0 * mu_i[j] * u + lam1 * mu_i[j] * below) / (
                delta + lam1 * tail_meet[j]
            )
        E[i] = e

    pi_raw = pop_x[:, None] * E
    pi = pi_raw / pi_raw.sum()

    rows = []
    for i in range(N_x):
        for j in range(N_y):
            rows.append({
                "x_label": str(i + 1),
                "y_label": str(j + 1),
                "x_val": x_vals[i],
                "y_val": y_vals[j],
                "w": W[i, j],
                "pi": pi[i, j],
            })
    return pd.DataFrame(rows)


def akm_cell_residuals_NxN(cells: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Per-cell residuals of the employment-weighted additive AKM projection."""
    x_labels = sorted(cells["x_label"].unique(), key=int)
    y_labels = sorted(cells["y_label"].unique(), key=int)
    N_x = len(x_labels)
    N_y = len(y_labels)

    w = cells["w"].to_numpy(dtype=float)
    pi = cells["pi"].to_numpy(dtype=float)

    n = len(cells)
    X = np.zeros((n, 1 + (N_x - 1) + (N_y - 1)))
    X[:, 0] = 1.0
    for k, xl in enumerate(x_labels[1:]):
        X[:, 1 + k] = (cells["x_label"] == xl).to_numpy(dtype=float)
    for k, yl in enumerate(y_labels[1:]):
        X[:, 1 + (N_x - 1) + k] = (cells["y_label"] == yl).to_numpy(dtype=float)

    W_mat = np.diag(pi)
    coef = np.linalg.solve(X.T @ W_mat @ X, X.T @ W_mat @ w)

    fitted = X @ coef
    residual = w - fitted

    out = cells.copy()
    out["scenario"] = scenario
    out["fitted_wage"] = fitted
    out["residual"] = residual
    return out


# %%
# 8. CELL-RESIDUAL 3D FIGURES (10x10 model)
def _residual_table(sigma: float, lam1: float, gamma: float,
                    label: str) -> pd.DataFrame:
    """One 10x10 cell-residual table for a single (sigma, lam1, gamma) config."""
    cells = model_cells_NxN(X_GRID, Y_GRID, gamma=gamma, sigma=sigma, lam1=lam1)
    scenario = f"sigma={sigma:.2f}, lam1={lam1:.2f}, gamma={gamma:.2f}"
    table = akm_cell_residuals_NxN(cells, scenario)
    table = table.copy()
    table["scenario"] = label
    return table


def _residual_3d_panels(table: pd.DataFrame, panels: list[str], title: str,
                        subtitle: str, note: str, grid: tuple[int, int]) -> None:
    """Multi-panel 3D bar chart of population AKM residuals by type."""
    x_labels = sorted(table["x_label"].unique(), key=int)
    y_labels = sorted(table["y_label"].unique(), key=int)
    N_x = len(x_labels)
    N_y = len(y_labels)

    elev, azim = 25, -55
    bar_w = 0.7
    n_rows, n_cols = grid
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for panel_idx, scen in enumerate(panels):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx + 1, projection="3d")
        sub = table[table["scenario"] == scen]

        vmax = max(float(sub["residual"].abs().max()), 1e-6)
        zlim = (-1.6 * vmax, 1.25 * vmax)

        xpos, ypos, dz = [], [], []
        for ix, fl in enumerate(y_labels):
            for iy, wl in enumerate(x_labels):
                m = (sub["y_label"] == fl) & (sub["x_label"] == wl)
                xpos.append(ix - bar_w / 2.0)
                ypos.append(iy - bar_w / 2.0)
                dz.append(float(sub.loc[m, "residual"].iloc[0]))
        xpos = np.array(xpos)
        ypos = np.array(ypos)
        dz = np.array(dz)

        pos_grey, neg_grey = "0.90", "0.55"
        bar_colors = [neg_grey if v < 0 else pos_grey for v in dz]

        z0 = np.minimum(dz, 0.0)
        height = np.where(np.abs(dz) < 1e-9, 1e-9, np.abs(dz))
        ax.bar3d(xpos, ypos, z0, bar_w, bar_w, height,
                 color=bar_colors, edgecolor="black", linewidth=0.3,
                 shade=False, zsort="max")

        lo, hi = -0.5, N_y - 0.5 + 0.4
        lo2, hi2 = -0.5, N_x - 0.5 + 0.4
        ax.plot([lo, hi, hi, lo, lo], [lo2, lo2, hi2, hi2, lo2], [0] * 5,
                color="0.35", linewidth=0.7, zorder=10)

        ax.set_xticks(range(N_y))
        ax.set_xticklabels([])
        ax.set_yticks(range(N_x))
        ax.set_yticklabels([])
        for ix, lab in enumerate(y_labels):
            ax.text(ix, -0.7, 0, lab, ha="center", va="center", fontsize=7,
                    color="red")
        for iy, lab in enumerate(x_labels):
            ax.text(N_y - 0.5 + 0.5, iy, 0, lab, ha="center", va="center",
                    fontsize=7, color="red")

        ax.set_xlabel("Firm type", fontsize=8, labelpad=10)
        ax.set_ylabel("Worker type", fontsize=8, labelpad=10)
        ax.set_zlabel("", fontsize=7, labelpad=1)
        ax.set_zlim(*zlim)
        ax.set_title(scen, fontsize=10, pad=2)
        ax.view_init(elev=elev, azim=azim)

        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor("none")
            axis._axinfo["grid"].update(color="0.55", linestyle=":", linewidth=0.5)

    fig.suptitle(title, fontsize=14, weight="bold", y=0.98)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10)
    wrapped_note = "\n".join(textwrap.wrap(note, width=150))
    fig.text(0.5, 0.01, wrapped_note, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10,
                        wspace=0.02, hspace=0.12)


# %%
def figure_card_parallel() -> None:
    """3D residuals, gamma sweep, sorting and OJS off."""
    panels = [rf"$\gamma = {g:.2f}$" for g in GAMMAS_FIG1]
    tables = [_residual_table(0.0, 0.0, g, rf"$\gamma = {g:.2f}$")
              for g in GAMMAS_FIG1]
    table = pd.concat(tables, ignore_index=True)
    note = (r"Bars report population mean residuals from the employment-weighted "
            r"additive AKM projection, by true worker and firm type. "
            r"Sorting ($\sigma=0$) and on-the-job search ($\lambda_1=0$) are "
            r"both off. The z-axis is scaled to each panel's residual range.")
    _residual_3d_panels(
        table, panels=panels,
        title="AKM residuals by worker and firm type (10x10 model)",
        subtitle=r"Production complementarity only: $\gamma \in \{0.00, 0.25, 0.50, 0.75\}$",
        note=note, grid=(2, 2),
    )


figure_card_parallel()
plt.show()


# %%
def figure_four_scenarios() -> None:
    """3D residuals, additive benchmark + S1-S4."""
    configs = [
        (0.0, 0.0, 0.0,
         r"Additive benchmark (Sanity): $\gamma = 0$, $\sigma = 0$, $\lambda_1 = 0$"),
        (0.0, 0.0, BASELINE_LAM1, r"S1: $\gamma = 0$, $\sigma = 0$ ($\lambda_1 = 0.20$)"),
        (0.0, 0.5, BASELINE_LAM1, r"S2: $\gamma = 0.5$, $\sigma = 0$"),
        (1.0, 0.0, BASELINE_LAM1, r"S3: $\gamma = 0$, $\sigma = 1$"),
        (1.0, 0.5, BASELINE_LAM1, r"S4: $\gamma = 0.5$, $\sigma = 1$"),
    ]
    panels = [label for _s, _g, _l, label in configs]
    tables = [_residual_table(sigma, lam1, gamma, label)
              for sigma, gamma, lam1, label in configs]
    table = pd.concat(tables, ignore_index=True)
    note = (r"Bars report population mean residuals from the employment-weighted "
            r"additive AKM projection, by true worker and firm type. "
            r"On-the-job search is off only in the additive benchmark "
            rf"($\lambda_1=0$); the other panels keep $\lambda_1={BASELINE_LAM1:.2f}$. "
            r"Panels S1-S4 are the main scenarios (no sorting / sorting x "
            r"$\gamma\in\{0,\,0.5\}$); S1 isolates the pure OJS effect "
            r"($\gamma=0,\,\sigma=0,\,\lambda_1>0$). "
            r"The z-axis is scaled to each panel's residual range.")
    _residual_3d_panels(
        table, panels=panels,
        title="AKM residuals across the main scenarios (10x10 model)",
        subtitle=(r"Additive benchmark plus S1-S4: $\sigma \in \{0, 1\}$ x "
                  r"$\gamma \in \{0, 0.5\}$, OJS off only in the additive benchmark"),
        note=note, grid=(2, 3),
    )


figure_four_scenarios()
plt.show()

