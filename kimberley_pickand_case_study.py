"""
Paper-oriented Kimberley stream geochemistry case study.

This code generates:
    - kappa(C,U), mean_align, max_align
    - extended LaTeX tables
    - performance tables
    - one multi-panel summary figure per stream/element
    - support for Cu, Zn, Ni, Pb
    - CSV summaries

 
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh, solve
from scipy.spatial.distance import cdist

KIMBERLEY_FILE = "kimberley_pickand.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_SILL = 0.9
NUGGET = 0.10
BASE_RANGE = 9000.0

STREAM_NUMBERS = [1, 2, 3, 4]
ELEMENTS = ["Cu_40PNB_p", "Zn_40PNB_p", "Ni_40PNB_p", "Pb_40PNB_p"]
M_PRED = 20
RANDOM_STATE = 1234


def sanitize_name(x: str) -> str:
    return x.replace("/", "_").replace(" ", "_")


def spherical_covariance(locs: np.ndarray, sill: float, rng: float) -> np.ndarray:
    dists = cdist(locs, locs, metric="euclidean")
    h = dists / rng
    C = np.zeros_like(h)
    mask = h <= 1.0
    h_m = h[mask]
    C[mask] = sill * (1.0 - 1.5 * h_m + 0.5 * (h_m ** 3))
    return C


def spherical_covariance_vector(s: np.ndarray, s0: np.ndarray, sill: float, rng: float) -> np.ndarray:
    d = np.linalg.norm(s - s0, axis=1)
    h = d / rng
    c0 = np.zeros_like(h)
    mask = h <= 1.0
    h_m = h[mask]
    c0[mask] = sill * (1.0 - 1.5 * h_m + 0.5 * (h_m ** 3))
    return c0


def build_prediction_subspace_basis(s: np.ndarray, s0: np.ndarray,
                                    sill: float = BASE_SILL,
                                    rng_param: float = BASE_RANGE) -> np.ndarray:
    n = s.shape[0]
    cols = [np.ones(n)]
    for loc in s0:
        cols.append(spherical_covariance_vector(s, loc, sill=sill, rng=rng_param))
    Uraw = np.column_stack(cols)
    Q, R = np.linalg.qr(Uraw)
    diagR = np.abs(np.diag(R)) if R.size else np.array([0.0])
    tol = 1e-12 * max(np.max(diagR), 1.0)
    keep = diagR > tol
    return Q[:, keep]


def projector_onto_subspace(BU: np.ndarray) -> np.ndarray:
    return BU @ BU.T


@dataclass
class PerturbedCovariances:
    mild: np.ndarray
    moderate: np.ndarray
    high: np.ndarray


@dataclass
class SpectralDiagnostics:
    min_eig: float
    neg_count: int
    neg_sum_abs: float
    mean_align: float
    max_align: float
    kappa_CU: float


@dataclass
class KrigingResults:
    rmse: float
    mae: float
    neg_variances: int
    min_variance: float


def mild_boundary_structure(C: np.ndarray, s: np.ndarray, reduction: float = 0.6) -> np.ndarray:
    x = s[:, 0]
    smin, smax = x.min(), x.max()
    third = (smax - smin) / 3.0
    mid_left = smin + third
    mid_right = smin + 2 * third
    left_idx = np.where(x < mid_left)[0]
    mid_idx = np.where((x >= mid_left) & (x <= mid_right))[0]
    right_idx = np.where(x > mid_right)[0]
    Cpert = C.copy()
    denom = max(smax - smin, 1.0)
    for i in mid_idx:
        for j in np.concatenate([left_idx, right_idx]):
            h = abs(x[i] - x[j])
            mod = 1.0 - reduction * (0.5 + 0.5 * np.sin(2 * np.pi * h / denom))
            Cpert[i, j] *= mod
            Cpert[j, i] = Cpert[i, j]
    return Cpert


def moderate_patch_structure(C: np.ndarray, s: np.ndarray,
                             reduction: float = 0.7,
                             leftscale: float = 0.9,
                             rightscale: float = 1.1) -> np.ndarray:
    x = s[:, 0]
    smid = 0.5 * (x.min() + x.max())
    left_idx = np.where(x <= smid)[0]
    right_idx = np.where(x > smid)[0]
    Cpert = C.copy()
    for i in left_idx:
        for j in left_idx:
            Cpert[i, j] *= leftscale
    for i in right_idx:
        for j in right_idx:
            Cpert[i, j] *= rightscale
    for i in left_idx:
        for j in right_idx:
            Cpert[i, j] *= (1.0 - reduction)
            Cpert[j, i] = Cpert[i, j]
    return Cpert


def high_local_inconsistency_structure(C: np.ndarray, s: np.ndarray,
                                       ntriples: int | None = None,
                                       magnitude: float = 0.25,
                                       random_state: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = s.shape[0]
    if ntriples is None:
        ntriples = max(n // 2, 1)
    Cpert = C.copy()
    for _ in range(ntriples):
        i, j, k = rng.choice(n, size=3, replace=False)
        deltaij = magnitude * rng.uniform(-1.0, 1.0)
        deltaik = magnitude * rng.uniform(-1.0, 1.0)
        deltajk = magnitude * rng.uniform(-1.0, 1.0)
        for a, b, delta in [(i, j, deltaij), (i, k, -deltaik), (j, k, deltajk)]:
            Cpert[a, b] += delta
            Cpert[b, a] = Cpert[a, b]
    return Cpert


def add_low_rank_negative(C: np.ndarray, BU: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    n = C.shape[0]
    PU = projector_onto_subspace(BU)
    I = np.eye(n)
    v = rng.standard_normal(n)
    u = (I - PU) @ v
    normu = np.linalg.norm(u)
    if normu < 1e-8:
        u = v
        normu = np.linalg.norm(u)
    u = u / normu
    return C - alpha * np.outer(u, u)


def build_perturbed_covariances(Ctrue: np.ndarray, s: np.ndarray, BU: np.ndarray,
                                random_state: int = RANDOM_STATE) -> PerturbedCovariances:
    rng = np.random.default_rng(random_state)
    Cmild_struct = mild_boundary_structure(Ctrue, s)
    Cmod_struct = moderate_patch_structure(Ctrue, s)
    Chigh_struct = high_local_inconsistency_structure(Ctrue, s, random_state=random_state)
    Cmild = add_low_rank_negative(Cmild_struct, BU, alpha=0.05, rng=rng)
    Cmoderate = add_low_rank_negative(Cmod_struct, BU, alpha=0.12, rng=rng)
    Chigh = add_low_rank_negative(Chigh_struct, BU, alpha=0.15, rng=rng)
    return PerturbedCovariances(Cmild, Cmoderate, Chigh)


def psd_projection_global(C: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    vals, vecs = eigh(C)
    vals_clipped = np.maximum(vals, eps)
    Cpsd = vecs @ np.diag(vals_clipped) @ vecs.T
    return 0.5 * (Cpsd + Cpsd.T)


def psd_projection_subspace(C: np.ndarray, BU: np.ndarray, eps: float = 0.0) -> np.ndarray:
    CU = BU.T @ C @ BU
    vals, vecs = eigh(CU)
    vals_clipped = np.maximum(vals, eps)
    CUpsd = vecs @ np.diag(vals_clipped) @ vecs.T
    deltaCU = CUpsd - CU
    deltaC = BU @ deltaCU @ BU.T
    return C + deltaC


def spectral_diagnostics(C: np.ndarray, BU: np.ndarray) -> Tuple[SpectralDiagnostics, np.ndarray]:
    vals, vecs = eigh(C)
    negmask = vals < 0.0
    negvals = vals[negmask]
    negvecs = vecs[:, negmask]
    if negvals.size == 0:
        return SpectralDiagnostics(
            min_eig=float(vals.min()),
            neg_count=0,
            neg_sum_abs=0.0,
            mean_align=0.0,
            max_align=0.0,
            kappa_CU=0.0,
        ), vals
    PU = projector_onto_subspace(BU)
    norms = np.linalg.norm(PU @ negvecs, axis=0)
    meanalign = float(norms.mean())
    maxalign = float(norms.max())
    kappa_CU = float(np.max(-negvals * norms ** 2))
    return SpectralDiagnostics(
        min_eig=float(vals.min()),
        neg_count=int(negvals.size),
        neg_sum_abs=float(np.sum(np.abs(negvals))),
        mean_align=meanalign,
        max_align=maxalign,
        kappa_CU=kappa_CU,
    ), vals


def build_ok_system(C: np.ndarray, ones: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = C
    K[:n, n] = ones
    K[n, :n] = ones
    K[n, n] = 0.0
    return K


def solve_ok(C: np.ndarray, c0: np.ndarray, C0: float, vartol: float = 1e-12) -> Tuple[np.ndarray, float, bool]:
    n = C.shape[0]
    ones = np.ones(n)
    K = build_ok_system(C, ones)
    rhs = np.zeros(n + 1)
    rhs[:n] = c0
    rhs[n] = 1.0
    try:
        sol = solve(K, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    w = sol[:n]
    var = C0 - 2.0 * np.dot(c0, w) + w @ C @ w
    is_neg = bool(var < -vartol)
    if 0.0 > var > -vartol:
        var = 0.0
    return w, float(var), is_neg


def kriging_experiment(Cindef: np.ndarray, Ctrue: np.ndarray,
                       s: np.ndarray, s0: np.ndarray, z: np.ndarray,
                       sill: float = BASE_SILL,
                       rng_param: float = BASE_RANGE):
    m = s0.shape[0]
    zref = np.zeros(m)
    for j, loc in enumerate(s0):
        c0true = spherical_covariance_vector(s, loc, sill=sill, rng=rng_param)
        C0true = sill + NUGGET
        wref, _, _ = solve_ok(Ctrue, c0true, C0true)
        zref[j] = wref @ z
    BU = build_prediction_subspace_basis(s, s0, sill=sill, rng_param=rng_param)
    Craw = Cindef
    Cglobal = psd_projection_global(Cindef)
    Csub = psd_projection_subspace(Cindef, BU)
    methods = {"raw_indef": Craw, "global_psd": Cglobal, "subspace_psd": Csub}
    summary = {}
    detailed = {}
    for name, Cuse in methods.items():
        preds = np.zeros(m)
        vars_ = np.zeros(m)
        negcount = 0
        minvar = np.inf
        for j, loc in enumerate(s0):
            c0 = spherical_covariance_vector(s, loc, sill=sill, rng=rng_param)
            C0 = sill + NUGGET
            w, var, isneg = solve_ok(Cuse, c0, C0)
            preds[j] = w @ z
            vars_[j] = var
            if isneg:
                negcount += 1
            minvar = min(minvar, var)
        errors = preds - zref
        summary[name] = KrigingResults(
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            mae=float(np.mean(np.abs(errors))),
            neg_variances=int(negcount),
            min_variance=float(minvar),
        )
        detailed[name] = {"preds": preds, "vars": vars_, "errors": errors}
    detailed["reference"] = {"preds": zref}
    return summary, detailed


def load_kimberley_points() -> pd.DataFrame:
    df = pd.read_csv(KIMBERLEY_FILE)
    df = df.dropna(subset=["X", "Y"])
    assay_cols = [c for c in df.columns if c.endswith("_40PNB_p")]
    for c in assay_cols:
        df.loc[df[c] < 0, c] = np.nan
    return df


def choose_available_elements(df: pd.DataFrame, requested: List[str]) -> List[str]:
    return [e for e in requested if e in df.columns]


def run_stream_element_case(df: pd.DataFrame, stream_no: int, element: str, m_pred: int = M_PRED):
    df_stream = df[df["StmNo"] == stream_no].copy()
    if df_stream.empty:
        raise ValueError(f"No samples found for stream {stream_no}.")
    if element not in df_stream.columns:
        raise KeyError(f"Element '{element}' not found.")
    df_stream = df_stream.dropna(subset=[element])
    if df_stream.empty:
        raise ValueError(f"All samples for stream {stream_no} are missing for '{element}'.")
    coords = df_stream[["X", "Y"]].to_numpy()
    z = df_stream[element].to_numpy()
    Ctrue = spherical_covariance(coords, sill=BASE_SILL, rng=BASE_RANGE)
    Ctrue = Ctrue + NUGGET * np.eye(Ctrue.shape[0])
    n = coords.shape[0]
    if n <= m_pred:
        s0 = coords.copy()
    else:
        idx = np.linspace(0, n - 1, m_pred, dtype=int)
        s0 = coords[idx, :]
    BU = build_prediction_subspace_basis(coords, s0, sill=BASE_SILL, rng_param=BASE_RANGE)
    perturbed = build_perturbed_covariances(Ctrue, coords, BU)
    scenarios = {"mild": perturbed.mild, "moderate": perturbed.moderate, "high": perturbed.high}
    diagnostics = {}
    spectra = {}
    performance = {}
    details = {}
    for scen, Cindef in scenarios.items():
        diagnostics[scen], spectra[scen] = spectral_diagnostics(Cindef, BU)
        performance[scen], details[scen] = kriging_experiment(Cindef, Ctrue, coords, s0, z)
    return coords, s0, z, diagnostics, spectra, performance, details


def format_extended_eigen_table(diagnostics, stream_no, element) -> str:
    rows = [
        ("Mild (Boundary-type)", "mild"),
        ("Moderate (Patchwise-type)", "moderate"),
        ("High (Local inconsistency)", "high"),
    ]
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Extended spectral diagnostics for Stream {stream_no} and {element}.}}")
    lines.append(f"\\label{{tab:extended_eigen_stream_{stream_no}_{sanitize_name(element)}}}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append("Scenario & Min eigenvalue & Neg. count & $\\sum |\\lambda_i^-|$ & Mean align. & Max align. & $\\kappa(C,U)$ \\\\")
    lines.append("\\hline")
    for label, key in rows:
        d = diagnostics[key]
        lines.append(
            f"{label} & {d.min_eig:.3f} & {d.neg_count:d} & {d.neg_sum_abs:.3f} & {d.mean_align:.3f} & {d.max_align:.3f} & {d.kappa_CU:.4f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_performance_table(results, m_pred, stream_no, element) -> str:
    scenario_order = ["mild", "moderate", "high"]
    scenario_labels = {"mild": "Mild", "moderate": "Moderate", "high": "High"}
    method_order = ["raw_indef", "global_psd", "subspace_psd"]
    method_labels = {"raw_indef": "Raw indefinite", "global_psd": "Global PSD", "subspace_psd": "Subspace PSD"}
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Kriging performance for Stream {stream_no} and {element}.}}")
    lines.append(f"\\label{{tab:performance_stream_{stream_no}_{sanitize_name(element)}}}")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\hline")
    lines.append("Scenario & Method & RMSE & MAE & Neg. vars. & Min var. \\\\")
    lines.append("\\hline")
    for scen in scenario_order:
        first = True
        for method in method_order:
            r = results[scen][method]
            scenlabel = scenario_labels[scen] if first else ""
            first = False
            lines.append(
                f"{scenlabel} & {method_labels[method]} & {r.rmse:.3f} & {r.mae:.3f} & {r.neg_variances}/{m_pred} & {r.min_variance:.3f} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_kappa_summary_table(summary_df: pd.DataFrame, element: str) -> str:
    sub = summary_df[summary_df["element"] == element].copy()
    scenario_order = ["mild", "moderate", "high"]
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{$\\kappa(C,U)$, mean alignment and max alignment by stream for {element}.}}")
    lines.append(f"\\label{{tab:kappa_summary_{sanitize_name(element)}}}")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\hline")
    lines.append("Stream & Scenario & Mean align. & Max align. & $\\kappa(C,U)$ \\\\")
    lines.append("\\hline")
    for stream_no in sorted(sub["stream_no"].unique()):
        part = sub[sub["stream_no"] == stream_no]
        first = True
        for scen in scenario_order:
            row = part[part["scenario"] == scen].iloc[0]
            stream_label = str(stream_no) if first else ""
            first = False
            lines.append(
                f"{stream_label} & {scen.capitalize()} & {row['mean_align']:.3f} & {row['max_align']:.3f} & {row['kappa_CU']:.4f} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def save_summary_figure(spectra, details, stream_no: int, element: str):
    safe_element = sanitize_name(element)
    scenarios = ["mild", "moderate", "high"]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 10))
    for i, scen in enumerate(scenarios):
        eigs = np.sort(spectra[scen])
        ax = axes[i, 0]
        ax.plot(np.arange(1, len(eigs) + 1), eigs, marker="o", ms=2, lw=1)
        ax.axhline(0.0, color="red", ls="--", lw=1)
        if i == 0:
            ax.set_title("Spectrum")
        ax.set_ylabel(scen.capitalize())
        ax.grid(alpha=0.3)

        ax = axes[i, 1]
        raw_vars = details[scen]["raw_indef"]["vars"]
        sub_vars = details[scen]["subspace_psd"]["vars"]
        bins = min(12, max(len(raw_vars) // 2, 5))
        ax.hist(raw_vars, bins=bins, alpha=0.6, label="Raw")
        ax.hist(sub_vars, bins=bins, alpha=0.6, label="Subspace")
        ax.axvline(0.0, color="red", ls="--", lw=1)
        if i == 0:
            ax.set_title("Variance histograms")
            ax.legend()
        ax.grid(alpha=0.25)

        ax = axes[i, 2]
        raw = details[scen]["raw_indef"]["preds"]
        sub = details[scen]["subspace_psd"]["preds"]
        mn = min(raw.min(), sub.min())
        mx = max(raw.max(), sub.max())
        ax.scatter(raw, sub, alpha=0.8)
        ax.plot([mn, mx], [mn, mx], color="red", ls="--", lw=1)
        if i == 0:
            ax.set_title("Raw vs Subspace")
        ax.grid(alpha=0.3)

    axes[2, 0].set_xlabel("Eigenvalue index")
    axes[2, 1].set_xlabel("Kriging variance")
    axes[2, 2].set_xlabel("Raw indefinite prediction")
    axes[0, 0].set_ylabel("Mild")
    axes[1, 0].set_ylabel("Moderate")
    axes[2, 0].set_ylabel("High")
    axes[0, 2].set_ylabel("Subspace PSD prediction")
    axes[1, 2].set_ylabel("Subspace PSD prediction")
    axes[2, 2].set_ylabel("Subspace PSD prediction")
    fig.suptitle(f"Diagnostics summary: Stream {stream_no}, {element}")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"summary_stream_{stream_no}_{safe_element}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    path = os.path.join(OUTPUT_DIR, f"summary_stream_{stream_no}_{safe_element}.eps")
    fig.savefig(path, format="eps", dpi=600, bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_kimberley_points()
    elements = choose_available_elements(df, ELEMENTS)
    if not elements:
        raise ValueError("None of the requested elements (Cu, Zn, Ni, Pb) are present in the dataset.")

    summary_rows = []
    perf_rows = []

    for element in elements:
        print("\n" + "#" * 90)
        print(f"ELEMENT {element}")
        print("#" * 90)

        for stream_no in STREAM_NUMBERS:
            print("\n" + "=" * 90)
            print(f"STREAM {stream_no} | ELEMENT {element}")
            print("=" * 90)
            try:
                coords, s0, z, diagnostics, spectra, performance, details = run_stream_element_case(
                    df, stream_no, element, m_pred=M_PRED
                )

                print("\nEIGENVALUE DIAGNOSTICS")
                print("-" * 90)
                for scen, diag in diagnostics.items():
                    print(f"\nScenario: {scen}")
                    print(f"  min eigenvalue : {diag.min_eig:.4f}")
                    print(f"  negative count : {diag.neg_count}")
                    print(f"  sum |neg eig|  : {diag.neg_sum_abs:.4f}")
                    print(f"  mean alignment : {diag.mean_align:.3f}")
                    print(f"  max alignment  : {diag.max_align:.3f}")
                    print(f"  kappa(C,U)     : {diag.kappa_CU:.4f}")
                    summary_rows.append({
                        "stream_no": stream_no,
                        "element": element,
                        "scenario": scen,
                        **asdict(diag),
                    })

                print("\nKRIGING PERFORMANCE")
                print("-" * 90)
                for scen, methods in performance.items():
                    print(f"\nScenario: {scen}")
                    for method_name, res in methods.items():
                        print(
                            f"{method_name:20s} RMSE={res.rmse:.3f} MAE={res.mae:.3f} "
                            f"NegVars={res.neg_variances} MinVar={res.min_variance:.3f}"
                        )
                        perf_rows.append({
                            "stream_no": stream_no,
                            "element": element,
                            "scenario": scen,
                            "method": method_name,
                            **asdict(res),
                        })

                safe_element = sanitize_name(element)
                eig_tex = format_extended_eigen_table(diagnostics, stream_no, element)
                perf_tex = format_performance_table(performance, s0.shape[0], stream_no, element)

                with open(os.path.join(OUTPUT_DIR, f"stream_{stream_no}_{safe_element}_extended_eigen.tex"), "w", encoding="utf-8") as f:
                    f.write(eig_tex)
                with open(os.path.join(OUTPUT_DIR, f"stream_{stream_no}_{safe_element}_performance.tex"), "w", encoding="utf-8") as f:
                    f.write(perf_tex)

                save_summary_figure(spectra, details, stream_no, element)

            except Exception as e:
                print(f"FAILED: Stream {stream_no}, element {element}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    perf_df = pd.DataFrame(perf_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "kimberley_paper_spectral_summary.csv"), index=False)
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "kimberley_paper_performance_summary.csv"), index=False)

    if not summary_df.empty:
        for element in sorted(summary_df["element"].unique()):
            tex = format_kappa_summary_table(summary_df, element)
            with open(os.path.join(OUTPUT_DIR, f"kappa_summary_{sanitize_name(element)}.tex"), "w", encoding="utf-8") as f:
                f.write(tex)

    print("\n" + "=" * 90)
    print("FILES WRITTEN TO output/")
    print("- Paper spectral summary CSV")
    print("- Paper performance summary CSV")
    print("- Extended eigenvalue tables per stream/element")
    print("- Performance tables per stream/element")
    print("- Kappa summary tables per element")
    print("- One multi-panel summary figure per stream/element")
    print("=" * 90)


if __name__ == "__main__":
    main()