"""
Simulation study for:

Maturana, F.J.
'Non-Positive-Definite Covariance Structures in Geostatistics:
Theory, Diagnostics, and Implications for Kriging'.

Purpose
-------
This program investigates whether departures from positive
semi-definiteness necessarily degrade ordinary kriging performance.

Methodology
-----------
1. Generate synthetic spatial data from a valid covariance model.
2. Construct a prediction subspace U.
3. Introduce controlled covariance inconsistency through:
      - boundary perturbations,
      - patchwise perturbations,
      - local inconsistency perturbations,
      - low-rank negative components concentrated in U⊥.
4. Compare:
      - raw indefinite covariance matrices,
      - global PSD projection,
      - prediction-subspace PSD projection.
5. Evaluate spectral diagnostics and kriging performance.

Outputs
-------
The program produces:
    - eigenvalue diagnostics,
    - prediction-subspace alignment measures,
    - kriging safety indices,
    - RMSE and MAE summaries,
    - LaTeX tables for direct inclusion in the manuscript.
"""

import numpy as np
from numpy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Tuple, Dict

# ----------------------------------------------------------------------
# Core covariance and design
# ----------------------------------------------------------------------

def covariance_vector_exp(s: np.ndarray,
                          s0: float,
                          sill: float = 1.0,
                          rng_param: float = 2.0) -> np.ndarray:
    d = np.abs(s - s0)
    return sill * np.exp(-d / rng_param)


def exponential_covariance(locations: np.ndarray,
                           sill: float = 1.0,
                           rng: float = 2.0) -> np.ndarray:
    locs = locations.reshape(-1, 1)
    dists = cdist(locs, locs, metric="euclidean")
    return sill * np.exp(-dists / rng)


def covariance_vector_sph(s: np.ndarray,
                          s0: float,
                          sill: float = 1.0,
                          rng_param: float = 2.0) -> np.ndarray:
    d = np.abs(s - s0)
    h = d / rng_param
    c0 = np.zeros_like(h)
    mask = h <= 1.0
    h_m = h[mask]
    c0[mask] = sill * (1.0 - 1.5 * h_m + 0.5 * (h_m**3))
    return c0


def spherical_covariance(locations: np.ndarray,
                         sill: float = 1.0,
                         rng: float = 2.0) -> np.ndarray:
    locs = locations.reshape(-1, 1)
    dists = cdist(locs, locs, metric="euclidean")
    h = dists / rng
    C = np.zeros_like(h)
    mask = h <= 1.0
    h_m = h[mask]
    C[mask] = sill * (1.0 - 1.5 * h_m + 0.5 * (h_m**3))
    return C


def generate_design_and_prediction(n: int = 30,
                                   m: int = 10,
                                   domain: Tuple[float, float] = (0.0, 10.0),
                                   random_state: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    s = np.sort(rng.uniform(domain[0], domain[1], size=n))
    s0 = np.linspace(domain[0], domain[1], m)
    return s, s0


def simulate_field(s: np.ndarray,
                   sill: float = 1.0,
                   rng_param: float = 2.0,
                   random_state: int = 123,
                   cov_model: str = "exp") -> Tuple[np.ndarray, np.ndarray]:
    if cov_model == "exp":
        C_true = exponential_covariance(s, sill=sill, rng=rng_param)
    elif cov_model == "sph":
        C_true = spherical_covariance(s, sill=sill, rng=rng_param)
    else:
        raise ValueError(f"Unknown cov_model '{cov_model}' (use 'exp' or 'sph').")

    C_true_jitter = C_true + 1e-10 * np.eye(len(s))
    rng = np.random.default_rng(random_state)
    L = np.linalg.cholesky(C_true_jitter)
    z = L @ rng.standard_normal(len(s))
    return z, C_true


# ----------------------------------------------------------------------
# Structural (PSD) perturbations
# ----------------------------------------------------------------------

def mild_boundary_structure(C: np.ndarray,
                            s: np.ndarray,
                            reduction: float = 0.6) -> np.ndarray:
    """
    Mild boundary-like structural perturbation; kept reasonably PSD.
    """
    s_min, s_max = s.min(), s.max()
    third = (s_max - s_min) / 3.0
    mid_left = s_min + third
    mid_right = s_min + 2 * third

    left_idx = np.where(s < mid_left)[0]
    mid_idx = np.where((s >= mid_left) & (s <= mid_right))[0]
    right_idx = np.where(s > mid_right)[0]

    C_pert = C.copy()

    for i in mid_idx:
        for j in np.concatenate([left_idx, right_idx]):
            h = abs(s[i] - s[j])
            mod = 1.0 - reduction * (0.5 + 0.5 * np.sin(2 * np.pi * h / (s_max - s_min)))
            C_pert[i, j] *= mod
            C_pert[j, i] = C_pert[i, j]

    return C_pert


def moderate_patch_structure(C: np.ndarray,
                             s: np.ndarray,
                             reduction: float = 0.7,
                             left_scale: float = 0.9,
                             right_scale: float = 1.1) -> np.ndarray:
    """
    Moderate patchwise structural perturbation (still PSD-ish).
    """
    s_mid = 0.5 * (s.min() + s.max())
    left_idx = np.where(s <= s_mid)[0]
    right_idx = np.where(s > s_mid)[0]

    C_pert = C.copy()

    for i in left_idx:
        for j in left_idx:
            C_pert[i, j] *= left_scale

    for i in right_idx:
        for j in right_idx:
            C_pert[i, j] *= right_scale

    for i in left_idx:
        for j in right_idx:
            C_pert[i, j] *= (1.0 - reduction)
            C_pert[j, i] = C_pert[i, j]

    return C_pert


def high_local_inconsistency_structure(C: np.ndarray,
                                       s: np.ndarray,
                                       n_triples: int = None,
                                       magnitude: float = 0.25,
                                       random_state: int = 1234) -> np.ndarray:
    """
    High local inconsistency structure; will later also get a low-rank
    negative component.
    """
    rng = np.random.default_rng(random_state)
    n = len(s)
    if n_triples is None:
        n_triples = max(n // 2, 1)

    C_pert = C.copy()

    for _ in range(n_triples):
        idx = rng.choice(n, size=3, replace=False)
        i, j, k = idx

        delta_ij = magnitude * rng.uniform(-1.0, 1.0)
        delta_ik = magnitude * rng.uniform(-1.0, 1.0)
        delta_jk = magnitude * rng.uniform(-1.0, 1.0)

        for (a, b, delta) in [(i, j, delta_ij),
                              (i, k, -delta_ik),
                              (j, k, delta_jk)]:
            C_pert[a, b] += delta
            C_pert[b, a] = C_pert[a, b]

    return C_pert


# ----------------------------------------------------------------------
# Prediction subspace and controlled indefiniteness
# ----------------------------------------------------------------------

def build_prediction_subspace_basis(s: np.ndarray,
                                    s0: np.ndarray,
                                    sill: float = 1.0,
                                    rng_param: float = 2.0,
                                    cov_model: str = "exp") -> np.ndarray:
    n = len(s)
    ones = np.ones(n)
    cols = [ones]

    for loc in s0:
        if cov_model == "exp":
            c0 = covariance_vector_exp(s, loc, sill=sill, rng_param=rng_param)
        elif cov_model == "sph":
            c0 = covariance_vector_sph(s, loc, sill=sill, rng_param=rng_param)
        else:
            raise ValueError(f"Unknown cov_model '{cov_model}'")
        cols.append(c0)

    U_raw = np.column_stack(cols)
    Q, R = np.linalg.qr(U_raw)
    diag_R = np.abs(np.diag(R))
    tol = 1e-12 * np.max(diag_R)
    keep = diag_R > tol
    B_U = Q[:, keep]
    return B_U


def projector_onto_subspace(B_U: np.ndarray) -> np.ndarray:
    return B_U @ B_U.T


def add_low_rank_negative(C: np.ndarray,
                          B_U: np.ndarray,
                          alpha: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Add a rank-1 negative term -alpha u u^T with u chosen to be mostly
    orthogonal to U (so indefiniteness is nearly prediction-irrelevant).
    """
    n = C.shape[0]
    P_U = projector_onto_subspace(B_U)
    I = np.eye(n)

    # Random direction mostly in U^\perp
    v = rng.standard_normal(n)
    u = (I - P_U) @ v
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-8:
        # Fallback: use raw v
        u = v
        norm_u = np.linalg.norm(u)
    u = u / norm_u

    C_indef = C - alpha * np.outer(u, u)
    return C_indef


@dataclass
class PerturbedCovariances:
    mild: np.ndarray
    moderate: np.ndarray
    high: np.ndarray


def build_perturbed_covariances(C_true: np.ndarray,
                                s: np.ndarray,
                                B_U: np.ndarray,
                                random_state: int = 1234) -> PerturbedCovariances:
    """
    Build structural perturbations and then add controlled low-rank
    negative components for mild and moderate, plus structure+negative
    for high.
    """
    rng = np.random.default_rng(random_state)

    # Start from structural perturbations (mostly PSD)
    C_mild_struct = mild_boundary_structure(C_true, s)
    C_mod_struct = moderate_patch_structure(C_true, s)
    C_high_struct = high_local_inconsistency_structure(C_true, s,
                                                       random_state=random_state)

    # Add controlled low-rank negative components
    # alpha_mild < alpha_mod < alpha_high
    C_mild = add_low_rank_negative(C_mild_struct, B_U, alpha=0.05, rng=rng)
    C_moderate = add_low_rank_negative(C_mod_struct, B_U, alpha=0.12, rng=rng)
    C_high = add_low_rank_negative(C_high_struct, B_U, alpha=0.15, rng=rng)

    return PerturbedCovariances(C_mild, C_moderate, C_high)


# ----------------------------------------------------------------------
# PSD projections
# ----------------------------------------------------------------------

def psd_projection_global(C: np.ndarray,
                          eps: float = 1e-3) -> np.ndarray:
    vals, vecs = eigh(C)
    vals_clipped = np.maximum(vals, eps)
    C_psd = (vecs * vals_clipped) @ vecs.T
    C_psd = 0.5 * (C_psd + C_psd.T)
    return C_psd


def psd_projection_subspace(C: np.ndarray,
                            B_U: np.ndarray,
                            eps: float = 0.0) -> np.ndarray:
    C_U = B_U.T @ C @ B_U
    vals, vecs = eigh(C_U)
    vals_clipped = np.maximum(vals, eps)
    C_U_psd = (vecs * vals_clipped) @ vecs.T
    delta_C_U = C_U_psd - C_U
    delta_C = B_U @ delta_C_U @ B_U.T
    return C + delta_C


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

@dataclass
class SpectralDiagnostics:
    min_eig: float
    neg_count: int
    neg_sum_abs: float
    mean_alignment: float
    max_alignment: float
    safety_index: float


def spectral_diagnostics(C: np.ndarray,
                         B_U: np.ndarray) -> SpectralDiagnostics:
    vals, vecs = eigh(C)
    neg_mask = vals < 0.0
    neg_vals = vals[neg_mask]
    neg_vecs = vecs[:, neg_mask]

    if neg_vals.size == 0:
        return SpectralDiagnostics(
            min_eig=float(vals.min()),
            neg_count=0,
            neg_sum_abs=0.0,
            mean_alignment=0.0,
            max_alignment=0.0,
            safety_index=0.0
        )

    P_U = projector_onto_subspace(B_U)
    norms = np.linalg.norm(P_U @ neg_vecs, axis=0)
    mean_align = float(norms.mean())
    max_align = float(norms.max())
    safety_index = float(np.max(-neg_vals * norms**2))

    return SpectralDiagnostics(
        min_eig=float(vals.min()),
        neg_count=int(neg_vals.size),
        neg_sum_abs=float(np.sum(np.abs(neg_vals))),
        mean_alignment=mean_align,
        max_alignment=max_align,
        safety_index=safety_index
    )


# ----------------------------------------------------------------------
# Kriging
# ----------------------------------------------------------------------

@dataclass
class KrigingResults:
    rmse: float
    mae: float
    neg_variances: int
    min_variance: float


def build_ok_system(C: np.ndarray,
                    ones: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = C
    K[:n, n] = ones
    K[n, :n] = ones
    K[n, n] = 0.0
    return K


def solve_ok(C: np.ndarray,
             c0: np.ndarray,
             C0: float,
             var_tol: float = 1e-12):
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
    var = C0 - 2.0 * np.dot(c0, w) + w @ (C @ w)

    is_neg = var < -var_tol
    if var < 0.0 and var >= -var_tol:
        var = 0.0
        is_neg = False

    return w, var, is_neg


def kriging_experiment(C_indef: np.ndarray,
                       C_true: np.ndarray,
                       s: np.ndarray,
                       s0: np.ndarray,
                       z: np.ndarray,
                       sill: float = 1.0,
                       rng_param: float = 2.0,
                       cov_model: str = "exp") -> Dict[str, KrigingResults]:
    n = len(s)
    m = len(s0)

    # Reference kriging with C_true
    z_ref = np.zeros(m)
    for j, loc in enumerate(s0):
        if cov_model == "exp":
            c0_true = covariance_vector_exp(s, loc, sill=sill, rng_param=rng_param)
        elif cov_model == "sph":
            c0_true = covariance_vector_sph(s, loc, sill=sill, rng_param=rng_param)
        else:
            raise ValueError(f"Unknown cov_model '{cov_model}'")
        C0_true = sill
        w_ref, _, _ = solve_ok(C_true, c0_true, C0_true)
        z_ref[j] = w_ref @ z

    B_U = build_prediction_subspace_basis(s, s0, sill=sill,
                                          rng_param=rng_param,
                                          cov_model=cov_model)

    C_raw = C_indef
    C_global_psd = psd_projection_global(C_indef)
    C_subspace_psd = psd_projection_subspace(C_indef, B_U)

    methods = {
        "raw_indef": C_raw,
        "global_psd": C_global_psd,
        "subspace_psd": C_subspace_psd
    }

    out: Dict[str, KrigingResults] = {}

    for name, C_use in methods.items():
        preds = np.zeros(m)
        vars_ = np.zeros(m)
        neg_count = 0
        min_var = np.inf

        for j, loc in enumerate(s0):
            if cov_model == "exp":
                c0 = covariance_vector_exp(s, loc, sill=sill, rng_param=rng_param)
            elif cov_model == "sph":
                c0 = covariance_vector_sph(s, loc, sill=sill, rng_param=rng_param)
            else:
                raise ValueError(f"Unknown cov_model '{cov_model}'")
            C0 = sill
            w, var, is_neg = solve_ok(C_use, c0, C0)
            preds[j] = w @ z
            vars_[j] = var
            if is_neg:
                neg_count += 1
            if var < min_var:
                min_var = var

        errors = preds - z_ref
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        out[name] = KrigingResults(rmse=rmse,
                                   mae=mae,
                                   neg_variances=neg_count,
                                   min_variance=float(min_var))

    return out


# ----------------------------------------------------------------------
# LaTeX formatting
# ----------------------------------------------------------------------

def format_eigenvalues_table(diagnostics):
    scenario_rows = [
        ("Mild (Boundary effect)",     "mild"),
        ("Moderate (Patchwise model)", "moderate"),
        ("High (Local inconsistency)", "high"),
    ]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Eigenvalue characteristics of perturbed covariance matrices.}")
    lines.append(r"    \label{tab:eigenvalues}")
    lines.append(r"    \begin{tabular}{lrrr}")
    lines.append(r"        \hline")
    lines.append(r"        Scenario & Min eigenvalue & Negative count & $\sum |\lambda_i^-|$ \\")
    lines.append(r"        \hline")

    for label, key in scenario_rows:
        d = diagnostics[key]
        lines.append(
            rf"        {label} & {d.min_eig:.3f} & {d.neg_count:d} & {d.neg_sum_abs:.3f} \\"
        )

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_performance_table(results, m_pred=10):
    scenario_order = ["mild", "moderate", "high"]
    scenario_labels = {
        "mild": "Mild",
        "moderate": "Moderate",
        "high": "High",
    }

    method_order = ["raw_indef", "global_psd", "subspace_psd"]
    method_labels = {
        "raw_indef": "Raw indefinite",
        "global_psd": "Global PSD",
        "subspace_psd": "Subspace PSD",
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Kriging performance: RMSE, MAE, and variance diagnostics.}")
    lines.append(r"    \label{tab:performance}")
    lines.append(r"    \begin{tabular}{llrrrr}")
    lines.append(r"        \hline")
    lines.append(r"        Scenario & Method & RMSE & MAE & Neg. vars. & Min var. \\")
    lines.append(r"        \hline")

    for scen in scenario_order:
        scen_label = scenario_labels[scen]
        scen_results = results[scen]
        first_in_block = True

        for method_key in method_order:
            r = scen_results[method_key]
            neg_fmt = f"{r.neg_variances}/{m_pred}"
            scen_col = scen_label if first_in_block else ""
            first_in_block = False

            lines.append(
                rf"        {scen_col} & {method_labels[method_key]} & "
                rf"{r.rmse:.3f} & {r.mae:.3f} & {neg_fmt} & {r.min_variance:.3f} \\"
            )

        lines.append(r"        \hline")

    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run_simulation(n: int = 30,
                   m: int = 10,
                   sill: float = 1.0,
                   rng_param: float = 2.0,
                   seed_design: int = 123,
                   seed_field: int = 123,
                   seed_high: int = 1234,
                   cov_model: str = "exp"):
    s, s0 = generate_design_and_prediction(n=n, m=m, random_state=seed_design)
    z, C_true = simulate_field(s, sill=sill, rng_param=rng_param,
                               random_state=seed_field,
                               cov_model=cov_model)

    # Build prediction subspace for controlled indefiniteness
    B_U = build_prediction_subspace_basis(s, s0, sill=sill,
                                          rng_param=rng_param,
                                          cov_model=cov_model)

    perturbed = build_perturbed_covariances(C_true, s, B_U, random_state=seed_high)

    scenarios = {
        "mild": perturbed.mild,
        "moderate": perturbed.moderate,
        "high": perturbed.high,
    }

    all_diagnostics = {}
    all_results = {}

    for name, C_indef in scenarios.items():
        diag = spectral_diagnostics(C_indef, B_U)
        all_diagnostics[name] = diag

        res = kriging_experiment(C_indef, C_true, s, s0, z,
                                 sill=sill, rng_param=rng_param,
                                 cov_model=cov_model)
        all_results[name] = res

    return {
        "cov_model": cov_model,
        "s": s,
        "s0": s0,
        "z": z,
        "C_true": C_true,
        "perturbed": perturbed,
        "diagnostics": all_diagnostics,
        "results": all_results,
    }


if __name__ == "__main__":
    # Exponential covariance run
    out_exp = run_simulation(cov_model="exp")
    print("=== Exponential covariance ===")
    for scen, diag in out_exp["diagnostics"].items():
        print(f"Scenario: {scen}")
        print(f"  min eigenvalue: {diag.min_eig:.3f}")
        print(f"  negative count: {diag.neg_count}")
        print(f"  sum |negative eigs|: {diag.neg_sum_abs:.3f}")
        print(f"  mean alignment: {diag.mean_alignment:.2f}")
        print(f"  max alignment: {diag.max_alignment:.2f}")
        print(f"  safety index: {diag.safety_index:.4f}")
        print()

    for scen, methods in out_exp["results"].items():
        print(f"Scenario: {scen}")
        for name, res in methods.items():
            print(f"  Method: {name}")
            print(f"    RMSE: {res.rmse:.3f}")
            print(f"    MAE: {res.mae:.3f}")
            print(f"    neg. variances: {res.neg_variances}")
            print(f"    min variance: {res.min_variance:.3f}")
        print()

    latex_eig_exp = format_eigenvalues_table(out_exp["diagnostics"])
    latex_perf_exp = format_performance_table(out_exp["results"],
                                              m_pred=len(out_exp["s0"]))

    # Spherical covariance run
    out_sph = run_simulation(cov_model="sph")
    print("=== Spherical covariance ===")
    for scen, diag in out_sph["diagnostics"].items():
        print(f"Scenario: {scen}")
        print(f"  min eigenvalue: {diag.min_eig:.3f}")
        print(f"  negative count: {diag.neg_count}")
        print(f"  sum |negative eigs|: {diag.neg_sum_abs:.3f}")
        print(f"  mean alignment: {diag.mean_alignment:.2f}")
        print(f"  max alignment: {diag.max_alignment:.2f}")
        print(f"  safety index: {diag.safety_index:.4f}")
        print()

    for scen, methods in out_sph["results"].items():
        print(f"Scenario: {scen}")
        for name, res in methods.items():
            print(f"  Method: {name}")
            print(f"    RMSE: {res.rmse:.3f}")
            print(f"    MAE: {res.mae:.3f}")
            print(f"    neg. variances: {res.neg_variances}")
            print(f"    min variance: {res.min_variance:.3f}")
        print()

    latex_eig_sph = format_eigenvalues_table(out_sph["diagnostics"])
    latex_perf_sph = format_performance_table(out_sph["results"],
                                              m_pred=len(out_sph["s0"]))

    # Print all LaTeX tables together (to paste into the paper)
    print("% ==== Exponential covariance tables ====")
    print(latex_eig_exp)
    print()
    print(latex_perf_exp)
    print()
    print("% ==== Spherical covariance tables ====")
    print(latex_eig_sph)
    print()
    print(latex_perf_sph)