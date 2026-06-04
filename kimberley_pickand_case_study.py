
"""
Experimental study for:

Maturana, F.J.
'Non-Positive-Definite Covariance Structures in Geostatistics:
Theory, Diagnostics, and Implications for Kriging'.

Purpose
-------
Numerical experiment for inconsistency-tolerant geostatistics using the
Kimberley stream geochemistry dataset.

The program evaluates the impact of non-positive-definite covariance
structures on ordinary kriging predictions for individual stream-element
subsets. For each selected stream and geochemical variable, a reference
spherical covariance model is constructed and then perturbed to create
three levels of covariance inconsistency:

    1. Mild boundary-type inconsistency
    2. Moderate patchwise inconsistency
    3. High local inconsistency

Each perturbed covariance matrix is analysed through spectral diagnostics,
including minimum eigenvalue, number of negative eigenvalues, total
negative spectral mass, alignment with the prediction subspace, and a
prediction safety index.

Three covariance treatments are compared:

    • Raw indefinite covariance matrix
    • Global positive-semidefinite (PSD) projection
    • Prediction-subspace PSD projection

For each scenario and treatment, ordinary kriging predictions are
computed and compared against predictions obtained from the original
positive-definite covariance model. Performance is assessed using:

    • Root Mean Squared Error (RMSE)
    • Mean Absolute Error (MAE)
    • Number of negative kriging variances
    • Minimum kriging variance

The program automatically generates LaTeX tables summarising spectral
properties and kriging performance for each stream analysed, providing a
reproducible case study of covariance inconsistency and its effects on
spatial prediction.


Dataset
-------
The analysis uses stream-sediment geochemical data from the
Pickands–Mather Kimberley survey (Western Australia).

Streams analysed:
    StmNo = 1, 2, 3, 4

Element variables:
    Cu_40PNB_p


Negative assay values are treated as missing data.
"""


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

# ----------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------

KIMBERLEY_FILE = "kimberley_pickand.csv"   # point-level data with X, Y, StmNo, Cu_40PNB_p, ...

# Base covariance parameters for stream 4
BASE_SILL = 0.9
NUGGET = 0.10          # nugget variance 
BASE_RANGE = 9000.0  # metres; adjust to your fitted Kimberley range

# ----------------------------------------------------------------------
# 1. BASIC COVARIANCE AND KRIGING BUILDING BLOCKS
# ----------------------------------------------------------------------


from numpy.linalg import eigh, solve
from scipy.spatial.distance import cdist
import numpy as np

def spherical_covariance(locs: np.ndarray,
                         sill: float,
                         rng: float) -> np.ndarray:

    dists = cdist(locs, locs, metric="euclidean")
    h = dists / rng
    C = np.zeros_like(h)
    mask = h <= 1.0
    h_m = h[mask]
    C[mask] = sill * (1.0 - 1.5 * h_m + 0.5 * (h_m**3))
    return C


def spherical_covariance_vector(s: np.ndarray,
                                s0: np.ndarray,
                                sill: float,
                                rng: float) -> np.ndarray:

    d = np.linalg.norm(s - s0, axis=1)
    h = d / rng
    c0 = np.zeros_like(h)
    mask = h <= 1.0
    h_m = h[mask]
    c0[mask] = sill * (1.0 - 1.5 * h_m + 0.5 * (h_m**3))
    return c0




def build_prediction_subspace_basis(
    s: np.ndarray, s0: np.ndarray, sill: float = BASE_SILL, rng_param: float = BASE_RANGE
) -> np.ndarray:
    """
    Orthonormal basis BU for U = span{1, c0(s0_j)} in R^n.
    """
    n = s.shape[0]
    ones = np.ones(n)
    cols = [ones]
    for loc in s0:
        c0 = spherical_covariance_vector(s, loc, sill=sill, rng=rng_param)
        cols.append(c0)
    Uraw = np.column_stack(cols)
    Q, R = np.linalg.qr(Uraw)
    diagR = np.abs(np.diag(R))
    tol = 1e-12 * np.max(diagR)
    keep = diagR > tol
    BU = Q[:, keep]
    return BU


def projector_onto_subspace(BU: np.ndarray) -> np.ndarray:
    return BU @ BU.T


@dataclass
class PerturbedCovariances:
    mild: np.ndarray
    moderate: np.ndarray
    high: np.ndarray


# Mild / moderate / high structural perturbations 


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
    for i in mid_idx:
        for j in np.concatenate([left_idx, right_idx]):
            h = abs(x[i] - x[j])
            mod = 1.0 - reduction * (0.5 + 0.5 * np.sin(2 * np.pi * h / (smax - smin)))
            Cpert[i, j] *= mod
            Cpert[j, i] = Cpert[i, j]
    return Cpert


def moderate_patch_structure(C: np.ndarray,
                             s: np.ndarray,
                             reduction: float = 0.7,
                             leftscale: float = 0.9,
                             rightscale: float = 1.1) -> np.ndarray:
    """
    Patchwise-type perturbation: two halves in x, asymmetric scaling and reduced cross-links.
    """
    x = s[:, 0]
    smid = 0.5 * (x.min() + x.max())
    left_idx = np.where(x <= smid)[0]
    right_idx = np.where(x > smid)[0]

    Cpert = C.copy()
    # Scale diagonal blocks
    for i in left_idx:
        for j in left_idx:
            Cpert[i, j] *= leftscale
    for i in right_idx:
        for j in right_idx:
            Cpert[i, j] *= rightscale
    # Reduce cross-block entries
    for i in left_idx:
        for j in right_idx:
            Cpert[i, j] *= (1.0 - reduction)
            Cpert[j, i] = Cpert[i, j]
    return Cpert


def high_local_inconsistency_structure(
    C: np.ndarray, s: np.ndarray, ntriples: int | None = None,
    magnitude: float = 0.25, random_state: int = 1234
) -> np.ndarray:
    """
###Strong local inconsistency via random triples of indices (2D analogue of your 1D code).
    """
    rng = np.random.default_rng(random_state)
    n = s.shape[0]
    if ntriples is None:
        ntriples = max(n // 2, 1)
    Cpert = C.copy()
    for _ in range(ntriples):
        idx = rng.choice(n, size=3, replace=False)
        i, j, k = idx
        deltaij = magnitude * rng.uniform(-1.0, 1.0)
        deltaik = magnitude * rng.uniform(-1.0, 1.0)
        deltajk = magnitude * rng.uniform(-1.0, 1.0)
        for a, b, delta in [(i, j, deltaij),
                            (i, k, -deltaik),
                            (j, k, deltajk)]:
            Cpert[a, b] += delta
            Cpert[b, a] = Cpert[a, b]
    return Cpert


def add_low_rank_negative(C: np.ndarray,
                          BU: np.ndarray,
                          alpha: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Add a rank-1 negative term 
    """
    n = C.shape[0]
    PU = projector_onto_subspace(BU)
    I = np.eye(n)
    v = rng.standard_normal(n)
    u = (I - PU) @ v
    normu = np.linalg.norm(u)
    if normu < 1e-8:
        # fallback: random direction
        u = v
        normu = np.linalg.norm(u)
    u = u / normu
    return C - alpha * np.outer(u, u)


def build_perturbed_covariances(
    Ctrue: np.ndarray, s: np.ndarray, BU: np.ndarray,
    random_state: int = 1234
) -> PerturbedCovariances:
    rng = np.random.default_rng(random_state)

    Cmild_struct = mild_boundary_structure(Ctrue, s)
    Cmod_struct = moderate_patch_structure(Ctrue, s)
    Chigh_struct = high_local_inconsistency_structure(Ctrue, s)

    Cmild = add_low_rank_negative(Cmild_struct, BU, alpha=0.05, rng=rng)
    Cmoderate = add_low_rank_negative(Cmod_struct, BU, alpha=0.12, rng=rng)
    Chigh = add_low_rank_negative(Chigh_struct, BU, alpha=0.15, rng=rng)

    return PerturbedCovariances(Cmild, Cmoderate, Chigh)


def psd_projection_global(C: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    vals, vecs = eigh(C)
    vals_clipped = np.maximum(vals, eps)
    Cpsd = vecs @ np.diag(vals_clipped) @ vecs.T
    Cpsd = 0.5 * (Cpsd + Cpsd.T)
    return Cpsd


def psd_projection_subspace(C: np.ndarray, BU: np.ndarray, eps: float = 0.0) -> np.ndarray:
    CU = BU.T @ C @ BU
    vals, vecs = eigh(CU)
    vals_clipped = np.maximum(vals, eps)
    CUpsd = vecs @ np.diag(vals_clipped) @ vecs.T
    deltaCU = CUpsd - CU
    deltaC = BU @ deltaCU @ BU.T
    return C + deltaC


@dataclass
class SpectralDiagnostics:
    min_eig: float
    neg_count: int
    neg_sum_abs: float
    mean_align: float
    max_align: float
    safety_index: float


def spectral_diagnostics(C: np.ndarray, BU: np.ndarray) -> SpectralDiagnostics:
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
            safety_index=0.0,
        )
    PU = projector_onto_subspace(BU)
    norms = np.linalg.norm(PU @ negvecs, axis=0)
    meanalign = float(norms.mean())
    maxalign = float(norms.max())
    safety_index = float(np.max(-negvals * norms**2))
    return SpectralDiagnostics(
        min_eig=float(vals.min()),
        neg_count=int(negvals.size),
        neg_sum_abs=float(np.sum(np.abs(negvals))),
        mean_align=meanalign,
        max_align=maxalign,
        safety_index=safety_index,
    )


@dataclass
class KrigingResults:
    rmse: float
    mae: float
    neg_variances: int
    min_variance: float


def build_ok_system(C: np.ndarray, ones: np.ndarray) -> np.ndarray:
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
             vartol: float = 1e-12) -> Tuple[np.ndarray, float, bool]:
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
    is_neg = False
    if var < -vartol:
        is_neg = True
    if var < 0.0 and var > -vartol:
        var = 0.0
    return w, float(var), is_neg


def kriging_experiment(
    Cindef: np.ndarray,
    Ctrue: np.ndarray,
    s: np.ndarray,
    s0: np.ndarray,
    z: np.ndarray,
    sill: float = BASE_SILL,
    rng_param: float = BASE_RANGE,
) -> Dict[str, KrigingResults]:
    n = s.shape[0]
    m = s0.shape[0]

    # Reference kriging under Ctrue
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

    methods = {
        "raw_indef": Craw,
        "global_psd": Cglobal,
        "subspace_psd": Csub,
    }

    out: Dict[str, KrigingResults] = {}
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
            if var < minvar:
                minvar = var
        errors = preds - zref
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        out[name] = KrigingResults(
            rmse=rmse,
            mae=mae,
            neg_variances=negcount,
            min_variance=float(minvar),
        )

    return out


# ----------------------------------------------------------------------
# 2. STREAM–ELEMENT DRIVER
# ----------------------------------------------------------------------

def load_kimberley_points() -> pd.DataFrame:
    df = pd.read_csv(KIMBERLEY_FILE)
    df = df.dropna(subset=["X", "Y"])
    geochem_cols = [c for c in df.columns if c.endswith("_40PNB_p")]

    for c in geochem_cols:
        df.loc[df[c] < 0, c] = np.nan

    return df    


import numpy as np
import pandas as pd
from typing import Dict, Tuple

def run_stream_element_case(stream_no: int,
                            element: str,
                            m_pred: int = 20
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, Dict[str, SpectralDiagnostics],
                                       Dict[str, Dict[str, KrigingResults]]]:

    # 1. Load full Kimberley dataset
    df = load_kimberley_points()   # this should at least ensure coords are present

    # 2. Filter to the chosen stream
    if "StmNo" not in df.columns:
        raise KeyError("Expected stream column 'StmNo' not found; adjust run_stream_element_case.")

    df_stream = df[df["StmNo"] == stream_no].copy()

    if df_stream.empty:
        raise ValueError(f"No samples found for stream {stream_no} in the Kimberley dataset.")

    # 3. Drop all rows where the chosen element is missing
    if element not in df_stream.columns:
        raise KeyError(
            f"Element '{element}' not found in stream data. "
            f"Available columns include: {list(df_stream.columns)}"
        )

    # Remove NaN for this element
    before = len(df_stream)
    df_stream = df_stream.dropna(subset=[element])
    after = len(df_stream)

    if df_stream.empty:
        raise ValueError(
            f"All {before} samples for stream {stream_no} have NaN for element '{element}'."
        )

    # Optional: quick sanity check
    z_values = df_stream[element].to_numpy()
    if not np.isfinite(z_values).all():
        raise ValueError(
            f"Non-finite values remain in element '{element}' after dropna "
            f"(stream {stream_no})."
        )

    # 4. Extract coordinates and element values
    for col in ("X", "Y"):
        if col not in df_stream.columns:
            raise KeyError(
                f"Expected coordinate column '{col}' not found; "
                f"adjust run_stream_element_case to your column names."
            )

    coords = df_stream[["X", "Y"]].to_numpy()
    z = z_values

    # 5. Build true covariance matrix on this stream

    Ctrue = spherical_covariance(coords, sill=BASE_SILL, rng=BASE_RANGE)
    
    Ctrue = Ctrue + NUGGET * np.eye(Ctrue.shape[0])
    # 6. Choose prediction locations
    n = coords.shape[0]
    if n <= m_pred:
        s0 = coords.copy()
    else:
        idx = np.linspace(0, n - 1, m_pred, dtype=int)
        s0 = coords[idx, :]

    # 7. Build prediction subspace basis
    BU = build_prediction_subspace_basis(coords, s0,
                                         sill=BASE_SILL,
                                         rng_param=BASE_RANGE)

    # 8. Build perturbed covariance matrices (mild / moderate / high)
    perturbed = build_perturbed_covariances(Ctrue, coords, BU)

    scenarios = {
        "mild": perturbed.mild,
        "moderate": perturbed.moderate,
        "high": perturbed.high,
    }

    diagnostics: Dict[str, SpectralDiagnostics] = {}
    results: Dict[str, Dict[str, KrigingResults]] = {}

    # 9. Spectral diagnostics and kriging for each scenario
    for name, Cindef in scenarios.items():
        diagnostics[name] = spectral_diagnostics(Cindef, BU)
        results[name] = kriging_experiment(
            Cindef=Cindef,
            Ctrue=Ctrue,
            s=coords,
            s0=s0,
            z=z,
            sill=BASE_SILL,
            rng_param=BASE_RANGE,
        )

    return coords, s0, z, Ctrue, diagnostics, results

# ----------------------------------------------------------------------
# 3. LATEX TABLE FORMATTERS (REUSE YOUR STRUCTURE)
# ----------------------------------------------------------------------

def format_eigenvalues_table(
        diagnostics,
        stream_no,
        element):

    scenariorows = [
        ("Mild (Boundary-type)", "mild"),
        ("Moderate (Patchwise-type)", "moderate"),
        ("High (Local inconsistency)", "high"),
    ]

    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")

    lines.append(
        f"\\caption{{Eigenvalue characteristics for "
        f"Stream {stream_no} and {element}.}}"
    )

    lines.append(
        f"\\label{{tab:eigen_stream_{stream_no}}}"
    )

    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\hline")

    lines.append(
        "Scenario & Min eigenvalue & Negative count & "
        "$\\sum |\\lambda_i^-|$ \\\\"
    )

    lines.append("\\hline")

    for label, key in scenariorows:

        d = diagnostics[key]

        lines.append(
            f"{label} & "
            f"{d.min_eig:.3f} & "
            f"{d.neg_count:d} & "
            f"{d.neg_sum_abs:.3f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def format_performance_table(
        results,
        m_pred,
        stream_no,
        element):

    scenario_order = [
        "mild",
        "moderate",
        "high"
    ]

    scenario_labels = {
        "mild": "Mild",
        "moderate": "Moderate",
        "high": "High",
    }

    method_order = [
        "raw_indef",
        "global_psd",
        "subspace_psd"
    ]

    method_labels = {
        "raw_indef": "Raw indefinite",
        "global_psd": "Global PSD",
        "subspace_psd": "Subspace PSD",
    }

    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")

    lines.append(
        f"\\caption{{Kriging performance for "
        f"Stream {stream_no} and {element}.}}"
    )

    lines.append(
        f"\\label{{tab:performance_stream_{stream_no}}}"
    )

    lines.append("\\begin{tabular}{llrrrr}")

    lines.append("\\hline")

    lines.append(
        "Scenario & Method & RMSE & MAE & "
        "Neg. vars. & Min var. \\\\"
    )

    lines.append("\\hline")

    for scen in scenario_order:

        first = True

        for method_key in method_order:

            r = results[scen][method_key]

            negfmt = (
                f"{r.neg_variances}/{m_pred}"
            )

            scenlabel = (
                scenario_labels[scen]
                if first else ""
            )

            first = False

            lines.append(
                f"{scenlabel} & "
                f"{method_labels[method_key]} & "
                f"{r.rmse:.3f} & "
                f"{r.mae:.3f} & "
                f"{negfmt} & "
                f"{r.min_variance:.3f} \\\\"
            )

        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 4. MAIN ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":

    element = "Cu_40PNB_p"
    all_results = {}
    all_diagnostics = {}


    for stream_no in range(1, 5):

        print("\n")
        print("=" * 80)
        print(
            f"STREAM {stream_no} | "
            f"ELEMENT {element}"
        )
        print("=" * 80)

        try:

            coords, s0, z, Ctrue, diagnostics, results = \
                run_stream_element_case(
                    stream_no=stream_no,
                    element=element,
                    m_pred=20
                )

            all_results[stream_no] = results
            all_diagnostics[stream_no] = diagnostics

            print("\nEIGENVALUE DIAGNOSTICS")
            print("-" * 80)

            for scen, diag in diagnostics.items():

                print(f"\nScenario: {scen}")

                print(
                    f"  min eigenvalue : "
                    f"{diag.min_eig:.4f}"
                )

                print(
                    f"  negative count : "
                    f"{diag.neg_count}"
                )

                print(
                    f"  sum |neg eig|  : "
                    f"{diag.neg_sum_abs:.4f}"
                )

                print(
                    f"  mean alignment : "
                    f"{diag.mean_align:.3f}"
                )

                print(
                    f"  max alignment  : "
                    f"{diag.max_align:.3f}"
                )

                print(
                    f"  safety index   : "
                    f"{diag.safety_index:.4f}"
                )

            print("\nKRIGING PERFORMANCE")
            print("-" * 80)

            for scen, methods in results.items():

                print(f"\nScenario: {scen}")

                for method_name, res in methods.items():

                    print(
                        f"{method_name:20s}"
                        f" RMSE={res.rmse:.3f}"
                        f" MAE={res.mae:.3f}"
                        f" NegVars={res.neg_variances}"
                        f" MinVar={res.min_variance:.3f}"
                    )

            latex_eig = format_eigenvalues_table(
                diagnostics,
                stream_no,
                element
            )

            latex_perf = format_performance_table(
                results,
                s0.shape[0],
                stream_no,
                element
            )

            eig_file = (
                f"stream_{stream_no}_"
                f"{element}_eigen.tex"
            )

            perf_file = (
                f"stream_{stream_no}_"
                f"{element}_performance.tex"
            )

            with open(
                eig_file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(latex_eig)

            with open(
                perf_file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(latex_perf)

            print("\nLATEX TABLES SAVED")

            print(eig_file)
            print(perf_file)

        except Exception as e:

            print(
                f"\nSTREAM {stream_no} FAILED"
            )

            print(str(e))

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for stream_no in sorted(all_results.keys()):

        print(f"\nStream {stream_no}")

        for scen, methods in all_results[
                stream_no].items():

            print(f"  {scen}")

            for method_name, res in methods.items():

                print(
                    f"    {method_name:20s}"
                    f" RMSE={res.rmse:.3f}"
                )