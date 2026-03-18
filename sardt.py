import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Geometry:
    wavelength: float          # radar wavelength (m)
    slant_ranges: np.ndarray   # shape [K]
    baselines_perp: np.ndarray # shape [K]
    incidence_angles: np.ndarray  # shape [K], radians


@dataclass
class SubApertureConfig:
    width_bins: int            # Doppler window width
    step_bins: int             # Doppler shift step
    n_apertures: int           # number of sub-aperture positions
    taper: str = "hann"        # taper for Doppler windows


def _make_1d_window(n: int, kind: str) -> np.ndarray:
    if kind == "hann":
        return np.hanning(n)
    if kind == "rect":
        return np.ones(n)
    raise ValueError(f"Unsupported taper: {kind}")


def build_doppler_subapertures(
    slc: np.ndarray,
    cfg: SubApertureConfig,
) -> List[np.ndarray]:
    """
    Create K low-resolution SLC images by sliding a Doppler window
    along the azimuth-frequency axis of the focused SLC spectrum.
    """
    if slc.ndim != 2:
        raise ValueError("slc must be a 2D complex array [range, azimuth]")

    nr, na = slc.shape
    spec = fftshift(fft2(slc), axes=(0, 1))

    if cfg.width_bins > na:
        raise ValueError("width_bins cannot exceed azimuth dimension")

    dop_win = _make_1d_window(cfg.width_bins, cfg.taper)
    sub_slcs: List[np.ndarray] = []

    # azimuth frequency axis = dim 1
    centers = []
    start = cfg.width_bins // 2
    end = na - cfg.width_bins // 2
    for c in range(start, end, cfg.step_bins):
        centers.append(c)
        if len(centers) >= cfg.n_apertures:
            break

    for c in centers:
        mask = np.zeros((nr, na), dtype=np.float32)
        left = c - cfg.width_bins // 2
        right = left + cfg.width_bins

        if left < 0 or right > na:
            continue

        mask[:, left:right] = dop_win[None, :]
        sub_spec = spec * mask
        sub_slc = ifft2(ifftshift(sub_spec, axes=(0, 1)))
        sub_slcs.append(sub_slc)

    return sub_slcs


def estimate_measurement_vector_phase_tracking(
    sub_slcs: List[np.ndarray],
    pixel: Tuple[int, int],
    ref_index: int = 0,
) -> np.ndarray:
    """
    Reconstruct raw tomographic measurement vector Y for one pixel.
    Uses complex phase relative to a reference sub-aperture.

    This is a practical stand-in for the paper's pixel tracking/coregistration stage.
    """
    if not sub_slcs:
        raise ValueError("No sub-aperture SLCs provided")

    r, a = pixel
    ref = sub_slcs[ref_index][r, a]
    if np.abs(ref) < 1e-12:
        ref = ref + 1e-12

    y = np.empty(len(sub_slcs), dtype=np.complex128)
    for k, img in enumerate(sub_slcs):
        val = img[r, a]
        y[k] = val * np.conj(ref)

    return y


def estimate_measurement_vector_patch_tracking(
    sub_slcs: List[np.ndarray],
    pixel: Tuple[int, int],
    patch_radius: int = 2,
    ref_index: int = 0,
) -> np.ndarray:
    """
    More stable variant: average complex coherence on a small patch.
    """
    r, a = pixel
    nr, na = sub_slcs[0].shape
    r0 = max(0, r - patch_radius)
    r1 = min(nr, r + patch_radius + 1)
    a0 = max(0, a - patch_radius)
    a1 = min(na, a + patch_radius + 1)

    ref_patch = sub_slcs[ref_index][r0:r1, a0:a1]
    denom = np.vdot(ref_patch, ref_patch).real + 1e-12

    y = np.empty(len(sub_slcs), dtype=np.complex128)
    for k, img in enumerate(sub_slcs):
        patch = img[r0:r1, a0:a1]
        # local complex coherence-like statistic
        y[k] = np.vdot(ref_patch, patch) / np.sqrt(
            denom * (np.vdot(patch, patch).real + 1e-12)
        )
    return y


def build_steering_matrix(
    geom: Geometry,
    z_grid: np.ndarray,
) -> np.ndarray:
    """
    A[k, n] = exp( j * Kz[k] * z[n] )

    with Kz[k] = 4*pi*B_perp[k] / (lambda * r[k] * sin(theta[k])).
    """
    K = len(geom.slant_ranges)
    if not (len(geom.baselines_perp) == K and len(geom.incidence_angles) == K):
        raise ValueError("Geometry vectors must have same length")

    kz = (
        4.0 * np.pi * geom.baselines_perp /
        (geom.wavelength * geom.slant_ranges * np.sin(geom.incidence_angles))
    )  # shape [K]

    A = np.exp(1j * kz[:, None] * z_grid[None, :])
    return A


def compute_kz(geom: Geometry) -> np.ndarray:
    return (
        4.0 * np.pi * geom.baselines_perp /
        (geom.wavelength * geom.slant_ranges * np.sin(geom.incidence_angles))
    )


def solve_tomography_pinv(
    y: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Paper-faithful linear inversion using the pseudoinverse."""
    return np.linalg.pinv(A) @ y


def solve_tomography_tikhonov(
    y: np.ndarray,
    A: np.ndarray,
    alpha: float = 1e-2,
) -> np.ndarray:
    """
    Regularized version better suited to noise and realtime stability:
    h = (A^H A + alpha I)^(-1) A^H y
    """
    AH = A.conj().T
    n = A.shape[1]
    return np.linalg.solve(AH @ A + alpha * np.eye(n), AH @ y)


def solve_tomography_omp(
    y: np.ndarray,
    A: np.ndarray,
    sparsity: int = 3,
) -> np.ndarray:
    residual = y.astype(np.complex128).copy()
    active: List[int] = []
    h = np.zeros(A.shape[1], dtype=np.complex128)

    for _ in range(min(sparsity, A.shape[0], A.shape[1])):
        corr = np.abs(A.conj().T @ residual)
        if active:
            corr[np.asarray(active)] = -np.inf
        idx = int(np.argmax(corr))
        if not np.isfinite(corr[idx]) or corr[idx] <= 0:
            break
        active.append(idx)
        A_active = A[:, active]
        h_active, *_ = np.linalg.lstsq(A_active, y, rcond=None)
        residual = y - A_active @ h_active
        if np.linalg.norm(residual) < 1e-8:
            break

    if active:
        h[np.asarray(active)] = h_active
    return h


def solve_tomography_ista(
    y: np.ndarray,
    A: np.ndarray,
    lam: float = 1e-2,
    n_iter: int = 80,
) -> np.ndarray:
    AH = A.conj().T
    lipschitz = float(np.linalg.norm(A, 2) ** 2) + 1e-12
    step = 1.0 / lipschitz
    h = np.zeros(A.shape[1], dtype=np.complex128)

    for _ in range(n_iter):
        grad = AH @ (A @ h - y)
        x = h - step * grad
        mag = np.abs(x)
        shrink = np.maximum(0.0, mag - lam * step) / np.maximum(mag, 1e-12)
        h = shrink * x

    return h


def solve_tomography(
    y: np.ndarray,
    A: np.ndarray,
    mode: str = "tikhonov",
    alpha: float = 1e-2,
    sparsity: int = 3,
    lam: float = 1e-2,
    n_iter: int = 80,
) -> np.ndarray:
    if mode == "pinv":
        return solve_tomography_pinv(y, A)
    if mode == "tikhonov":
        return solve_tomography_tikhonov(y, A, alpha=alpha)
    if mode == "omp":
        return solve_tomography_omp(y, A, sparsity=sparsity)
    if mode == "ista":
        return solve_tomography_ista(y, A, lam=lam, n_iter=n_iter)
    raise ValueError(f"Unsupported inversion mode: {mode}")


def preprocess_measurement(
    y: np.ndarray,
    A: np.ndarray,
    z_grid: np.ndarray,
    mode: str = "none",
) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "unit_phase":
        return y / np.maximum(np.abs(y), 1e-12)
    if mode == "demean":
        return y - np.mean(y)
    if mode == "unit_phase_demean":
        y = y / np.maximum(np.abs(y), 1e-12)
        return y - np.mean(y)
    if mode == "remove_surface":
        idx = int(np.argmin(np.abs(z_grid)))
        a = A[:, idx]
        coeff = np.vdot(a, y) / (np.vdot(a, a) + 1e-12)
        return y - coeff * a
    if mode == "unit_phase_remove_surface":
        y = y / np.maximum(np.abs(y), 1e-12)
        idx = int(np.argmin(np.abs(z_grid)))
        a = A[:, idx]
        coeff = np.vdot(a, y) / (np.vdot(a, a) + 1e-12)
        return y - coeff * a
    raise ValueError(f"Unsupported preprocess mode: {mode}")


def compute_tomogram_for_pixels(
    sub_slcs: List[np.ndarray],
    pixels: List[Tuple[int, int]],
    geom: Geometry,
    z_grid: np.ndarray,
    patch_radius: int = 2,
    alpha: float = 1e-2,
    measurement_mode: str = "patch",
    inversion_mode: str = "tikhonov",
    sparsity: int = 3,
    lam: float = 1e-2,
    n_iter: int = 80,
    preprocess_mode: str = "none",
) -> np.ndarray:
    """
    Returns tomogram volume [num_pixels, num_depths]
    """
    K = len(sub_slcs)
    if K != len(geom.slant_ranges):
        raise ValueError("Need one geometry entry per sub-aperture")

    A = build_steering_matrix(geom, z_grid)
    out = np.empty((len(pixels), len(z_grid)), dtype=np.complex128)

    for i, px in enumerate(pixels):
        if measurement_mode == "patch":
            y = estimate_measurement_vector_patch_tracking(
                sub_slcs, px, patch_radius=patch_radius
            )
        elif measurement_mode == "phase":
            y = estimate_measurement_vector_phase_tracking(sub_slcs, px)
        else:
            raise ValueError(f"Unsupported measurement mode: {measurement_mode}")
        y = preprocess_measurement(y, A, z_grid, mode=preprocess_mode)
        h = solve_tomography(
            y,
            A,
            mode=inversion_mode,
            alpha=alpha,
            sparsity=sparsity,
            lam=lam,
            n_iter=n_iter,
        )
        out[i, :] = h

    return out
