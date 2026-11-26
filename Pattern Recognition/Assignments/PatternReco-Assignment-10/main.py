#!/usr/bin/env python3
"""
Pattern Recognition Lab - Assignment 10
2D Discrete Fourier Transform (2D DFT) Spectral Analysis,
Frequency-Domain Filtering, and Invariant Feature Extraction.
Strictly implemented from scratch using NumPy matrix algebra.
"""

import os
import gc
import time
import glob
import math
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory

app = Flask(
    __name__,
    template_folder='Templates',
    static_folder='Statics'
)
app.config['SECRET_KEY'] = 'vn-w26-pr-lab-10-secret'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'Uploads')
app.config['PLOTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'Statics', 'plots')
app.config['DATASET_FOLDER'] = os.path.join(os.path.dirname(__file__), 'Dataset')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)


# ==============================================================================
# 1. FROM-SCRATCH 2D DFT & 2D IDFT MATRIX ENGINE
# ==============================================================================

class Custom2DDFT:
    """
    Implements 2D Discrete Fourier Transform (2D DFT) and 2D Inverse DFT (2D IDFT)
    strictly from scratch using NumPy matrix algebra (separable 1D DFT matrices).
    No np.fft routines are used in forward/inverse transformation.
    """
    _matrix_cache = {}

    @classmethod
    def get_dft_matrix(cls, N: int) -> np.ndarray:
        """
        Generates or retrieves cached 1D DFT matrix W_N of size (N, N).
        (W_N)_{u, x} = exp(-2j * pi * u * x / N)
        """
        if N not in cls._matrix_cache:
            n = np.arange(N)
            # Outer product u * x
            ux = np.outer(n, n)
            W_N = np.exp(-2j * np.pi * ux / N)
            cls._matrix_cache[N] = W_N
        return cls._matrix_cache[N]

    @classmethod
    def fft2(cls, image: np.ndarray, center: bool = True) -> np.ndarray:
        """
        Computes 2D DFT of an (M, N) 2D real or complex array from scratch.
        F = W_M @ ((-1)^(x+y) * f) @ W_N
        """
        M, N = image.shape
        WM = cls.get_dft_matrix(M)
        WN = cls.get_dft_matrix(N)

        if center:
            x = np.arange(M).reshape(-1, 1)
            y = np.arange(N).reshape(1, -1)
            shift_mat = (-1.0) ** (x + y)
            f_in = image * shift_mat
        else:
            f_in = image.astype(complex)

        # Separable 2D DFT via matrix multiplications
        F = WM @ f_in @ WN
        return F

    @classmethod
    def ifft2(cls, F: np.ndarray, center: bool = True) -> np.ndarray:
        """
        Computes 2D IDFT of an (M, N) frequency spectrum from scratch.
        f = Re( (1/(MN)) * (W_M^* @ F @ W_N^*) * (-1)^(x+y) )
        """
        M, N = F.shape
        WM = cls.get_dft_matrix(M)
        WN = cls.get_dft_matrix(N)

        # Inverse transformation using conjugate transpose
        # Note: (W_N)^-1 = (1/N) * W_N^*
        f_raw = (WM.conj().T @ F @ WN.conj().T) / (M * N)

        if center:
            x = np.arange(M).reshape(-1, 1)
            y = np.arange(N).reshape(1, -1)
            shift_mat = (-1.0) ** (x + y)
            f_spatial = np.real(f_raw * shift_mat)
        else:
            f_spatial = np.real(f_raw)

        return f_spatial

    @classmethod
    def verify_against_numpy(cls, image: np.ndarray, center: bool = True) -> dict:
        """
        Verifies numerical accuracy and benchmarks runtime between from-scratch DFT
        and numpy.fft.fft2.
        """
        # Timing scratch
        t0 = time.perf_counter()
        F_scratch = cls.fft2(image, center=center)
        t_scratch = (time.perf_counter() - t0) * 1000.0  # ms

        # Timing numpy
        t1 = time.perf_counter()
        if center:
            F_numpy = np.fft.fftshift(np.fft.fft2(image))
        else:
            F_numpy = np.fft.fft2(image)
        t_numpy = (time.perf_counter() - t1) * 1000.0  # ms

        # Numerical difference
        abs_diff = np.abs(F_scratch - F_numpy)
        max_err = float(np.max(abs_diff))
        mean_err = float(np.mean(abs_diff))

        # IDFT verification
        img_rec = cls.ifft2(F_scratch, center=center)
        idft_err = float(np.max(np.abs(image - img_rec)))

        return {
            'time_scratch_ms': round(t_scratch, 4),
            'time_numpy_ms': round(t_numpy, 4),
            'speedup_ratio': round(t_scratch / (t_numpy + 1e-9), 2),
            'max_abs_error': max_err,
            'mean_abs_error': mean_err,
            'idft_max_error': idft_err,
            'is_valid': bool(max_err < 1e-10)
        }


# ==============================================================================
# 2. FILTER TRANSFER FUNCTIONS & SPATIAL RECONSTRUCTION
# ==============================================================================

class FrequencyFilters:
    @staticmethod
    def get_distance_grid(M: int, N: int) -> np.ndarray:
        """Computes distance matrix D(u, v) from center (M/2, N/2)."""
        u0, v0 = M / 2.0, N / 2.0
        u = np.arange(M).reshape(-1, 1) - u0
        v = np.arange(N).reshape(1, -1) - v0
        return np.sqrt(u**2 + v**2)

    @classmethod
    def ideal_lowpass(cls, M: int, N: int, D0: float) -> np.ndarray:
        D = cls.get_distance_grid(M, N)
        return (D <= D0).astype(float)

    @classmethod
    def ideal_highpass(cls, M: int, N: int, D0: float) -> np.ndarray:
        return 1.0 - cls.ideal_lowpass(M, N, D0)

    @classmethod
    def gaussian_lowpass(cls, M: int, N: int, D0: float) -> np.ndarray:
        D = cls.get_distance_grid(M, N)
        return np.exp(-(D**2) / (2.0 * (D0**2) + 1e-12))

    @classmethod
    def gaussian_highpass(cls, M: int, N: int, D0: float) -> np.ndarray:
        return 1.0 - cls.gaussian_lowpass(M, N, D0)

    @classmethod
    def butterworth_lowpass(cls, M: int, N: int, D0: float, order: int = 2) -> np.ndarray:
        D = cls.get_distance_grid(M, N)
        return 1.0 / (1.0 + (D / (D0 + 1e-12))**(2 * order))

    @classmethod
    def butterworth_highpass(cls, M: int, N: int, D0: float, order: int = 2) -> np.ndarray:
        D = cls.get_distance_grid(M, N)
        # Avoid division by zero at center
        safe_D = np.where(D == 0, 1e-12, D)
        return 1.0 / (1.0 + (D0 / safe_D)**(2 * order))

    @classmethod
    def get_filter(cls, filter_name: str, filter_mode: str, M: int, N: int, D0: float, order: int = 2) -> np.ndarray:
        name = filter_name.lower()
        mode = filter_mode.lower()

        if 'ideal' in name:
            return cls.ideal_lowpass(M, N, D0) if mode == 'lowpass' else cls.ideal_highpass(M, N, D0)
        elif 'gaussian' in name:
            return cls.gaussian_lowpass(M, N, D0) if mode == 'lowpass' else cls.gaussian_highpass(M, N, D0)
        elif 'butterworth' in name:
            return cls.butterworth_lowpass(M, N, D0, order) if mode == 'lowpass' else cls.butterworth_highpass(M, N, D0, order)
        else:
            return cls.ideal_lowpass(M, N, D0)


# ==============================================================================
# 3. METRICS, PARSEVAL THEOREM & RINGING QUANTIFICATION
# ==============================================================================

def compute_spectral_metrics(orig_img: np.ndarray, F: np.ndarray, filtered_F: np.ndarray, recon_img: np.ndarray, D0: float) -> dict:
    """
    Computes spatial energy, spectral energy (Parseval theorem),
    energy retention ratio, MSE, PSNR, and spatial ringing index.
    """
    M, N = orig_img.shape

    # 1. Parseval Theorem check
    total_spatial_energy = float(np.sum(orig_img ** 2))
    total_freq_energy = float((1.0 / (M * N)) * np.sum(np.abs(F) ** 2))
    parseval_diff = abs(total_spatial_energy - total_freq_energy)
    parseval_ratio = total_spatial_energy / (total_freq_energy + 1e-15) * 100.0

    # 2. Spectral Energy Retained
    filtered_freq_energy = float((1.0 / (M * N)) * np.sum(np.abs(filtered_F) ** 2))
    energy_retained_pct = (filtered_freq_energy / (total_freq_energy + 1e-15)) * 100.0

    # 3. MSE and PSNR (dB)
    mse = float(np.mean((orig_img - recon_img) ** 2))
    max_val = float(np.max(orig_img))
    if max_val <= 0:
        max_val = 1.0

    if mse < 1e-12:
        psnr = 150.0  # Numerical ceiling for identical images
    else:
        psnr = float(10.0 * np.log10((max_val ** 2) / mse))

    # 4. Ringing Artifact Quantification
    # Measure total variation of error residual (high frequency oscillations around edges)
    diff_img = np.abs(orig_img - recon_img)
    tv_diff = np.sum(np.abs(np.diff(diff_img, axis=0))) + np.sum(np.abs(np.diff(diff_img, axis=1)))
    tv_orig = np.sum(np.abs(np.diff(orig_img, axis=0))) + np.sum(np.abs(np.diff(orig_img, axis=1)))
    ringing_metric = float(tv_diff / (tv_orig + 1e-6))

    return {
        'total_spatial_energy': round(total_spatial_energy, 4),
        'total_freq_energy': round(total_freq_energy, 4),
        'parseval_diff': f"{parseval_diff:.2e}",
        'parseval_match_pct': round(parseval_ratio, 4),
        'energy_retained_pct': round(energy_retained_pct, 2),
        'mse': float(mse),
        'psnr_db': round(psnr, 2),
        'ringing_metric': round(ringing_metric, 4)
    }


def find_top_frequencies(F: np.ndarray, top_k: int = 5) -> list:
    """
    Finds top k dominant frequency components coordinates (u, v) and magnitudes.
    """
    M, N = F.shape
    mag = np.abs(F)

    # Flatten indices sorted descending
    flat_indices = np.argsort(mag.ravel())[::-1]

    results = []
    for idx in flat_indices[:top_k]:
        u = int(idx // N)
        v = int(idx % N)
        magnitude = float(mag[u, v])
        phase_rad = float(np.angle(F[u, v]))
        results.append({
            'rank': len(results) + 1,
            'u': u,
            'v': v,
            'magnitude': round(magnitude, 2),
            'phase_rad': round(phase_rad, 3),
            'phase_deg': round(float(np.degrees(phase_rad)), 1)
        })
    return results


# ==============================================================================
# 4. INVARIANT SPECTRAL FEATURE EXTRACTION (RADIAL RINGS & ANGULAR WEDGES)
# ==============================================================================

class InvariantSpectralFeatures:
    """
    Extracts Radial Ring and Angular Wedge Fourier energy descriptors.
    - Radial Rings (S_r): Rotation invariant (rotation in spatial domain rotates spectrum,
      leaving radial ring integrals unchanged).
    - Angular Wedges (S_theta): Scale invariant (scaling in spatial domain changes radial frequency,
      leaving angular wedge integrals unchanged).
    - Normalized by total spectral energy: Invariant to illumination scale and spatial translation.
    """
    @classmethod
    def extract(cls, image: np.ndarray, num_rings: int = 16, num_wedges: int = 16) -> np.ndarray:
        M, N = image.shape
        # Compute centered DFT
        F = Custom2DDFT.fft2(image, center=True)
        mag_sq = np.abs(F) ** 2

        u0, v0 = M / 2.0, N / 2.0
        u = np.arange(M).reshape(-1, 1) - u0
        v = np.arange(N).reshape(1, -1) - v0

        radius = np.sqrt(u**2 + v**2)
        # Angular orientation theta in [0, pi) due to conjugate symmetry |F(u, v)| = |F(-u, -v)|
        theta = np.arctan2(v, u) % np.pi

        # 1. Radial rings
        r_max = min(M, N) / 2.0
        ring_bins = np.linspace(0, r_max, num_rings + 1)
        Sr = np.zeros(num_rings, dtype=float)
        for i in range(num_rings):
            mask = (radius >= ring_bins[i]) & (radius < ring_bins[i+1])
            Sr[i] = np.sum(mag_sq[mask])

        # 2. Angular wedges
        wedge_bins = np.linspace(0, np.pi, num_wedges + 1)
        Stheta = np.zeros(num_wedges, dtype=float)
        for j in range(num_wedges):
            mask = (theta >= wedge_bins[j]) & (theta < wedge_bins[j+1])
            Stheta[j] = np.sum(mag_sq[mask])

        # Normalization by total spectral energy (translation and illumination invariance)
        total_energy = float(np.sum(mag_sq)) + 1e-12
        normalized_features = np.concatenate([Sr, Stheta]) / total_energy
        return normalized_features


# ==============================================================================
# 5. REFERENCE PATTERN LIBRARY & CLASSIFIER
# ==============================================================================

class PatternLibrary:
    _reference_db = None

    @classmethod
    def load_library(cls, dataset_dir: str):
        """Indexes reference images and precomputes invariant spectral features."""
        library = []
        if not os.path.exists(dataset_dir):
            cls._reference_db = []
            return

        image_files = sorted(glob.glob(os.path.join(dataset_dir, '*.png')) +
                            glob.glob(os.path.join(dataset_dir, '*.jpg')))

        for path in image_files:
            fname = os.path.basename(path)
            # Skip test queries or test images
            if fname.startswith('test_') or 'test' in fname:
                continue

            # Class name from filename: e.g. "vertical_stripes_1.png" -> "Vertical Stripes"
            base = os.path.splitext(fname)[0]
            clean_class = '_'.join(base.split('_')[:-1]) if '_' in base and base.split('_')[-1].isdigit() else base
            class_label = clean_class.replace('_', ' ').title()

            try:
                pil_img = Image.open(path).convert('L')
                pil_img = pil_img.resize((64, 64), Image.Resampling.LANCZOS)
                arr = np.array(pil_img, dtype=float) / 255.0
                feat = InvariantSpectralFeatures.extract(arr, num_rings=16, num_wedges=16)

                library.append({
                    'filename': fname,
                    'path': path,
                    'class_name': class_label,
                    'features': feat,
                    'shape': arr.shape,
                    'mean_val': float(np.mean(arr)),
                    'std_val': float(np.std(arr))
                })
            except Exception as e:
                print(f"Error loading {path}: {e}")

        cls._reference_db = library

    @classmethod
    def classify_query(cls, query_img: np.ndarray, top_k: int = 5) -> dict:
        """
        Extracts invariant spectral features from query image and finds
        nearest matches in reference library using Euclidean and Cosine distances.
        """
        if cls._reference_db is None or len(cls._reference_db) == 0:
            cls.load_library(app.config['DATASET_FOLDER'])

        if not cls._reference_db:
            return {
                'predicted_class': 'Unknown (Dataset Empty)',
                'confidence': 0.0,
                'matches': []
            }

        # Extract features
        query_feat = InvariantSpectralFeatures.extract(query_img, num_rings=16, num_wedges=16)
        query_norm = np.linalg.norm(query_feat) + 1e-12

        matches = []
        for item in cls._reference_db:
            ref_feat = item['features']
            ref_norm = np.linalg.norm(ref_feat) + 1e-12

            # Euclidean distance
            euclidean = float(np.linalg.norm(query_feat - ref_feat))
            # Cosine similarity
            cosine = float(np.dot(query_feat, ref_feat) / (query_norm * ref_norm))
            # Confidence score [0, 100]
            confidence = max(0.0, min(100.0, (1.0 / (1.0 + euclidean)) * 100.0))

            matches.append({
                'class_name': item['class_name'],
                'filename': item['filename'],
                'euclidean_distance': round(euclidean, 4),
                'cosine_similarity': round(cosine, 4),
                'confidence_pct': round(confidence, 1)
            })

        # Sort matches by Euclidean distance ascending
        matches.sort(key=lambda x: x['euclidean_distance'])

        best_match = matches[0]
        best_class = best_match['class_name']

        return {
            'predicted_class': best_class,
            'top_confidence': best_match['confidence_pct'],
            'top_cosine': best_match['cosine_similarity'],
            'matches': matches[:top_k]
        }


# ==============================================================================
# 6. APPLICATION STATE MANAGER
# ==============================================================================

class AppStateManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_image_path = None
        self.raw_filename = None
        self.image_info = {}
        self.preprocess_info = {}
        self.preprocessed_array = None  # Float [0, 1], (M, N)
        self.centering_enabled = True
        self.dft_results = {}
        self.filter_analysis = {}
        self.query_results = {}
        self.optimal_filter = {}

    def clear_plots(self):
        for f in glob.glob(os.path.join(app.config['PLOTS_FOLDER'], '*.*')):
            try:
                os.remove(f)
            except OSError:
                pass

    def clear_uploads(self):
        for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.*')):
            try:
                os.remove(f)
            except OSError:
                pass


state = AppStateManager()


# ==============================================================================
# 7. VISUALIZATION GENERATORS
# ==============================================================================

def generate_four_panel_plot(orig_img: np.ndarray, F: np.ndarray, recon_img: np.ndarray,
                             filter_name: str, D0: float, mode: str = 'lowpass') -> str:
    """
    Creates the required 4-Panel Visualization:
    (1) Pre-processed Spatial Image f(x, y)
    (2) Centered Log-Magnitude Spectrum log(1 + |F(u, v)|)
    (3) Phase Spectrum angle F(u, v)
    (4) Filtered & Reconstructed Spatial Image via 2D IDFT
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), facecolor='#0f172a')
    plt.subplots_adjust(wspace=0.25, top=0.86, bottom=0.1)

    titles = [
        "(1) Spatial Image f(x, y)",
        "(2) Log-Magnitude Spectrum",
        "(3) Phase Spectrum ∠F(u, v)",
        f"(4) Reconstructed ({filter_name.title()} D₀={D0})"
    ]

    # 1. Spatial image
    im0 = axes[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(titles[0], color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    axes[0].tick_params(colors='#94a3b8')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).ax.tick_params(colors='#94a3b8')

    # 2. Centered Log-Magnitude spectrum
    mag = np.abs(F)
    log_mag = np.log1p(mag)
    im1 = axes[1].imshow(log_mag, cmap='viridis')
    axes[1].set_title(titles[1], color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    axes[1].tick_params(colors='#94a3b8')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(colors='#94a3b8')

    # 3. Phase spectrum
    phase = np.angle(F)
    im2 = axes[2].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[2].set_title(titles[2], color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
    axes[2].tick_params(colors='#94a3b8')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(colors='#94a3b8')

    # 4. Reconstructed spatial image
    recon_disp = np.clip(recon_img, 0.0, 1.0)
    im3 = axes[3].imshow(recon_disp, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(titles[3], color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
    axes[3].tick_params(colors='#94a3b8')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04).ax.tick_params(colors='#94a3b8')

    for ax in axes:
        ax.set_facecolor('#0f172a')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    plot_filename = f"four_panel_{int(time.time()*1000)}.png"
    out_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
    plt.savefig(out_path, dpi=180, facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    return plot_filename


def generate_filter_comparison_curves(cutoffs: list, ideal_res: list, gauss_res: list, butter_res: list) -> str:
    """
    Plots dual graphs on the same figure:
    Graph 1: D0 vs Spectral Energy Retained (%)
    Graph 2: D0 vs PSNR (dB)
    Comparing Ideal, Gaussian, and Butterworth transfer functions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor='#0f172a')
    plt.subplots_adjust(wspace=0.28, top=0.88, bottom=0.14)

    # Style
    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.grid(True, linestyle='--', alpha=0.3, color='#64748b')
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#475569')

    # Plot 1: D0 vs Energy Retained (%)
    ax1.plot(cutoffs, [r['energy_retained_pct'] for r in ideal_res], 'o-', color='#38bdf8', lw=2.2, label='Ideal Filter')
    ax1.plot(cutoffs, [r['energy_retained_pct'] for r in gauss_res], 's--', color='#4ade80', lw=2.2, label='Gaussian Filter')
    ax1.plot(cutoffs, [r['energy_retained_pct'] for r in butter_res], '^-.', color='#f43f5e', lw=2.0, label='Butterworth (n=2)')
    ax1.axhline(95.0, color='#eab308', linestyle=':', lw=1.8, label='95% Retention Threshold')
    ax1.set_xlabel('Cutoff Frequency Radius ($D_0$)', color='#f8fafc', fontsize=11, labelpad=8)
    ax1.set_ylabel('Spectral Energy Retained (%)', color='#f8fafc', fontsize=11, labelpad=8)
    ax1.set_title('Cutoff Radius $D_0$ vs. Spectral Energy Retained (%)', color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

    # Plot 2: D0 vs PSNR (dB)
    # Clip inf/extreme psnr values for clean plotting
    def safe_psnr(val):
        return min(val, 80.0)

    ax2.plot(cutoffs, [safe_psnr(r['psnr_db']) for r in ideal_res], 'o-', color='#38bdf8', lw=2.2, label='Ideal Filter')
    ax2.plot(cutoffs, [safe_psnr(r['psnr_db']) for r in gauss_res], 's--', color='#4ade80', lw=2.2, label='Gaussian Filter')
    ax2.plot(cutoffs, [safe_psnr(r['psnr_db']) for r in butter_res], '^-.', color='#f43f5e', lw=2.0, label='Butterworth (n=2)')
    ax2.set_xlabel('Cutoff Frequency Radius ($D_0$)', color='#f8fafc', fontsize=11, labelpad=8)
    ax2.set_ylabel('Reconstruction PSNR (dB)', color='#f8fafc', fontsize=11, labelpad=8)
    ax2.set_title('Cutoff Radius $D_0$ vs. Reconstruction PSNR (dB)', color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)

    plot_filename = f"filter_sweep_{int(time.time()*1000)}.png"
    out_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
    plt.savefig(out_path, dpi=180, facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    return plot_filename


# ==============================================================================
# 8. FLASK APPLICATION ROUTES
# ==============================================================================

@app.route('/')
def index():
    # Discover available sample patterns in Dataset/
    sample_files = []
    if os.path.exists(app.config['DATASET_FOLDER']):
        files = sorted(glob.glob(os.path.join(app.config['DATASET_FOLDER'], '*.png')))
        for p in files:
            fname = os.path.basename(p)
            sample_files.append({
                'filename': fname,
                'name': fname.replace('.png', '').replace('_', ' ').title(),
                'is_query': fname.startswith('test_')
            })

    return render_template(
        'index.html',
        samples=sample_files,
        state=state,
        cache_buster=int(time.time())
    )


@app.route('/docs')
def documentation():
    """Serves the interactive documentation portal."""
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    return send_from_directory(docs_dir, 'index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image upload or sample dataset selection."""
    state.clear_plots()
    state.dft_results = {}
    state.filter_analysis = {}
    state.query_results = {}

    selected_sample = request.form.get('selected_sample')
    uploaded_file = request.files.get('image_file')

    save_path = None
    filename = None

    if selected_sample and selected_sample.strip():
        sample_path = os.path.join(app.config['DATASET_FOLDER'], selected_sample.strip())
        if os.path.exists(sample_path):
            filename = selected_sample.strip()
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shutil.copyfile(sample_path, save_path)
    elif uploaded_file and uploaded_file.filename:
        filename = f"user_{int(time.time())}_{uploaded_file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(save_path)

    if not save_path or not os.path.exists(save_path):
        return redirect(url_for('index'))

    # Inspect image properties
    try:
        pil_img = Image.open(save_path)
        w, h = pil_img.size
        mode = pil_img.mode
        channels = len(pil_img.getbands())
        arr = np.array(pil_img)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))

        # Check pre-processing requirements
        needs_grayscale = (mode != 'L' and channels > 1)
        needs_resize = not (w == 64 and h == 64)
        needs_normalize = (max_val > 1.0 or min_val < 0.0)

        state.raw_image_path = save_path
        state.raw_filename = filename
        state.image_info = {
            'width': w,
            'height': h,
            'mode': mode,
            'channels': channels,
            'min_val': round(min_val, 2),
            'max_val': round(max_val, 2),
            'mean_val': round(mean_val, 2),
            'std_val': round(std_val, 2),
            'aspect_ratio': round(w / h, 2),
            'needs_grayscale': needs_grayscale,
            'needs_resize': needs_resize,
            'needs_normalize': needs_normalize,
            'is_ready': not (needs_grayscale or needs_resize or needs_normalize)
        }

        # Auto pre-process default if user wishes or populate array
        if state.image_info['is_ready']:
            state.preprocessed_array = arr.astype(float)
            state.preprocess_info = {
                'processed': False,
                'message': 'Image is already Grayscale 64x64 with normalized intensity [0, 1]. No conversion needed.'
            }
        else:
            state.preprocessed_array = None
            state.preprocess_info = {'processed': False}

    except Exception as e:
        print(f"Error inspecting uploaded image: {e}")

    return redirect(url_for('index', _anchor='step-preprocess'))


@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    """Applies conditional pre-processing: Grayscale luminance weighting, Resizing, and Normalization."""
    if not state.raw_image_path or not os.path.exists(state.raw_image_path):
        return redirect(url_for('index'))

    target_size = int(request.form.get('target_size', 64))
    luminance_gray = request.form.get('convert_gray', 'true') == 'true'
    normalize_intensity = request.form.get('normalize', 'true') == 'true'

    try:
        pil_img = Image.open(state.raw_image_path)

        # 1. Grayscale Conversion using luminance weighting I = 0.299R + 0.587G + 0.114B
        if luminance_gray and pil_img.mode != 'L':
            if pil_img.mode in ('RGBA', 'LA'):
                # Handle alpha by compositing on white background
                bg = Image.new('RGB', pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = bg
            # Standard Pillow .convert('L') adheres strictly to ITU-R 601-2 luminance weights (0.299, 0.587, 0.114)
            pil_img = pil_img.convert('L')

        # 2. Spatial Resizing
        orig_w, orig_h = pil_img.size
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # 3. Pixel Intensity Normalization to [0, 1]
        arr = np.array(pil_img, dtype=float)
        if normalize_intensity:
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            else:
                arr = arr / 255.0

        state.preprocessed_array = arr
        state.preprocess_info = {
            'processed': True,
            'final_shape': f"{target_size} × {target_size}",
            'final_min': round(float(np.min(arr)), 4),
            'final_max': round(float(np.max(arr)), 4),
            'final_mean': round(float(np.mean(arr)), 4),
            'final_std': round(float(np.std(arr)), 4),
            'message': f"Pre-processing completed: Grayscale luminance applied, resized to {target_size}×{target_size}, normalized to [0, 1]."
        }

        # Save preprocessed image thumbnail in static/plots
        prep_thumb_filename = f"preprocessed_{int(time.time()*1000)}.png"
        prep_path = os.path.join(app.config['PLOTS_FOLDER'], prep_thumb_filename)
        disp_prep = (arr * 255.0).astype(np.uint8)
        Image.fromarray(disp_prep).save(prep_path)
        state.preprocess_info['thumbnail'] = prep_thumb_filename

    except Exception as e:
        print(f"Error pre-processing image: {e}")

    return redirect(url_for('index', _anchor='step-dft'))


@app.route('/analyze_dft', methods=['POST'])
def analyze_dft():
    """
    Computes 2D DFT from scratch, centering, dominant frequencies,
    Parseval theorem check, default filtering, and 4-panel visualizer.
    """
    if state.preprocessed_array is None:
        return redirect(url_for('index'))

    # Form parameters
    centering = request.form.get('centering_toggle') == 'on'
    filter_type = request.form.get('filter_type', 'ideal').lower()
    filter_mode = request.form.get('filter_mode', 'lowpass').lower()
    D0 = float(request.form.get('cutoff_d0', 15.0))

    img = state.preprocessed_array
    M, N = img.shape
    state.centering_enabled = centering

    # 1. From-scratch 2D DFT calculation
    t0 = time.perf_counter()
    F_scratch = Custom2DDFT.fft2(img, center=centering)
    scratch_time = (time.perf_counter() - t0) * 1000.0

    # 2. Verification vs np.fft
    verification = Custom2DDFT.verify_against_numpy(img, center=centering)

    # 3. DC component & Dominant frequencies
    dc_coord = (int(M // 2), int(N // 2)) if centering else (0, 0)
    dc_val = F_scratch[dc_coord[0], dc_coord[1]]
    dominant_freqs = find_top_frequencies(F_scratch, top_k=5)
    peak_mag = float(np.max(np.abs(F_scratch)))

    # 4. Frequency filtering with selected filter
    H = FrequencyFilters.get_filter(filter_type, filter_mode, M, N, D0=D0, order=2)
    G = H * F_scratch

    # 5. Spatial Reconstruction via 2D IDFT from scratch
    recon_img = Custom2DDFT.ifft2(G, center=centering)

    # 6. Parseval and Reconstruction Metrics
    metrics = compute_spectral_metrics(img, F_scratch, G, recon_img, D0)

    # 7. Generate 4-Panel Visualization
    plot_name = generate_four_panel_plot(img, F_scratch, recon_img, filter_type, D0, filter_mode)

    state.dft_results = {
        'dimensions': f"{M} × {N}",
        'centering': centering,
        'filter_name': filter_type.capitalize(),
        'filter_mode': filter_mode.capitalize(),
        'cutoff_D0': D0,
        'scratch_time_ms': round(scratch_time, 3),
        'numpy_time_ms': verification['time_numpy_ms'],
        'verification_error': f"{verification['max_abs_error']:.2e}",
        'idft_error': f"{verification['idft_max_error']:.2e}",
        'dc_coord': f"({dc_coord[0]}, {dc_coord[1]})",
        'dc_value': f"{dc_val.real:.2f} + {dc_val.imag:.2f}j",
        'peak_magnitude': round(peak_mag, 2),
        'dominant_freqs': dominant_freqs,
        'metrics': metrics,
        'plot_filename': plot_name
    }

    return redirect(url_for('index', _anchor='step-dft-results'))


@app.route('/filter_sweep', methods=['POST'])
def filter_sweep():
    """
    Evaluates multi-cutoff frequency response across Ideal, Gaussian, and Butterworth filters.
    Generates comparative dual plots (D0 vs Energy Retained, D0 vs PSNR) and determines optimal D0*.
    """
    if state.preprocessed_array is None:
        return redirect(url_for('index'))

    # Parse cutoff radii list
    raw_cutoffs = request.form.get('cutoffs', '5, 10, 15, 20, 25, 30, 40, 50')
    filter_mode = request.form.get('sweep_mode', 'lowpass').lower()

    try:
        cutoffs = sorted(list(set([float(x.strip()) for x in raw_cutoffs.split(',') if x.strip()])))
    except ValueError:
        cutoffs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    img = state.preprocessed_array
    M, N = img.shape
    centering = state.centering_enabled

    # Forward DFT
    F = Custom2DDFT.fft2(img, center=centering)

    ideal_results = []
    gaussian_results = []
    butter_results = []

    for d in cutoffs:
        # Ideal
        H_id = FrequencyFilters.get_filter('ideal', filter_mode, M, N, D0=d)
        G_id = H_id * F
        rec_id = Custom2DDFT.ifft2(G_id, center=centering)
        m_id = compute_spectral_metrics(img, F, G_id, rec_id, d)
        m_id['d0'] = d
        ideal_results.append(m_id)

        # Gaussian
        H_ga = FrequencyFilters.get_filter('gaussian', filter_mode, M, N, D0=d)
        G_ga = H_ga * F
        rec_ga = Custom2DDFT.ifft2(G_ga, center=centering)
        m_ga = compute_spectral_metrics(img, F, G_ga, rec_ga, d)
        m_ga['d0'] = d
        gaussian_results.append(m_ga)

        # Butterworth
        H_bu = FrequencyFilters.get_filter('butterworth', filter_mode, M, N, D0=d, order=2)
        G_bu = H_bu * F
        rec_bu = Custom2DDFT.ifft2(G_bu, center=centering)
        m_bu = compute_spectral_metrics(img, F, G_bu, rec_bu, d)
        m_bu['d0'] = d
        butter_results.append(m_bu)

    # Optimal Cutoff Selection Algorithm:
    # 1. Preserves at least 95% total spectral energy
    # 2. Maximizes PSNR (dB)
    # 3. Minimizes ringing artifacts (Gaussian / Butterworth generally have zero/minimal ringing compared to Ideal)
    candidates = []
    for r in gaussian_results + butter_results + ideal_results:
        if r['energy_retained_pct'] >= 95.0:
            candidates.append(r)

    if not candidates:
        # Fallback to candidate with maximum energy retention
        candidates = sorted(gaussian_results + butter_results, key=lambda x: x['energy_retained_pct'], reverse=True)

    # Score candidates: weighted combination of PSNR, energy retention, and penalizing ringing
    def opt_score(item):
        return item['psnr_db'] * 1.5 + item['energy_retained_pct'] * 1.0 - item['ringing_metric'] * 20.0

    optimal_item = max(candidates, key=opt_score)

    # Determine filter name for optimal
    opt_filter_name = "Gaussian"
    if optimal_item in ideal_results:
        opt_filter_name = "Ideal"
    elif optimal_item in butter_results:
        opt_filter_name = "Butterworth"

    state.optimal_filter = {
        'filter_name': opt_filter_name,
        'd0': optimal_item['d0'],
        'energy_retained_pct': optimal_item['energy_retained_pct'],
        'psnr_db': optimal_item['psnr_db'],
        'mse': float(optimal_item['mse']),
        'ringing_metric': optimal_item['ringing_metric']
    }

    # Generate comparative curves plot
    plot_name = generate_filter_comparison_curves(cutoffs, ideal_results, gaussian_results, butter_results)

    state.filter_analysis = {
        'mode': filter_mode.capitalize(),
        'cutoffs': cutoffs,
        'ideal_results': ideal_results,
        'gaussian_results': gaussian_results,
        'butter_results': butter_results,
        'plot_filename': plot_name,
        'optimal': state.optimal_filter
    }

    return redirect(url_for('index', _anchor='step-filter-sweep'))


@app.route('/predict_query', methods=['POST'])
def predict_query():
    """
    Validates uploaded query image, checks aspect ratio & contrast outliers,
    extracts invariant Radial Ring & Angular Wedge descriptors, and classifies pattern.
    """
    selected_query = request.form.get('selected_query')
    query_file = request.files.get('query_image_file')

    query_path = None
    filename = None

    if selected_query and selected_query.strip():
        sp = os.path.join(app.config['DATASET_FOLDER'], selected_query.strip())
        if os.path.exists(sp):
            filename = selected_query.strip()
            query_path = os.path.join(app.config['UPLOAD_FOLDER'], f"query_{filename}")
            shutil.copyfile(sp, query_path)
    elif query_file and query_file.filename:
        filename = f"query_{int(time.time())}_{query_file.filename}"
        query_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query_file.save(query_path)

    if not query_path or not os.path.exists(query_path):
        return redirect(url_for('index'))

    # Validation & Outlier checks
    warnings = []
    try:
        pil_img = Image.open(query_path)
        w, h = pil_img.size
        aspect_ratio = w / h

        # Supported formats check
        ext = os.path.splitext(query_path)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.bmp'):
            warnings.append(f"Format {ext} may have compression artifacts; standard formats (.png, .jpg, .bmp) recommended.")

        # Aspect ratio outlier check
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            warnings.append(f"Aspect ratio ({aspect_ratio:.2f}) deviates significantly from 1:1 reference domain.")

        # Preprocessing query image: convert to Grayscale & resize to 64x64
        pil_gray = pil_img.convert('L').resize((64, 64), Image.Resampling.LANCZOS)
        arr = np.array(pil_gray, dtype=float)

        # Contrast check (standard deviation of pixel intensities)
        std_val = float(np.std(arr))
        if std_val < 10.0:
            warnings.append(f"Low contrast warning (std dev = {std_val:.1f}). Image has very low spectral dynamics.")
        elif std_val > 115.0:
            warnings.append(f"Extreme contrast warning (std dev = {std_val:.1f}). Strong saturation detected.")

        # Intensity normalization
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = arr / 255.0

        # Classification matching
        res = PatternLibrary.classify_query(arr, top_k=5)

        # Extract features for radial and wedge plots
        feat = InvariantSpectralFeatures.extract(arr, num_rings=16, num_wedges=16)
        rings = feat[:16]
        wedges = feat[16:]

        # Generate feature bar plot
        fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(14, 3.8), facecolor='#0f172a')
        plt.subplots_adjust(wspace=0.25, bottom=0.18)

        for ax in (ax_r, ax_w):
            ax.set_facecolor('#1e293b')
            ax.grid(True, linestyle='--', alpha=0.3, color='#64748b')
            ax.tick_params(colors='#cbd5e1', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#475569')

        ax_r.bar(range(1, 17), rings, color='#38bdf8', edgecolor='#0284c7', alpha=0.85)
        ax_r.set_xlabel('Concentric Radial Ring Index ($S_r$)', color='#f8fafc', fontsize=10)
        ax_r.set_ylabel('Normalized Energy', color='#f8fafc', fontsize=10)
        ax_r.set_title('Radial Ring Energy Spectrum (Rotation Invariant)', color='#f8fafc', fontsize=11, fontweight='bold')

        ax_w.bar(range(1, 17), wedges, color='#a855f7', edgecolor='#7e22ce', alpha=0.85)
        ax_w.set_xlabel('Angular Wedge Sector Index ($S_\\theta$)', color='#f8fafc', fontsize=10)
        ax_w.set_ylabel('Normalized Energy', color='#f8fafc', fontsize=10)
        ax_w.set_title('Angular Wedge Energy Spectrum (Scale Invariant)', color='#f8fafc', fontsize=11, fontweight='bold')

        feat_plot_name = f"query_features_{int(time.time()*1000)}.png"
        plt.savefig(os.path.join(app.config['PLOTS_FOLDER'], feat_plot_name), dpi=180, facecolor='#0f172a', bbox_inches='tight')
        plt.close(fig)

        # Query thumbnail
        query_thumb_name = f"query_thumb_{int(time.time()*1000)}.png"
        Image.fromarray((arr * 255.0).astype(np.uint8)).save(os.path.join(app.config['PLOTS_FOLDER'], query_thumb_name))

        state.query_results = {
            'filename': filename,
            'aspect_ratio': round(aspect_ratio, 2),
            'std_contrast': round(std_val, 2),
            'warnings': warnings,
            'predicted_class': res['predicted_class'],
            'confidence': res['top_confidence'],
            'cosine_sim': res['top_cosine'],
            'matches': res['matches'],
            'feature_plot': feat_plot_name,
            'thumbnail': query_thumb_name
        }

    except Exception as e:
        print(f"Error classifying query: {e}")

    return redirect(url_for('index', _anchor='step-classification'))


@app.route('/cleanup', methods=['POST'])
def cleanup_temp_files():
    """Garbage Collector: clears generated plots, uploads, and triggers gc.collect()."""
    state.clear_plots()
    state.clear_uploads()
    collected = gc.collect()
    return jsonify({
        'status': 'success',
        'message': f'Temporary uploads and plot cache cleaned. Garbage collector reclaimed {collected} references.'
    })


@app.route('/reset', methods=['POST'])
def reset_studio():
    """Resets application state and clears generated assets."""
    state.reset()
    state.clear_plots()
    state.clear_uploads()
    gc.collect()
    return redirect(url_for('index'))


@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    """Terminates the Flask server process cleanly."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()
    else:
        # Fallback termination
        os._exit(0)
    return 'Server shutting down...'


if __name__ == '__main__':
    # Initialize reference database
    PatternLibrary.load_library(app.config['DATASET_FOLDER'])
    print(f"Initialized PatternLibrary with {len(PatternLibrary._reference_db or [])} reference patterns.")
    app.run(host='0.0.0.0', port=5000, debug=False)
