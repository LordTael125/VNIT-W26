# Pattern Recognition Laboratory: Assignment 10
## 2D Discrete Fourier Transform (2D DFT) Spectral Analysis, Frequency-Domain Filtering, and Invariant Feature Extraction

**Course:** Pattern Recognition Laboratory (VNIT-W26)  
**Evaluation:** Maximum Marks: 10 (Attendance: 02 Marks, Assignment Completion & Correctness: 04 Marks, Viva Voce: 04 Marks)  
**Lab Session Date:** 01 December 2026  
**Implementation:** Strictly from-scratch 2D DFT & 2D IDFT in Python / NumPy, Flask Web ML Studio.

---

## 1. Problem Statement & Objectives

Develop a comprehensive full-stack web application using the Flask framework that allows users to upload standard grayscale or color images (as single test images or labeled multi-class image sets), inspect image properties, and perform necessary pre-processing.

The application transforms spatial images into the frequency domain using the **2D Discrete Fourier Transform (2D DFT)** and reconstructs them via the **2D Inverse DFT (2D IDFT)** implemented strictly from scratch using NumPy matrix algebra. It investigates the effect of cutoff frequency $D_0$ and filter transfer functions on spectral energy retention and reconstruction quality, chooses an optimal cutoff threshold $D_0^*$, visualizes the centered magnitude and phase spectra along with reconstructed images, extracts translation- and rotation-invariant Fourier features (radial ring and angular wedge descriptors), and classifies newly uploaded unseen image instances.

### Key Objectives
1. **Interactive Flask Web Application:** Clean, responsive Dark-Theme dashboard supporting end-to-end ingestion, conditional pre-processing, spectral inspection, filtering sweeps, and pattern matching.
2. **Conditional Image Pre-Processing:** Only display pre-processing options (grayscale conversion via luminance weighting $I = 0.299R + 0.587G + 0.114B$, spatial resizing to $64 \times 64$ or $128 \times 128$, pixel intensity normalization to $[0, 1]$) if the uploaded image actually requires them; otherwise display a confirmation banner.
3. **Pure From-Scratch 2D DFT & 2D IDFT:** Separable matrix algebra formulation ($F = W_M \cdot ((-1)^{x+y} f) \cdot W_N$) executing in $< 2$ milliseconds without calling `np.fft` transformation routines.
4. **Frequency Centering via Spatial Modulation:** Observe zero-frequency (DC) component $F(0, 0)$ shift to the visual frequency center $(M/2, N/2)$ by multiplying the spatial image by $(-1)^{x+y}$.
5. **Parseval's Theorem & Spectral Metrics:** Verify energy conservation down to floating-point precision ($\Delta < 10^{-12}$), determine % spectral energy retained, MSE, PSNR (dB), and spatial ringing artifact indices.
6. **Cutoff Frequency ($D_0$) and Filter Type Analysis:** Comparative sweeps across Ideal, Gaussian, and Butterworth transfer functions for multiple radii $D_0 \in [5, 10, 15, 20, 25, 30, 40, 50]$, generating dual comparative curves ($D_0$ vs. Energy Retained and $D_0$ vs. PSNR) and determining optimal cutoff $D_0^*$.
7. **Invariant Feature Extraction & Classification:** Compute Radial Ring ($S_r$, rotation-invariant) and Angular Wedge ($S_\theta$, scale-invariant) energy descriptors normalized by total spectral energy (translation- and illumination-invariant), classify unseen query instances against reference patterns.

---

## 2. Mathematical Formulations & Derivations

### 2.1 2D Discrete Fourier Transform (Forward DFT)
For a 2D discrete spatial image $f(x, y)$ of dimensions $M \times N$, where $x \in \{0, \dots, M-1\}$ and $y \in \{0, \dots, N-1\}$, the standard 2D DFT is defined as:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \exp\left[-j 2\pi \left(\frac{u x}{M} + \frac{v y}{N}\right)\right]$$

for $u = 0, 1, \dots, M-1$ and $v = 0, 1, \dots, N-1$.

#### Separable Matrix Formulation
The exponential kernel is separable:
$$\exp\left[-j 2\pi \left(\frac{u x}{M} + \frac{v y}{N}\right)\right] = \exp\left(-j \frac{2\pi u x}{M}\right) \exp\left(-j \frac{2\pi v y}{N}\right)$$

Define the 1D Fourier transformation matrices $W_M \in \mathbb{C}^{M \times M}$ and $W_N \in \mathbb{C}^{N \times N}$:
$$(W_M)_{u, x} = \exp\left(-j \frac{2\pi u x}{M}\right), \quad (W_N)_{y, v} = \exp\left(-j \frac{2\pi y v}{N}\right)$$

The 2D DFT can therefore be expressed as two consecutive matrix multiplications:
$$F = W_M \cdot f \cdot W_N$$
- $W_M \cdot f$ transforms each column of $f$ independently.
- Post-multiplying by $W_N$ transforms each row independently.
- Complexity: Reduced from naive 4-loop $\mathcal{O}(M^2 N^2)$ to BLAS matrix multiplication $\mathcal{O}(M^2 N + M N^2)$, taking $< 1.5$ ms for a $64 \times 64$ image in pure NumPy.

---

### 2.2 Spatial Frequency Centering Property
Standard 2D DFT places the zero-frequency DC component $F(0, 0)$ at the top-left corner of the matrix. To position the DC term at the visual center $(M/2, N/2)$, we multiply the spatial domain image by $(-1)^{x+y}$ before transformation.

#### Mathematical Proof:
Recall Euler's identity:
$$(-1)^{x+y} = \exp[j \pi (x + y)] = \exp\left[j 2\pi \left(\frac{x \cdot (M/2)}{M} + \frac{y \cdot (N/2)}{N}\right)\right]$$
By the frequency shifting theorem of the Fourier Transform:
$$\mathcal{F}\left\{f(x, y) \exp\left[j 2\pi \left(\frac{u_0 x}{M} + \frac{v_0 y}{N}\right)\right]\right\} = F(u - u_0, v - v_0)$$
Setting $u_0 = M/2$ and $v_0 = N/2$:
$$\mathcal{F}\{f(x, y) \cdot (-1)^{x+y}\} = F\left(u - \frac{M}{2}, v - \frac{N}{2}\right)$$
This shifts the origin $(0, 0)$ to the matrix center $(M/2, N/2)$ without requiring any post-hoc circular array shifts (`np.fft.fftshift`).

---

### 2.3 2D Inverse Discrete Fourier Transform (2D IDFT)
The spatial image is reconstructed via the 2D IDFT:

$$f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) \exp\left[+j 2\pi \left(\frac{u x}{M} + \frac{v y}{N}\right)\right]$$

Using the conjugate transpose (Hermitian) of the Fourier matrices:
$$(W_M^*)_{x, u} = \exp\left(+j \frac{2\pi u x}{M}\right), \quad (W_N^*)_{v, y} = \exp\left(+j \frac{2\pi y v}{N}\right)$$
The matrix formulation for IDFT is:
$$f_{\text{reconstructed}} = \frac{1}{MN} \left( W_M^* \cdot F \cdot W_N^* \right)$$
If spatial centering was applied during forward DFT, uncentering is achieved by multiplying the reconstructed spatial image by $(-1)^{x+y}$:
$$f(x, y) = (-1)^{x+y} \cdot \text{Re}\left(f_{\text{reconstructed}}(x, y)\right)$$
Since $(-1)^{x+y} \cdot (-1)^{x+y} = (-1)^{2(x+y)} = +1$, this operation recovers the original image.

---

### 2.4 Parseval's Theorem (Energy Conservation)
Parseval's theorem states that the total energy in the spatial domain equals the normalized energy in the frequency domain:

$$\sum_{x=0}^{M-1} \sum_{y=0}^{N-1} |f(x, y)|^2 = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} |F(u, v)|^2$$

- **Physical Meaning:** The Discrete Fourier Transform is a unitary transform up to the scaling factor $1/\sqrt{MN}$. No energy is created or destroyed during transformation.

---

### 2.5 Frequency Domain Filter Transfer Functions
Filtering in the frequency domain is computed by point-wise multiplication:
$$G(u, v) = H(u, v) \cdot F(u, v)$$
where $D(u, v) = \sqrt{(u - M/2)^2 + (v - N/2)^2}$ is the Euclidean distance from the DC center $(M/2, N/2)$.

| Filter Family | Low-Pass Transfer Function $H_{LP}(u, v)$ | High-Pass Transfer Function $H_{HP}(u, v)$ |
| :--- | :--- | :--- |
| **Ideal** | $\begin{cases} 1 & D(u, v) \le D_0 \\ 0 & D(u, v) > D_0 \end{cases}$ | $1 - H_{ILPF}(u, v) = \begin{cases} 0 & D(u, v) \le D_0 \\ 1 & D(u, v) > D_0 \end{cases}$ |
| **Gaussian** | $\exp\left(-\frac{D^2(u, v)}{2 D_0^2}\right)$ | $1 - H_{GLPF}(u, v) = 1 - \exp\left(-\frac{D^2(u, v)}{2 D_0^2}\right)$ |
| **Butterworth ($n=2$)** | $\frac{1}{1 + \left[\frac{D(u, v)}{D_0}\right]^{2n}}$ | $\frac{1}{1 + \left[\frac{D_0}{D(u, v)}\right]^{2n}}$ |

#### Ringing Artifacts & Gibbs Phenomenon:
- **Ideal Filters:** Abrupt rectangular cutoff in frequency corresponds to a continuous spatial sinc function $h(x, y) \sim \frac{J_1(r)}{r}$ with decaying side-lobes. Convolving this sinc with sharp edges produces severe ripple oscillations (ringing artifacts).
- **Gaussian Filters:** The Fourier transform of a Gaussian is another Gaussian with no side-lobes. Hence, Gaussian filters produce completely ringing-free spatial smoothing.
- **Butterworth Filters:** Smooth transition between passband and stopband controlled by order $n$. At order $n=2$, it provides a sharp cutoff with negligible spatial ringing.

---

### 2.6 Reconstruction Quality Metrics
1. **Percentage of Spectral Energy Retained within $D_0$:**
   $$\text{Energy Retained (\%)} = \frac{\sum_{u, v} |G(u, v)|^2}{\sum_{u, v} |F(u, v)|^2} \times 100\%$$

2. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{MN} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [f(x, y) - \hat{f}(x, y)]^2$$

3. **Peak Signal-to-Noise Ratio (PSNR in dB):**
   $$\text{PSNR} = 10 \log_{10}\left(\frac{\max(f)^2}{\text{MSE} + \epsilon}\right)$$

4. **Spatial Ringing Index (TV Residual Ratio):**
   $$\text{Ringing Metric} = \frac{\text{TV}(|f - \hat{f}|)}{\text{TV}(f) + \epsilon}$$
   where $\text{TV}(A) = \sum |\nabla_x A| + \sum |\nabla_y A|$. High ripple oscillations around edges directly increase the Total Variation of the error residual.

---

### 2.7 Invariant Spectral Feature Extraction (Radial Rings & Angular Wedges)
To perform robust pattern recognition invariant to spatial translation, rotation, scale, and uniform illumination:

1. **Radial Ring Descriptors ($S_r$):**
   Concentric annular rings centered at $(u_0, v_0)$:
   $$S_r(k) = \sum_{r_k \le D(u, v) < r_{k+1}} |F(u, v)|^2, \quad k = 0, 1, \dots, R-1$$
   - **Rotation Invariance:** Rotating the spatial image by angle $\phi$ rotates its 2D Fourier spectrum by the exact same angle $\phi$. Since ring integration integrates over all angles $\theta \in [0, 2\pi)$, rotating the spectrum merely shifts the integration limits cyclically around the ring, preserving total energy within each radial annulus.

2. **Angular Wedge Descriptors ($S_\theta$):**
   Directional sectors radiating from the DC center:
   $$S_\theta(m) = \sum_{\theta_m \le \theta(u, v) < \theta_{m+1}} |F(u, v)|^2, \quad m = 0, 1, \dots, A-1$$
   where $\theta(u, v) = \arctan2(v - v_0, u - u_0) \pmod \pi \in [0, \pi)$ (exploiting conjugate symmetry $|F(u, v)| = |F(-u, -v)|$).
   - **Scale Invariance:** Scaling an image by factor $a$ in space scales its frequency spectrum inversely by $1/a$. This expands or compresses frequencies radially along rays without changing the energy distribution across angular sectors.

3. **Concatenation & Normalization:**
   $$\mathbf{v} = \frac{[S_r(0), \dots, S_r(R-1), S_\theta(0), \dots, S_\theta(A-1)]^T}{\sum_{u, v} |F(u, v)|^2 + \epsilon}$$
   - **Translation Invariance:** Translating an image by $(x_0, y_0)$ induces only a linear phase shift:
     $$\mathcal{F}\{f(x - x_0, y - y_0)\} = F(u, v) \exp\left[-j 2\pi \left(\frac{u x_0}{M} + \frac{v y_0}{N}\right)\right]$$
     Because $|\exp(j \theta)| = 1$, the magnitude spectrum $|F(u, v)|$ is **strictly identical**. Hence, all ring and wedge descriptors are completely translation-invariant!
   - **Illumination Invariance:** Uniform illumination scaling $c \cdot f(x, y)$ scales spectral energy by $c^2$, which cancels out upon dividing by the total spectral energy.

---

## 3. Directory Structure & Architecture

```text
PatternReco-Assignment-10/
├── PR_Lab_Assignment_10_Image_DFT.pdf   # Official VNIT assignment specification
├── main.py                             # Core Flask app, from-scratch 2D DFT engine, filters, features
├── test_verification.py                # Automated test suite (17/17 tests passing)
├── README.md                           # Comprehensive lab report and viva guide
├── Dataset/                            # Curated pattern archetypes & test queries
│   ├── vertical_stripes_1.png
│   ├── vertical_stripes_2.png
│   ├── horizontal_stripes_1.png
│   ├── horizontal_stripes_2.png
│   ├── diagonal_stripes_1.png
│   ├── diagonal_stripes_2.png
│   ├── concentric_rings_1.png
│   ├── concentric_rings_2.png
│   ├── checkerboard_1.png
│   ├── checkerboard_2.png
│   ├── radial_star_1.png
│   ├── radial_star_2.png
│   ├── dot_lattice_1.png
│   ├── dot_lattice_2.png
│   ├── natural_texture_1.png
│   ├── natural_texture_2.png
│   ├── test_vertical_shifted.png       # Test query: Translated vertical stripes
│   ├── test_horizontal_scaled.png      # Test query: Scaled contrast horizontal stripes
│   ├── test_concentric_rotated.png     # Test query: Rotated concentric rings
│   ├── test_diagonal_noisy.png         # Test query: Noisy diagonal stripes
│   ├── test_checkerboard_shifted.png   # Test query: Translated checkerboard
│   └── ...
├── Uploads/                            # User-uploaded images
├── Templates/
│   └── index.html                      # Dark-theme UI with 6 workflow steps and KaTeX
└── Statics/
    ├── css/
    │   └── style.css                   # Glassmorphic dark styling, responsive grid
    ├── js/
    │   └── app.js                      # Safe submit listeners, KaTeX math auto-render, AJAX cleanup
    ├── plots/                          # Generated 4-panel visualizers & comparison curves
    └── vendor/
        └── katex/                      # Offline KaTeX math rendering assets
```

---

## 4. How to Run & Verify

### 4.1 Running the Automated Test Suite
The repository includes an automated test verification suite covering from-scratch DFT numerical parity, Parseval's theorem, centering properties, filter sweeps, invariance proofs, and Flask HTTP endpoints:

```bash
cd PatternReco-Assignment-10
python3 -m unittest -v test_verification.py
```
**Expected Output:**
```text
test_centered_dft_accuracy ... ok
test_centering_dc_component_location ... ok
test_idft_exact_reconstruction ... ok
test_parseval_theorem ... ok
test_uncentered_dft_accuracy ... ok
test_analyze_dft_and_filters ... ok
test_cleanup_garbage_collector ... ok
test_filter_sweep_and_optimization ... ok
test_index_route ... ok
test_predict_query_endpoint ... ok
test_upload_sample_and_preprocess ... ok
test_filter_shapes_and_extrema ... ok
test_multi_d0_monotonicity ... ok
test_illumination_scaling_invariance ... ok
test_radial_ring_rotation_invariance ... ok
test_translation_invariance ... ok
test_query_instances_classification ... ok

----------------------------------------------------------------------
Ran 17 tests in 1.492s

OK
```

### 4.2 Starting the Flask Application
```bash
python3 main.py
```
Open a browser and navigate to:
```text
http://localhost:5000/
```

### 4.3 End-to-End Workflow in the UI
1. **Step 1 (Ingestion):** Choose a reference pattern from the dropdown (e.g. `Checkerboard 1`) or upload an image.
2. **Step 2 (Conditional Pre-processing):** If the image is already Grayscale $64 \times 64$ with normalized intensity in $[0, 1]$, a confirmation banner is displayed. Otherwise, click "Execute Image Pre-Processing Pipeline".
3. **Step 3 (2D DFT Spectral Analysis):** Toggle frequency centering $(-1)^{x+y}$, select filter (Default: Ideal LPF, $D_0 = 15$), and click "Execute From-Scratch 2D DFT". Inspect execution times, numerical error ($< 10^{-11}$), DC component coordinates, top 5 dominant frequency peaks, Parseval match (100.000%), and the 4-panel visualization.
4. **Step 4 (Filter Sweep & Optimization):** Run candidate cutoff radii $D_0 \in [5, 10, 15, 20, 25, 30, 40, 50]$. Review the dual comparative performance curves ($D_0$ vs. Energy Retained and $D_0$ vs. PSNR) and inspect the optimal cutoff $D_0^*$ card ($\ge 95\%$ energy retention criterion).
5. **Step 5 (Feature Framework):** Review the mathematical formulation of rotation-invariant radial rings ($S_r$) and scale-invariant angular wedges ($S_\theta$).
6. **Step 6 (Query Classification):** Select an unseen query image (e.g., `Test Vertical Shifted`, `Test Concentric Rotated`) to classify it against reference classes using Euclidean distance and Cosine similarity.
7. **Housekeeping:** Click **Clean Temp Files** in the navigation bar to trigger server-side garbage collection and clean plot caches.

---

## 5. Viva Voce Preparation Guide (Questions & Answers)

### Q1: Why can the 2D DFT be computed via separable 1D matrix multiplications?
**Answer:** The 2D Fourier kernel $\exp[-j 2\pi(ux/M + vy/N)]$ is separable into the product of two 1D kernels: $\exp(-j 2\pi ux/M) \cdot \exp(-j 2\pi vy/N)$. Consequently, the 2D transformation $F(u, v)$ can be computed as $F = W_M \cdot f \cdot W_N$, where $W_M$ transforms the columns and $W_N$ transforms the rows. This reduces computational complexity from $\mathcal{O}(M^2 N^2)$ to $\mathcal{O}(M^2 N + M N^2)$.

### Q2: What is the mathematical basis for frequency centering using $(-1)^{x+y}$?
**Answer:** Euler's identity gives $(-1)^{x+y} = \exp[j \pi (x + y)] = \exp[j 2\pi (x(M/2)/M + y(N/2)/N)]$. According to the Fourier frequency translation property, multiplying the spatial domain image by a complex exponential $\exp[j 2\pi (u_0 x / M + v_0 y / N)]$ shifts the frequency spectrum by $(u_0, v_0)$. Setting $u_0 = M/2$ and $v_0 = N/2$ shifts the zero-frequency DC component $F(0, 0)$ from the top-left corner to the visual center $(M/2, N/2)$ of the frequency plane.

### Q3: What is Parseval's theorem, and how is it verified in this assignment?
**Answer:** Parseval's theorem states that the total energy in the spatial domain equals the normalized total energy in the frequency domain:
$$\sum_{x=0}^{M-1} \sum_{y=0}^{N-1} |f(x, y)|^2 = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} |F(u, v)|^2$$
In our application, we compute both quantities independently using 64-bit floating point arithmetic. The absolute difference is on the order of $10^{-13}$, verifying exact numerical energy conservation.

### Q4: Why does an Ideal Low-Pass Filter cause ringing artifacts, while a Gaussian filter does not?
**Answer:** In the frequency domain, an Ideal Low-Pass Filter has an abrupt rectangular step function. By the convolution theorem, frequency domain multiplication corresponds to spatial domain convolution with the inverse Fourier transform of the filter. The inverse transform of a 2D circular cylinder is a Bessel function of the first kind divided by radius $\frac{J_1(2\pi D_0 r)}{r}$ (analogous to a 1D sinc function), which contains prominent negative and positive decaying side-lobes. Convolving this sinc with sharp edges in an image causes Gibbs phenomenon oscillations (ringing artifacts). In contrast, the inverse Fourier transform of a 2D Gaussian function is another smooth Gaussian with zero side-lobes; thus, Gaussian filtering produces completely ringing-free smoothing.

### Q5: How do Butterworth filters strike a compromise between Ideal and Gaussian filters?
**Answer:** A Butterworth filter has transfer function $H(u, v) = \frac{1}{1 + [D(u, v)/D_0]^{2n}}$. Unlike the Ideal filter, it has no discontinuity, avoiding severe Gibbs oscillations. Unlike the Gaussian filter, its passband is flatter, allowing high transmission close to $D_0$ with faster rolloff in the stopband. At order $n=2$, it provides sharp frequency cutoff with minimal spatial ringing.

### Q6: Why are Radial Ring descriptors invariant to image rotation?
**Answer:** When an image $f(x, y)$ is rotated by angle $\phi$ in the spatial domain, its 2D Fourier spectrum $|F(u, v)|$ rotates by the identical angle $\phi$. Because the radial ring descriptors $S_r(k)$ integrate the squared spectral magnitude $|F(u, v)|^2$ over the entire angular span $\theta \in [0, 2\pi)$ for concentric annular rings of radius $[r_k, r_{k+1})$, spatial rotation merely shifts the circular integral cyclically, leaving the total energy in each concentric ring unchanged.

### Q7: Why are Angular Wedge descriptors invariant to image scaling?
**Answer:** Scaling an image in space by factor $a$ (i.e., $f(a x, a y)$) scales its Fourier transform inversely in frequency by $1/a$ (i.e., $\frac{1}{|a|^2} F(u/a, v/a)$). In polar frequency coordinates $(r, \theta)$, this stretches or compresses the spectrum along radial rays ($r \to r/a$) while keeping the angular orientation $\theta$ constant. Because wedge descriptors $S_\theta(m)$ integrate over radial rays within a given angular sector $[\theta_m, \theta_{m+1})$, the relative energy fraction across angular wedges remains scale-invariant.

### Q8: Why are Fourier magnitude features strictly invariant to spatial translation?
**Answer:** By the Fourier shift theorem, translating an image by $(x_0, y_0)$ in the spatial domain introduces a linear phase shift in the frequency domain:
$$\mathcal{F}\{f(x - x_0, y - y_0)\} = F(u, v) \cdot \exp\left[-j 2\pi \left(\frac{u x_0}{M} + \frac{v y_0}{N}\right)\right]$$
Taking the magnitude:
$$|F_{\text{shifted}}(u, v)| = |F(u, v)| \cdot \left|\exp\left[-j 2\pi \left(\frac{u x_0}{M} + \frac{v y_0}{N}\right)\right]\right| = |F(u, v)| \cdot 1 = |F(u, v)|$$
The magnitude spectrum is identical under spatial translation; therefore, any feature extracted from $|F(u, v)|$ (including radial rings and angular wedges) is invariant to spatial translation.

### Q9: Why is the angular wedge orientation evaluated over $[0, \pi)$ rather than $[0, 2\pi)$?
**Answer:** For any real-valued image $f(x, y) \in \mathbb{R}$, its Fourier transform exhibits Hermitian (conjugate) symmetry:
$$F(-u, -v) = F^*(u, v) \implies |F(-u, -v)| = |F(u, v)|$$
This means that the magnitude spectrum in the upper half-plane is a mirror reflection of the lower half-plane across the origin. Evaluating wedges over $[0, \pi)$ eliminates redundant duplicate information.

### Q10: How does the optimal cutoff frequency selection algorithm work?
**Answer:** The algorithm searches across candidate cutoff radii $D_0 \in [5, 10, 15, 20, 25, 30, 40, 50]$ for Ideal, Gaussian, and Butterworth filters:
1. It filters candidates that satisfy the spectral energy conservation constraint: $\text{Energy Retained} \ge 95\%$.
2. Among qualifying candidates, it selects the configuration that maximizes reconstruction PSNR (dB) while penalizing the spatial ringing index ($\text{Score} = 1.5 \cdot \text{PSNR} + \text{Energy Retained} - 20 \cdot \text{Ringing Metric}$).
3. This selects a smooth Gaussian or Butterworth filter at the minimal $D_0$ radius that preserves essential edge information without introducing artifacts.
