# Module 4: Frequency-Domain Filtering & Ringing Artifacts
## Transfer Functions, the Gibbs Phenomenon, Dual Plots, and Optimal Cutoff $D_0^*$

---

## 1. What Is Frequency-Domain Filtering?

In the spatial domain, filtering an image (like blurring or edge detection) requires **2D spatial convolution** with a sliding kernel $h(x, y)$:
$$g(x, y) = f(x, y) * h(x, y)$$
Spatial convolution is computationally expensive because every output pixel involves sliding a window and taking dot products.

By the **Fourier Convolution Theorem**:
$$\mathcal{F}\{f(x, y) * h(x, y)\} = F(u, v) \cdot H(u, v)$$
Spatial convolution becomes **simple point-by-point matrix multiplication** in the frequency domain!

```text
SPATIAL DOMAIN                       FREQUENCY DOMAIN
f(x, y)                             F(u, v)
   *    (Expensive Convolution)        ·    (Instant Element-wise Multiplication)
h(x, y)                             H(u, v)
   =                                   =
g(x, y)  <====== 2D IDFT ======     G(u, v) = H(u, v) · F(u, v)
```

---

## 2. Low-Pass vs. High-Pass: What Do They Do?

Let $D(u, v) = \sqrt{(u - M/2)^2 + (v - N/2)^2}$ be the distance of frequency $(u, v)$ from the centered DC origin $(M/2, N/2)$.

1. **Low-Pass Filter (LPF):**
   - **Passes:** Frequencies close to center ($D(u, v) \le D_0$).
   - **Blocks:** Frequencies far from center ($D(u, v) > D_0$).
   - **Visual Effect:** Smooths and blurs the image; removes high-frequency noise, fine textures, and sharp edges.
2. **High-Pass Filter (HPF):**
   - **Passes:** Frequencies far from center ($D(u, v) > D_0$).
   - **Blocks:** Frequencies close to center ($D(u, v) \le D_0$), including the DC component.
   - **Visual Effect:** Removes constant background shading; highlights only sharp object boundaries, lines, and edges on a dark background.

---

## 3. The Three Classic Filter Families Compared

```text
FILTER TRANSFER FUNCTION SHAPES (1D Slice through Frequency Plane)

Ideal LPF (Abrupt Step)      Gaussian LPF (Smooth Bell)    Butterworth LPF (Flat & Rolloff)
        H(D)                          H(D)                          H(D)
  1.0 +-----+                   1.0 +---\                     1.0 +-----\
      |     |                       |    \                        |      \
      |     |                       |     \                       |       \
  0.0 +-----+-----> D           0.0 +------\----> D           0.0 +--------\---> D
         D0                            D0                            D0
  Abrupt drop:                   Zero side-lobes:              Controlled rolloff:
  Causes severe ringing!         Zero ringing!                 Negligible ringing (n=2).
```

### 1. Ideal Low-Pass Filter (ILPF):
$$H_{\text{ideal}}(u, v) = \begin{cases} 1 & \text{if } D(u, v) \le D_0 \\ 0 & \text{if } D(u, v) > D_0 \end{cases}$$
- **Characteristics:** Perfectly preserves all frequencies within radius $D_0$ and completely eradicates all frequencies outside $D_0$.
- **Major Defect:** Produces severe **spatial ringing artifacts (Gibbs phenomenon)**.

### 2. Gaussian Low-Pass Filter (GLPF):
$$H_{\text{gaussian}}(u, v) = \exp\left(-\frac{D(u, v)^2}{2 D_0^2}\right)$$
- **Characteristics:** The transfer function transitions smoothly from $1$ at the center to $0$ at high frequencies without any sudden step or discontinuity.
- **Major Advantage:** **100% ringing-free!** The inverse Fourier transform of a Gaussian is another Gaussian, which has no negative side-lobes.

### 3. Butterworth Low-Pass Filter (BLPF) (Order $n=2$):
$$H_{\text{butterworth}}(u, v) = \frac{1}{1 + \left[\frac{D(u, v)}{D_0}\right]^{2n}}$$
- **Characteristics:** At $D(u, v) = D_0$, the transfer function is exactly $H = \frac{1}{1 + 1} = 0.5$ (the $-3\text{ dB}$ cutoff point).
- **Major Advantage:** Strikes a balance between the sharpness of the Ideal filter and the smoothness of the Gaussian filter. At $n=2$, it preserves high contrast without noticeable ringing.

---

## 4. What Is the Gibbs Phenomenon & Ringing Artifacts?

When you apply an Ideal Low-Pass Filter with a small cutoff like $D_0 = 10$ or $15$ to an image with sharp edges (like a checkerboard or text), you will notice **faint ripple waves / ghost echoes** radiating outward from every sharp edge:

```text
Cross-section of a sharp spatial edge after Ideal Low-Pass Filtering:

Intensity
  ^
  |          Ringing Overshoot (Gibbs)
  |            _.-._
  |          .'     '.         Ripples Decay Gradually
  |         /         \     .-.
  |        /           '---'   '---.________________ (Target Level 1)
  |       /
  |      /
  |  ---' <--- Undershoot
  +----------------------------------------------------> Space (x)
```

### Why Does This Happen? (The Mathematical Cause)
In the frequency domain, the Ideal filter is a rectangular cylinder of radius $D_0$.
According to the Fourier Transform pairs:
$$\mathcal{F}^{-1}\{\text{Cylinder of radius } D_0\} = \frac{J_1(2\pi D_0 r)}{r}$$
where $J_1(r)$ is the Bessel function of the first kind (the 2D circular equivalent of the 1D $\frac{\sin(x)}{x}$ sinc function).

A sinc function does not drop smoothly to zero; **it oscillates with positive and negative side-lobes that decay very slowly**:

```text
Spatial Impulse Response h(r) of Ideal Filter:
         ^
        / \
       /   \
  ----+-----+----+-----+-----+----> r
       \   /      \   /
        '-'        '-'
     Negative side-lobes convolve with edges to produce ripples!
```

When you convolve this oscillating sinc kernel with an edge in the spatial domain, every point on the edge radiates decaying ripples. This ripple effect is called **Gibbs Ringing**.

---

## 5. Understanding the Dual Comparative Curves

In Step 4 of the application, we sweep across cutoff radii:
$$D_0 \in [5, 10, 15, 20, 25, 30, 40, 50]$$
and plot two side-by-side comparative graphs:

### Graph 1: Cutoff Radius $D_0$ vs. Spectral Energy Retained (%)
```text
Energy Retained (%)
 100% +-----------------------------......------ (100% Total Energy)
      |                       . - '   (Gaussian)
  95% | - - - - - - - - - - * - - - - - - - - -  (95% Retention Threshold)
      |                . - '
  80% |          . - '       (Ideal Filter climbs faster because it passes 100% inside D0)
      |     . - '
  50% | . - '
      +-----+-----+-----+-----+-----+-----+-----> Cutoff D0
            5    10    15    20    25    30    40
```
- **What it shows:** As the radius $D_0$ increases, the filter's aperture widens, capturing more and more of the high-frequency Fourier harmonics.
- **Key Observation:** Because real images concentrate almost all their energy in low frequencies, even a small radius like $D_0 = 15$ to $25$ captures over **$95\%$** of the entire image's energy!

### Graph 2: Cutoff Radius $D_0$ vs. Reconstruction PSNR (dB)
```text
PSNR (dB)
  60 dB +                                     * (D0 = 40)
        |                              *
  40 dB +                       *
        |                * (D0 = 25)
  25 dB +         * (D0 = 15)
        |  * (D0 = 10)
  10 dB +
        +-----+-----+-----+-----+-----+-----+-----> Cutoff D0
              5    10    15    20    25    30
```
- **What PSNR (Peak Signal-to-Noise Ratio) means:**
  - $\text{PSNR} < 15\text{ dB}$ indicates poor reconstruction with severe blur.
  - $\text{PSNR} \in [20, 30]\text{ dB}$ indicates good reconstruction with acceptable fidelity.
  - $\text{PSNR} > 35\text{ dB}$ indicates near-perfect reconstruction indistinguishable from the original to the human eye.
- **Why PSNR increases with $D_0$:** As $D_0$ increases, fewer high frequencies are discarded, so the Mean Squared Error (MSE) between the original image and reconstructed image drops toward zero.

---

## 6. How the Optimal Cutoff Radius ($D_0^*$) Is Selected

The assignment asks:
> *"Choose the filter type and optimal cutoff radius $D_0$ that preserves at least $95\%$ of total spectral energy while maximizing PSNR and minimizing ringing artifacts."*

Our state manager in `main.py` implements this via an objective optimization algorithm:

1. **Step 1: Constraint Filtering ($\ge 95\%$ Energy):**
   Filter all evaluated combinations (Ideal, Gaussian, Butterworth across all $D_0$) and keep only those that satisfy:
   $$\text{Energy Retained} \ge 95.0\%$$

2. **Step 2: Multi-Objective Scoring:**
   For each qualifying candidate, compute an optimization score:
   $$\text{Score} = 1.5 \cdot \text{PSNR} + 1.0 \cdot \text{Energy Retained} - 20.0 \cdot \text{Ringing Metric}$$
   - Rewards high PSNR (reconstruction fidelity).
   - Rewards high spectral energy retention.
   - Heavily penalizes the spatial ringing metric $\text{TV}(|f - \hat{f}|) / \text{TV}(f)$.

3. **Step 3: Winner Selection:**
   - In almost all image patterns, the **Gaussian or Butterworth filter at $D_0^* \approx 25 \text{ to } 30$** emerges as the winner.
   - It captures $> 95\%$ of the spectral energy, achieves high PSNR ($> 26\text{ dB}$), and **eliminates the harsh Gibbs ringing** produced by the default Ideal filter at $D_0 = 15$.
