# Module 5: Invariant Features & Pattern Recognition
## Radial Rings, Angular Wedges, Invariance Proofs, and Nearest-Neighbor Matching

---

## 1. Why Traditional Spatial Matching Fails

In pattern recognition, the simplest approach to classify an image is template matching in the spatial domain (e.g. comparing pixel values $f(x, y)$ directly against a stored reference template $t(x, y)$).

This approach fails completely in the real world:
1. **If the object shifts by just 3 pixels (translation):** Pixels no longer line up; the Euclidean distance between pixel vectors shoots through the roof.
2. **If the object rotates by $15^\circ$ (rotation):** The spatial overlap becomes almost zero.
3. **If the lighting dims or brightens (illumination change):** Every pixel value changes by a scaling factor.

To build an intelligent pattern recognition system, we need **features that describe the fundamental nature of the pattern regardless of where it is, how it is turned, or how bright it is**.

The centered 2D Fourier magnitude spectrum $|F(u, v)|$ provides the mathematical key.

---

## 2. Radial Ring Descriptors ($S_r$): Capturing Scale & Coarseness

Imagine dividing the centered frequency plane into $R$ concentric annular donuts (rings) around the DC center $(u_0, v_0) = (M/2, N/2)$:

```text
RADIAL RING PARTITIONING (R Concentric Rings)
              +-------------------------+
              |           . - .         |
              |        .-'     '-.      |
              |      .'   .-.     '.    |
              |     /   .' (o) '.   \   |  <-- Ring k (between radius r_k and r_{k+1})
              |    |   /   DC    \   |  |
              |     \   '.     .'   /   |
              |      '.   '-.-'   .'    |
              |        '-.     .-'      |
              |           ' - '         |
              +-------------------------+
```

### Mathematical Definition:
Let $r_{\text{max}} = \frac{\min(M, N)}{2}$. We partition the radius interval $[0, r_{\text{max}}]$ into $R$ equal bins:
$$r_k = k \cdot \frac{r_{\text{max}}}{R}, \quad k = 0, 1, \dots, R-1$$

The energy contained in the $k$-th radial ring is the sum of squared spectral magnitudes of all frequencies falling inside that ring:
$$S_r(k) = \sum_{r_k \le D(u, v) < r_{k+1}} |F(u, v)|^2$$

### Physical Meaning:
- **Inner rings (small radius $r$):** Measure low-frequency content (broad textures, smooth variations).
- **Outer rings (large radius $r$):** Measure high-frequency content (fine textures, sharp transitions).

### Mathematical Proof of Rotation Invariance:
Suppose the spatial image $f(x, y)$ is rotated by an arbitrary angle $\phi$ to produce $f_{\text{rot}}(x, y)$.
In polar frequency coordinates $(r, \theta)$, the Fourier Rotation Theorem proves that:
$$F_{\text{rot}}(r, \theta) = F(r, \theta - \phi)$$
The entire 2D Fourier spectrum simply rotates by the exact same angle $\phi$ around the DC center!

Now look at the ring integral:
$$S_r(k) = \int_{r_k}^{r_{k+1}} \int_{0}^{2\pi} |F_{\text{rot}}(r, \theta)|^2 \, d\theta \, r \, dr = \int_{r_k}^{r_{k+1}} \int_{0}^{2\pi} |F(r, \theta - \phi)|^2 \, d\theta \, r \, dr$$
Because the angular integration covers the complete circle $[0, 2\pi)$, shifting the angle by $\phi$ merely changes the starting point of a circular integral around a closed loop:
$$\int_{0}^{2\pi} |F(r, \theta - \phi)|^2 \, d\theta = \int_{0}^{2\pi} |F(r, \theta')|^2 \, d\theta'$$

#### Conclusion:
The Radial Ring descriptor vector $\mathbf{S}_r = [S_r(0), S_r(1), \dots, S_r(R-1)]$ is **strictly invariant to spatial image rotation**!

---

## 3. Angular Wedge Descriptors ($S_\theta$): Capturing Orientation & Direction

Now, imagine dividing the centered frequency plane into $A$ directional pie slices (wedges) radiating outward from the DC center:

```text
ANGULAR WEDGE PARTITIONING (A Directional Sectors)
              +-------------------------+
              |           \ | /         |
              |      \     \|/     /    |
              |        \    |    /      |
              |   ------- (o) -------   |  <-- Sector m (between angle θ_m and θ_{m+1})
              |        /    |    \      |
              |      /     /|\     \    |
              |           / | \         |
              +-------------------------+
```

### Mathematical Definition:
Because real images exhibit Hermitian conjugate symmetry $|F(u, v)| = |F(-u, -v)|$, the magnitude spectrum in the lower half-plane is an exact mirror reflection of the upper half-plane.
Therefore, we only need to span the orientation range $[0, \pi)$:
$$\theta_m = m \cdot \frac{\pi}{A}, \quad m = 0, 1, \dots, A-1$$

The energy contained in the $m$-th angular wedge is:
$$S_\theta(m) = \sum_{\theta_m \le \theta(u, v) < \theta_{m+1}} |F(u, v)|^2$$
where $\theta(u, v) = \arctan2(v - N/2, u - M/2) \pmod \pi$.

### Physical Meaning:
- Captures the **principal directional edge orientations** of the pattern.
- Vertical stripes produce high energy in horizontal wedges.
- Horizontal stripes produce high energy in vertical wedges.
- Isotropic patterns (like concentric rings) distribute energy uniformly across all wedges.

### Mathematical Proof of Scale Invariance:
Suppose the spatial image is scaled by factor $a$: $f_{\text{scaled}}(x, y) = f(a x, a y)$.
By the Fourier Scaling Theorem:
$$\mathcal{F}\{f(a x, a y)\} = \frac{1}{|a|^2} F\left(\frac{u}{a}, \frac{v}{a}\right)$$
In polar coordinates:
$$F_{\text{scaled}}(r, \theta) = \frac{1}{|a|^2} F\left(\frac{r}{a}, \theta\right)$$
Notice that the radial distance $r$ is scaled to $r/a$, but **the angular direction $\theta$ is completely unchanged!**

When we integrate along an angular wedge from radius $0$ to $\infty$:
The scaling factor simply stretches or compresses the energy along the ray without shifting any energy from one angular wedge to another. When normalized by the total spectral energy, the angular wedge fractions $\mathbf{S}_\theta$ remain **scale-invariant**.

---

## 4. Normalization: Translation & Illumination Invariance

We concatenate the two descriptors into a single $(R + A)$-dimensional feature vector:
$$\mathbf{f}_{\text{raw}} = [S_r(0), \dots, S_r(R-1), \; S_\theta(0), \dots, S_\theta(A-1)]^T$$

To make the vector robust to real-world conditions, we divide it by the total spectral energy:
$$\mathbf{v} = \frac{\mathbf{f}_{\text{raw}}}{\sum_{u=0}^{M-1} \sum_{v=0}^{N-1} |F(u, v)|^2 + \epsilon}$$

### Proof of Translation Invariance:
If the image shifts spatially by $(x_0, y_0)$:
$$f_{\text{shift}}(x, y) = f(x - x_0, y - y_0)$$
Its Fourier transform is:
$$F_{\text{shift}}(u, v) = F(u, v) \exp\left[-j 2\pi \left(\frac{u x_0}{M} + \frac{v y_0}{N}\right)\right]$$
Taking the magnitude:
$$|F_{\text{shift}}(u, v)| = |F(u, v)| \cdot \left|\exp\left[-j 2\pi \left(\frac{u x_0}{M} + \frac{v y_0}{N}\right)\right]\right| = |F(u, v)| \cdot 1 = |F(u, v)|$$
The magnitude spectrum $|F(u, v)|$ **does not change by even a fraction of a percent under translation!**
Because both rings and wedges are computed purely from $|F(u, v)|$, the feature vector is **strictly translation-invariant**.

### Proof of Illumination Invariance:
If the image brightness is uniformly scaled by constant $c$:
$$f_{\text{illum}}(x, y) = c \cdot f(x, y) \implies F_{\text{illum}}(u, v) = c \cdot F(u, v)$$
The squared magnitude becomes:
$$|F_{\text{illum}}(u, v)|^2 = c^2 |F(u, v)|^2$$
Every raw ring $S_r$ and wedge $S_\theta$ is multiplied by $c^2$.
Similarly, the total energy in the denominator is multiplied by $c^2$:
$$\mathbf{v}_{\text{illum}} = \frac{c^2 \mathbf{f}_{\text{raw}}}{c^2 \sum |F(u, v)|^2} = \frac{\mathbf{f}_{\text{raw}}}{\sum |F(u, v)|^2} = \mathbf{v}$$
The scaling factor $c^2$ cancels out completely!

---

## 5. Nearest-Neighbor Matching & Classification

When a user uploads an unseen query image (e.g. `test_vertical_shifted.png` or `test_concentric_rotated.png`):

1. **Step 1: Input Validation:**
   - Checks file format (.png, .jpg, .bmp).
   - Validates aspect ratio: issues a warning if $W/H < 0.5$ or $> 2.0$.
   - Validates contrast (standard deviation of pixel values): issues a warning if $\sigma < 10.0$ (blank image) or $\sigma > 115.0$ (heavy saturation).

2. **Step 2: Pre-Processing:**
   - Converts to grayscale luminance and resizes to $64 \times 64$.
   - Normalizes intensity to $[0, 1]$.

3. **Step 3: Feature Extraction:**
   - Computes from-scratch centered 2D DFT.
   - Extracts the normalized $(16 + 16 = 32)$-dimensional feature vector $\mathbf{v}_{\text{query}}$.

4. **Step 4: Distance Metrics to Reference Library:**
   For every reference archetype $\mathbf{v}_{\text{ref}}$ stored in `Dataset/`:
   - **Euclidean Distance:**
     $$d = \|\mathbf{v}_{\text{query}} - \mathbf{v}_{\text{ref}}\|_2 = \sqrt{\sum_{i=1}^{32} (v_{\text{query}}[i] - v_{\text{ref}}[i])^2}$$
   - **Cosine Similarity:**
     $$S_c = \frac{\mathbf{v}_{\text{query}} \cdot \mathbf{v}_{\text{ref}}}{\|\mathbf{v}_{\text{query}}\| \|\mathbf{v}_{\text{ref}}\|}$$
   - **Match Confidence (%):**
     $$\text{Confidence} = \frac{1}{1 + d} \times 100\%$$

5. **Step 5: Output:**
   - Ranks all candidate classes from nearest to furthest.
   - Assigns the predicted class label to the highest-scoring candidate.

---

## 6. Curated Pattern Archetype Gallery (What to Expect in Dataset)

Here is a breakdown of the reference pattern classes bundled in the application:

| Class Name | Spatial Pattern Description | Frequency Spectrum Appearance | Feature Profile |
| :--- | :--- | :--- | :--- |
| **Vertical Stripes** | Periodic lines alternating along $x$-axis | Bright horizontal dots along $u$-axis ($v=0$) | High energy in horizontal wedges ($S_\theta \approx 0, \pi$) |
| **Horizontal Stripes** | Periodic lines alternating along $y$-axis | Bright vertical dots along $v$-axis ($u=0$) | High energy in vertical wedges ($S_\theta \approx \pi/2$) |
| **Diagonal Stripes** | Waves tilted at $45^\circ$ or $135^\circ$ | Diagonal line of harmonic peaks at $45^\circ$ | High energy in diagonal wedges |
| **Concentric Rings** | Circular target ripples centered in image | Concentric circular rings in spectrum | Energy concentrated in specific radial rings $S_r$; flat across all wedges |
| **Checkerboard** | 2D orthogonal square grid | Symmetrical grid of harmonic peaks | Concentrated in both horizontal, vertical, and cross wedges |
| **Radial Star** | Spokes radiating outward from center | Orthogonal spoke lines in frequency plane | Distributed across high-frequency radial rings |
| **Dot Lattice** | Regular 2D array of isolated dots | 2D lattice of frequency impulses | Symmetrical discrete spikes in both rings and wedges |
| **Natural Texture** | Smooth fractal / Gaussian noise | Continuous cloud decaying as $1/f^\alpha$ | Smooth exponential decay across radial rings |
