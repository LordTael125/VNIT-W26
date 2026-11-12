# Pattern Recognition Lab: Assignment 08
## Title: Web-Based Machine Learning Application Using Flask for Linear Classification via the Ho-Kashyap Procedure (with Integrated KNN & Bayes Classifiers)

**Course:** Pattern Recognition Laboratory (VNIT-W26)  
**Student Roll No:** MT26MCS024  
**Application Name:** **Classifiers** (Unified Pattern Recognition Studio)  
**Lab Session Date:** 3rd November 2026  

---

## 1. Problem Statement & Objectives

### Problem Statement
Develop a web application using Flask that allows users to upload any standard classification CSV dataset, inspect and pre-process the data, and standardize numerical features. The application shall train a linear classifier using the **Ho-Kashyap procedure** (implemented from scratch using NumPy), study its convergence for different learning rates, and use it to test whether the classes are linearly separable. Finally, it shall evaluate classification performance and predict the class of a new user-provided input.

Furthermore, integrate this application with the **Classifiers** web application (Bayes Classifier and KNN) developed in previous lab assignments, allowing users to select the classifier (using a dropdown list: **Bayes, KNN, Ho-Kashyap**) to apply on the uploaded dataset.

### Learning Objectives
1. **Flask Web Framework:** Build an interactive, stateful web-based Machine Learning studio with modular endpoints, dynamic UI controls, and session isolation.
2. **Data Ingestion & Pre-Processing:** Load, inspect, and pre-process datasets using Pandas and NumPy.
3. **Scikit-Learn Pre-Training Preparation:** Apply feature standardization (`StandardScaler`), categorical encoding (or exclusion), and missing-value imputation (`SimpleImputer`). Conditionally display pre-processing options only when needed; otherwise display a clean message.
4. **Ho-Kashyap Implementation from Scratch:** Implement the Ho-Kashyap procedure from first principles using NumPy to find the weight vector $\mathbf{a}$ and positive margin vector $\mathbf{b}$ of a linear discriminant function.
5. **Comprehensive Metric Evaluation:** Evaluate classifiers using Accuracy, Precision, Recall, F1-score, Classification Error Rate, and Confusion Matrix heatmaps.
6. **Convergence & Linear Separability Analysis:** 
   - Record squared error $\|\mathbf{e}(k)\|^2$ and training misclassifications ($\mathbf{a}^T \mathbf{y}_i \le 0$) at every iteration.
   - Plot **Iteration $k$ vs. $\|\mathbf{e}(k)\|^2$** and **Iteration $k$ vs. Number of misclassified samples**.
   - Systematically explore learning rates $\eta \in [0.1, 0.9]$, plot **$\eta$ vs. Number of iterations**, and automatically choose the optimal $\eta^*$ that converges in the fewest iterations with the lowest error rate.
   - Formally diagnose and display whether the dataset is **Linearly Separable** or **Linearly Non-Separable**.
7. **Instance Prediction Interface:** Build a dynamic form interface validating user inputs (non-empty, numeric, and training domain range warnings), presenting predictions descriptively with class codes ($0$ or $1$) and display labels.
8. **Unified Integration:** Integrate Ho-Kashyap with K-Nearest Neighbors (KNN with the Elbow Method) and Gaussian Naive Bayes under a single dropdown selector with a dedicated comparative benchmark studio (`/performance-analysis`).

---

## 2. Theoretical Foundations

### 2.1 The Ho-Kashyap Linear Classification Procedure

The Ho-Kashyap procedure is a classical algorithm for finding a linear separating hyperplane between two classes. Given a binary classification problem with training samples $\{(\mathbf{x}_i, t_i)\}_{i=1}^N$ where $\mathbf{x}_i \in \mathbb{R}^d$ and $t_i \in \{C_1, C_2\}$:

#### 1. Augmentation and Normalization
Augment each sample vector with a bias coordinate:
$$\mathbf{x}_{\text{aug}, i} = [1, x_{i,1}, x_{i,2}, \dots, x_{i,d}]^T \in \mathbb{R}^{d+1}$$

Normalize the sample vectors according to class membership ($z_i = +1$ for $C_1$, $z_i = -1$ for $C_2$):
$$\mathbf{y}_i = z_i \mathbf{x}_{\text{aug}, i}$$

This yields the normalized augmented design matrix $Y \in \mathbb{R}^{N \times (d+1)}$:
$$Y = \begin{bmatrix} \mathbf{y}_1^T \\ \mathbf{y}_2^T \\ \vdots \\ \mathbf{y}_N^T \end{bmatrix}$$

We seek a weight vector $\mathbf{a} = [a_0, a_1, \dots, a_d]^T$ and a margin vector $\mathbf{b} = [b_1, \dots, b_N]^T$ satisfying:
$$Y \mathbf{a} = \mathbf{b} > \mathbf{0} \quad \iff \quad \mathbf{a}^T \mathbf{y}_i = b_i > 0 \quad \forall i \in \{1, \dots, N\}$$

#### 2. Criterion Function & Gradient Optimization
Ho-Kashyap defines a squared-error criterion function with respect to *both* $\mathbf{a}$ and $\mathbf{b}$:
$$J_H(\mathbf{a}, \mathbf{b}) = \frac{1}{2} \|Y\mathbf{a} - \mathbf{b}\|^2 = \frac{1}{2} \sum_{i=1}^N (\mathbf{a}^T \mathbf{y}_i - b_i)^2 \quad \text{subject to } \mathbf{b} > \mathbf{0}$$

Minimizing $J_H$ with respect to $\mathbf{a}$ while holding $\mathbf{b}$ fixed yields:
$$\nabla_{\mathbf{a}} J_H = Y^T (Y \mathbf{a} - \mathbf{b}) = \mathbf{0} \implies \mathbf{a} = (Y^T Y)^{-1} Y^T \mathbf{b} = Y^\dagger \mathbf{b}$$
Where $Y^\dagger$ is the **Moore-Penrose pseudo-inverse** of $Y$.

Minimizing $J_H$ with respect to $\mathbf{b}$ via gradient descent subject to the constraint $\mathbf{b} > \mathbf{0}$:
$$\nabla_{\mathbf{b}} J_H = -(Y \mathbf{a} - \mathbf{b}) = -\mathbf{e}$$
Where the error vector at iteration $k$ is:
$$\mathbf{e}(k) = Y \mathbf{a}(k) - \mathbf{b}(k)$$

To prevent components of $\mathbf{b}$ from decreasing below zero, only the positive components of $\mathbf{e}(k)$ are used to update $\mathbf{b}$:
$$\mathbf{e}^+(k) = \frac{1}{2} (\mathbf{e}(k) + |\mathbf{e}(k)|) = \max(\mathbf{0}, \mathbf{e}(k))$$

The margin update rule with learning rate $\eta \in (0, 1]$ is:
$$\mathbf{b}(k+1) = \mathbf{b}(k) + 2 \eta \mathbf{e}^+(k)$$

And the weight vector update is:
$$\mathbf{a}(k+1) = Y^\dagger \mathbf{b}(k+1) = \mathbf{a}(k) + 2 \eta Y^\dagger \mathbf{e}^+(k)$$

---

### 2.2 Linear Separability & Convergence Proof (Duda & Hart Theorem)

A critical advantage of the Ho-Kashyap procedure over the standard Perceptron algorithm or relaxation methods is its ability to test whether a dataset is linearly separable:

1. **If the training samples are Linearly Separable:**
   The algorithm is mathematically guaranteed to terminate in a finite number of iterations with:
   $$\mathbf{e}(k) = \mathbf{0} \quad (\|\mathbf{e}(k)\|^2 \le \epsilon)$$
   At this point, $Y \mathbf{a} = \mathbf{b} > \mathbf{0}$, meaning all training samples satisfy $\mathbf{a}^T \mathbf{y}_i > 0$ (zero training misclassifications), yielding a valid separating hyperplane.

2. **If the training samples are Linearly Non-Separable:**
   At some iteration $k$, the error vector satisfies:
   $$\mathbf{e}(k) \le \mathbf{0} \quad \text{with } \mathbf{e}(k) \ne \mathbf{0}$$
   When this happens, all components $e_i(k) \le 0$, which implies:
   $$\mathbf{e}^+(k) = \mathbf{0} \implies \mathbf{b}(k+1) = \mathbf{b}(k)$$
   The margin vector ceases to update, providing a **conclusive mathematical proof** that the classes are linearly non-separable.

---

### 2.3 Learning Rate ($\eta$) Dynamics

The learning rate $\eta \in [0.1, 0.9]$ dictates the step size taken along the modified error gradient:
- **Low $\eta$ (e.g., $0.1 - 0.2$):** Conservative updates; monotonic error reduction; requires more iterations to converge.
- **High $\eta$ (e.g., $0.7 - 0.9$):** Rapid progression towards the separating margin; converges in fewer iterations without loss in classification accuracy.
- **Selection Criterion:**
  $$\eta^* = \arg\min_{\eta \in \mathcal{H}} \left\{ \text{Error Rate}(\eta), \text{Iterations}(\eta) \right\}$$
  Where $\text{Error Rate} = \frac{\text{Incorrect Predictions}}{\text{Total Predictions}} = 1 - \text{Accuracy}$.

---

### 2.4 Feature Standardization (`StandardScaler`)

Unscaled features with disparate numerical variances (e.g., Age $\in [20, 80]$ vs. Cholesterol $\in [150, 550]$) distort the design matrix $Y$ and inflate condition numbers of $Y^T Y$, resulting in numerical instability in the pseudo-inverse $Y^\dagger$.

Scikit-Learn `StandardScaler` centers and standardizes features:
$$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$
Where $\mu_j$ is the training sample mean and $\sigma_j$ is the sample standard deviation.

---

### 2.5 Multi-Classifier Paradigms: Ho-Kashyap vs. KNN vs. Bayes

| Property | Ho-Kashyap Procedure | K-Nearest Neighbors (KNN) | Gaussian Naive Bayes |
|---|---|---|---|
| **Paradigm** | Linear Discriminant (Direct Margin Optimization) | Non-Parametric (Instance-Based) | Generative Parametric (Probabilistic) |
| **Decision Boundary** | Planar Hyperplane: $\mathbf{a}^T \mathbf{x}_{\text{aug}} = 0$ | Piecewise Linear / Arbitrary Non-Linear | Quadratic / Linear (Gaussian Surfs) |
| **Separability Test** | **Yes** (Formal $\mathbf{e}(k) \le \mathbf{0}$ test) | No | No |
| **Inference Time** | $\mathcal{O}(d)$ (Dot product) | $\mathcal{O}(N \cdot d)$ (Pairwise distances) | $\mathcal{O}(d)$ (Feature log-likelihoods) |
| **Tuning Required** | Learning rate $\eta$ | Number of neighbors $K$ (Elbow Method) | None (Closed-form MLE) |

---

## 3. Directory Structure

```
PatternReco-Assignment-08/
├── Pattern Recognition Lab Assignment 08.pdf  # Assignment specification
├── README.md                                    # Comprehensive report and theoretical documentation
├── main.py                                      # Flask application with Ho-Kashyap, KNN, and Bayes
├── test_verification.py                         # Automated test suite (100% pass)
├── Dataset/                                     # Classification datasets
│   ├── linearly_separable_data.csv              # Synthetic linearly separable benchmark (200 samples)
│   ├── heart.csv                                # Heart disease classification (non-separable)
│   ├── Titanic-Dataset.csv                      # Titanic survival (mixed types, missing values)
│   ├── breast_cancer.csv                        # Breast Cancer Wisconsin (30 features)
│   ├── Naive-Bayes-Classification-Data.csv      # Blood Pressure & Glucose dataset
│   └── netflix_customer_churn.csv               # Customer churn classification
├── Statics/
│   ├── css/
│   │   └── style.css                            # Modern dark glassmorphic responsive design system
│   ├── js/
│   │   └── app.js                               # Interactive UI scripts (live slider, math fallback)
│   ├── plots/                                   # Generated plots
│   │   ├── hk_convergence.png                   # Dual plot: Iteration vs ||e||^2 & Misclassifications
│   │   ├── hk_eta_iterations.png                # Eta vs Number of iterations plot
│   │   ├── hk_confusion_matrix.png              # Test set confusion matrix heatmap
│   │   ├── knn_elbow_curve.png                  # K vs Error Rate Elbow curve
│   │   └── target_distribution.png              # Target class distribution bar chart
│   └── vendor/
│       └── katex/                               # Local KaTeX offline formula rendering assets
├── Templates/
│   ├── index.html                               # Main 4-step studio interface
│   └── performance_analysis.html                # Multi-dataset & multi-model benchmark matrix
└── Uploads/                                     # Upload cache directory
```

---

## 4. Key Functional Features Implemented

### 1. Dataset Ingestion & Inspection (Step 1)
- Supports user-uploaded CSV files and pre-loaded benchmark datasets (`linearly_separable_data.csv`, `heart.csv`, `Titanic-Dataset.csv`, `breast_cancer.csv`, etc.).
- Inspects row counts, column counts, null counts, duplicate counts, and column metadata types.
- Previews the first 5 rows and provides session state flushing upon new dataset load.

### 2. Pre-Training Preparation & Standardization (Step 2)
- **Conditional Alert:** If missing values, duplicates, or categorical features exist, displays the pre-processing configuration panel.
- **Clean Message:** If the dataset is clean (0 missing, 0 duplicates, all numeric), displays:
  > *"Pre-Processing Not Required: Dataset is clean and ready: 0 missing values, 0 duplicate rows, and all feature vectors are strictly numeric."*
- Imputation via Scikit-Learn `SimpleImputer` (median, mean, or mode).
- Categorical feature handling: option to encode to numeric vectors (One-Hot / Label encoding) or exclude non-numeric features.
- Standardization toggle: choice between Standardized (`StandardScaler`) and Raw unscaled features.
- Custom target class display names: map numeric codes (e.g. `0` and `1`) to human-readable labels (e.g. `Healthy` and `Heart Disease`).

### 3. Linear Classification via Ho-Kashyap (Step 3)
- Scratch implementation using NumPy matrix operations and pseudo-inverse `np.linalg.pinv(Y)`.
- Default training execution recording:
  - Final weight vector $\mathbf{a} = [a_0, a_1, \dots, a_d]^T$ with bias $a_0$ and feature weights.
  - Margin vector $\mathbf{b}$ summary: minimum $\min(\mathbf{b})$, maximum $\max(\mathbf{b})$, mean $\bar{\mathbf{b}}$, and positivity verification.
  - Test set metrics: Accuracy, Error Rate, Precision, Recall, F1-Score, and Confusion Matrix heatmap.
- **Convergence Tracking:** Records squared error $\|\mathbf{e}(k)\|^2$ and misclassified samples ($a^T y_i \le 0$) at every iteration.
- **Plots Generated:**
  1. *Iteration $k$ vs. $\|\mathbf{e}(k)\|^2$*
  2. *Iteration $k$ vs. Number of misclassified training samples*
  3. *Learning Rate $\eta$ vs. Number of iterations*
  4. *Test Confusion Matrix heatmap*
- **Multi-$\eta$ Convergence Study:** Tests $\eta \in [0.1, 0.9]$, plots $\eta$ vs. iterations, and identifies the optimal $\eta^*$ with a 1-click adoption button.
- **Linear Separability Diagnosis:** Formally determines if the dataset is **Linearly Separable** or **Linearly Non-Separable**.

### 4. Integration with Classifiers Studio (KNN & Bayes)
- Unified dropdown selector: **Ho-Kashyap Procedure, K-Nearest Neighbors (KNN), Bayes Classifier**.
- Dynamic UI controls display parameters relevant to each chosen classifier.
- KNN: Baseline $K=1$, Elbow Method curve across $K \in [1, 40]$, optimal elbow selection, and side-by-side scorecard.
- Bayes: Priors, Gaussian class-conditional density parameters, and posterior distributions.

### 5. Prediction of New User Instances (Step 4)
- Form dynamically generated from selected feature columns.
- Server-side and client-side input validations:
  - Non-empty check.
  - Numeric parseability.
  - Training range bounds check with yellow warning callouts for out-of-domain extrapolation.
- Output presentation:
  - Class variable ($0$ or $1$) and display label.
  - Linear discriminant score $g(\mathbf{x}) = \mathbf{a}^T \mathbf{x}_{\text{aug}}$.
  - Descriptive interpretation: explaining whether $g(\mathbf{x}) > 0$ (positive half-space) or $g(\mathbf{x}) \le 0$.
  - Sigmoid-calibrated margin confidence percentage.
  - Linear discriminant decomposition bar chart showing each feature's contribution $a_j \cdot x_j$.

### 6. Dedicated Comparative Analysis Matrix (`/performance-analysis`)
- Simultaneously evaluates 5 model architectures across 5 benchmark datasets:
  1. Ho-Kashyap (Raw Unscaled)
  2. Ho-Kashyap (Standardized)
  3. KNN ($K=1$ Scaled)
  4. KNN (Optimal Elbow $K$ Scaled)
  5. Gaussian Naive Bayes

---

## 5. Experimental Results & Observations

| Benchmark Dataset | Dimension | Ho-Kashyap Separability | Ho-Kashyap (Raw) | Ho-Kashyap (Scaled) | KNN (K=1 Scaled) | KNN (Optimal K) | Bayes Classifier |
|---|---|---|---|---|---|---|---|
| **Linearly Separable Data** | 200 &times; 2 | **Linearly Separable** | 100.0% (0.0% Err) | **100.0% (0.0% Err)** | 100.0% | 100.0% (K=1) | 100.0% |
| **Heart Disease** | 1025 &times; 13 | Non-Separable | 56.59% (43.41% Err) | **82.93% (17.07% Err)** | 73.68% | 85.53% (K=16) | 82.89% |
| **Breast Cancer (WDBC)** | 569 &times; 30 | Non-Separable (Near) | 91.23% (8.77% Err) | **95.61% (4.39% Err)** | 93.81% | 96.46% (K=9) | 92.92% |
| **Titanic Survival** | 891 &times; 6 | Non-Separable | 68.54% (31.46% Err) | **76.97% (23.03% Err)** | 70.79% | 78.65% (K=5) | 77.53% |
| **Blood Pressure / Glucose** | 995 &times; 2 | Non-Separable | 73.87% (26.13% Err) | **75.38% (24.62% Err)** | 74.00% | 78.00% (K=7) | 76.00% |

### Key Experimental Insights:
1. **Linear Separability Verification:**
   On `linearly_separable_data.csv`, the Ho-Kashyap procedure achieved 0 misclassifications and 100.0% test accuracy. The error vector norm decreased monotonically to zero, proving the theoretical convergence theorem.
2. **Impact of Learning Rate ($\eta$):**
   Across all benchmark datasets, as $\eta$ increased from $0.1$ to $0.9$, the number of iterations required to reach termination decreased by over $90\%$ (e.g., from 136 iterations at $\eta=0.1$ down to 8 iterations at $\eta=0.9$ on Heart Disease) while maintaining the identical optimal error rate ($17.07\%$).
3. **Crucial Role of Standardization:**
   Standardizing features via `StandardScaler` boosted Ho-Kashyap accuracy dramatically on `heart.csv` (+26.34% improvement from 56.59% to 82.93%), demonstrating that large continuous scales severely skew pseudo-inverse calculations when unscaled.
4. **Ho-Kashyap vs. KNN & Bayes:**
   Ho-Kashyap delivers near-optimal linear accuracy with $\mathcal{O}(d)$ test inference and zero distributional assumptions, while KNN achieves marginally higher accuracy on non-linear boundaries at the expense of $\mathcal{O}(Nd)$ inference latency.

---

## 6. How to Run the Application

### Prerequisites
Ensure Python 3.9+ and required dependencies are installed:
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```

### Running the Test Verification Suite
Run the automated verification suite to validate all mathematical and web components:
```bash
python3 test_verification.py
```
Expected output:
```
=====================================================================
  STARTING ASSIGNMENT 08 AUTOMATED TEST & VERIFICATION SUITE
=====================================================================
--- Test 1: Ho-Kashyap Procedure on Linearly Separable Dataset ---
  ✓ Test 1 Passed!
--- Test 2: Ho-Kashyap Multi-Eta Convergence on Heart Disease (Non-Separable) ---
  ✓ Test 2 Passed!
--- Test 3: Full AppState Workflow with Ingestion, Preprocessing, Multi-Classifier Training & Prediction ---
  ✓ Test 3 Passed!
--- Test 4: Flask Web Application Routes & Endpoints ---
  ✓ Test 4 Passed!
=====================================================================
  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! (100% PASS)
=====================================================================
```

### Starting the Flask Server
Navigate to the assignment directory and execute `main.py`:
```bash
cd "PatternReco-Assignment-08"
python3 main.py
```

### Accessing the Web Application
Open your browser and navigate to:
```
http://localhost:5000
```
- **Main Studio:** `http://localhost:5000/`
- **Comparative Analysis Studio:** `http://localhost:5000/performance-analysis`

---

## 7. Viva Voce Questions & Answers

### Q1: What is the primary difference between the Perceptron learning algorithm and the Ho-Kashyap procedure?
**Answer:** The Perceptron algorithm minimizes the Perceptron criterion function using fixed margins and gradient descent. If the data is linearly separable, it is guaranteed to converge; however, if the data is **linearly non-separable**, the Perceptron algorithm oscillates indefinitely without ever terminating or detecting non-separability. In contrast, the Ho-Kashyap procedure optimizes both the weight vector $\mathbf{a}$ and the margin vector $\mathbf{b} > \mathbf{0}$ simultaneously. It is guaranteed to terminate in finite steps for both cases: converging to $\mathbf{e}(k) = \mathbf{0}$ if separable, or detecting $\mathbf{e}(k) \le \mathbf{0}$ if non-separable.

### Q2: Why is the modified error vector $\mathbf{e}^+(k) = \frac{1}{2}(\mathbf{e}(k) + |\mathbf{e}(k)|)$ used instead of raw $\mathbf{e}(k)$ to update $\mathbf{b}$?
**Answer:** The constraint on the margin vector is $\mathbf{b} > \mathbf{0}$. If we updated $\mathbf{b}$ directly using $-\nabla_{\mathbf{b}} J_H = \mathbf{e}(k)$, components of $\mathbf{b}$ where $e_i(k) < 0$ (meaning $\mathbf{a}^T \mathbf{y}_i > b_i$) would decrease, potentially driving $b_i$ towards zero or negative values (violating the positivity constraint). By setting $e_i^+(k) = \max(0, e_i(k))$, the margin $b_i$ is only increased when the sample does not have sufficient margin ($e_i(k) > 0$), ensuring that $\mathbf{b}$ remains strictly positive.

### Q3: How does the Ho-Kashyap procedure mathematically prove that a dataset is linearly non-separable?
**Answer:** In the Ho-Kashyap theorem, if at any iteration $k$ all components of the error vector are non-positive ($\mathbf{e}(k) \le \mathbf{0}$ with $\mathbf{e}(k) \ne \mathbf{0}$), it implies that no positive margin vector $\mathbf{b} > \mathbf{0}$ exists such that $Y\mathbf{a} = \mathbf{b}$. Since $\mathbf{e}^+(k) = \mathbf{0}$, $\mathbf{b}$ stops updating, and this condition serves as an exact mathematical proof of linear non-separability.

### Q4: What is the Moore-Penrose pseudo-inverse and why is it used?
**Answer:** The pseudo-inverse $Y^\dagger = (Y^T Y)^{-1} Y^T$ computes the unique least-squares solution to the linear system $Y\mathbf{a} = \mathbf{b}$. Since the design matrix $Y$ is non-square ($N \times (d+1)$ with $N > d+1$), an exact matrix inverse does not exist. The pseudo-inverse minimizes $\|Y\mathbf{a} - \mathbf{b}\|^2$, providing the optimal weight vector $\mathbf{a}$ for any given margin vector $\mathbf{b}$.

### Q5: How does the learning rate $\eta$ affect Ho-Kashyap convergence?
**Answer:** The learning rate $\eta \in [0.1, 0.9]$ controls the step size of the margin update $\Delta \mathbf{b} = 2\eta \mathbf{e}^+$. A small $\eta$ (0.1) results in small margin increments, requiring more iterations to achieve convergence. A larger $\eta$ (0.7 to 0.9) accelerates convergence, reaching the optimal boundary in significantly fewer iterations. Our UI evaluates $\eta \in [0.1, 0.9]$ and selects the optimal $\eta^*$ that converges in the fewest iterations with the lowest test error rate.

### Q6: Why must categorical features be encoded or excluded before running Ho-Kashyap?
**Answer:** The Ho-Kashyap procedure relies on matrix multiplication and Euclidean inner products ($Y\mathbf{a} = \mathbf{b}$). String or categorical labels have no defined geometric or algebraic meaning in $\mathbb{R}^d$. Therefore, they must either be encoded into numeric indicator vectors (via One-Hot or Ordinal encoding) or excluded so that the augmented matrix $Y$ consists solely of real numbers.

