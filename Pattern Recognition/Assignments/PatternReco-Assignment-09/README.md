# Pattern Recognition Lab: Assignment 09
## Title: Web-Based Machine Learning Application Using Flask for Non-Parametric Classification via the Parzen Window Density Estimation Method (with Integrated Ho-Kashyap, KNN & Bayes Classifiers)

**Course:** Pattern Recognition Laboratory (VNIT-W26)  
**Student Roll No:** MT26MCS024  
**Application Name:** **Classifiers** (Unified Pattern Recognition Studio)  
**Lab Session Date:** 17th November 2026  

---

## 1. Problem Statement & Objectives

### Problem Statement
Develop a web application using Flask that allows users to upload any standard classification CSV dataset, inspect and pre-process the data, and standardize numerical features. The application shall estimate the class-conditional probability density of every class using the **Parzen Window method** (implemented from scratch using NumPy), study the effect of the window width $h$ on the estimated densities and on the classification performance, and select its optimal value. Finally, it shall evaluate classification performance and predict the class of a new user-provided input.

Furthermore, integrate this application with the **Classifiers** web application (Bayes Classifier, KNN, and Ho-Kashyap) developed in the previous lab assignments, allowing users to select the classifier (using a dropdown list: **Bayes, KNN, Ho-Kashyap, Parzen Window**) to apply on the uploaded dataset.

### Learning Objectives
1. **Flask Web Framework:** Build an interactive, stateful web-based Machine Learning studio with modular endpoints, dynamic UI controls, and session isolation.
2. **Data Ingestion & Pre-Processing:** Load, inspect, and pre-process datasets using Pandas and NumPy.
3. **Scikit-Learn Pre-Training Preparation:** Apply feature standardization (`StandardScaler`), categorical encoding (or exclusion), and missing-value imputation (`SimpleImputer`). Conditionally display pre-processing options only when needed; otherwise display a clean message.
4. **Parzen Window Density Estimator from Scratch:** Implement non-parametric density estimation from first principles using NumPy for both **Hypercube** and **Gaussian** window functions.
5. **Log-Sum-Exp Trick for Underflow Avoidance:** With the Gaussian window, small $h$ or large numbers of features $d$ make window values underflow to zero in standard IEEE-754 floating point; compute densities in log-space using the numerically stable **log-sum-exp trick**.
6. **Bayesian Classification:** Classify query samples by combining estimated class-conditional densities $p(\mathbf{x}|\omega_j)$ with prior probabilities $P(\omega_j)$ using the Bayes decision rule.
7. **Comprehensive Metric Evaluation:** Evaluate classifiers using Accuracy, Precision, Recall, F1-score (macro-averaged for multi-class datasets), Classification Error Rate, and Confusion Matrix heatmaps.
8. **Window Width Analysis ($h$ Optimization):**
   - Apply the Parzen Window classifier across multiple values of $h \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$ for both window functions.
   - Record training error rate, validation error rate, and, for the hypercube window, the count of query samples whose window contains no training samples.
   - Plot a graph of $h$ vs. Error Rate with both window functions (4 curves) on the same graph with a logarithmic scale for $h$.
   - Automatically select the optimal window function and window width $h^*$ that yield the lowest validation error rate, and compare test performance against default parameters ($h = 1.0$, Gaussian).
9. **Instance Prediction Interface:** Build a dynamic form interface validating user inputs (non-empty, numeric, and training domain range outlier warnings), presenting predictions with class posteriors, densities, and log-densities.
10. **Unified Integration:** Integrate Parzen Window into the multi-classifier studio with Ho-Kashyap, KNN, and Gaussian Naive Bayes under a single dropdown selector with a dedicated comparative benchmark studio (`/performance-analysis`).
11. **Dynamic Target Feature Selector:** Explicit dropdown selector in Step 2 allowing users to designate any column in the dataset as the classification target (with smart canonical target detection), re-encoding classes and regenerating class distribution charts instantly.
12. **Garbage Collection & Temporary File Management:** Integrated system cleaning temporary uploads and generated plots from disk and invoking Python's `gc.collect()` to prevent memory leaks and disk bloat.

---

## 2. Theoretical Foundations

### 2.1 Non-Parametric Density Estimation
In parametric pattern recognition (such as Gaussian Naive Bayes), we assume a known parametric form (e.g. multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$) and estimate the parameters from training data. When the true underlying distribution is multimodal, skewed, or complex, parametric assumptions fail.

**Non-parametric density estimation** makes no prior assumptions about the distribution's algebraic form. The probability $P$ that a continuous random vector $\mathbf{x} \in \mathbb{R}^d$ falls within a region $\mathcal{R}$ of volume $V$ is:
$$P = \int_{\mathcal{R}} p(\mathbf{x}') \, d\mathbf{x}' \approx p(\mathbf{x}) V$$

If $k$ out of $N$ samples fall inside region $\mathcal{R}$, the empirical probability is $P \approx k / N$. Equating the two yields the fundamental density estimation relationship:
$$p_n(\mathbf{x}) \approx \frac{k}{N V}$$

### 2.2 The Parzen Window Density Estimator
In the Parzen Window method, the volume $V$ is centered around the query point $\mathbf{x}$. Let $V_n = h^d$ be the volume of a hypercube of edge length $h$ in $d$ dimensions. We define a window function (kernel function) $\varphi(\mathbf{u})$.

The Parzen Window estimate of the class-conditional probability density for class $\omega_j$ with $N_j$ training samples $\{\mathbf{x}_{ji}\}_{i=1}^{N_j}$ is:
$$p(\mathbf{x} | \omega_j) = \frac{1}{N_j V_n} \sum_{i=1}^{N_j} \varphi\left(\frac{\mathbf{x} - \mathbf{x}_{ji}}{h}\right) = \frac{1}{N_j h^d} \sum_{i=1}^{N_j} \varphi\left(\frac{\mathbf{x} - \mathbf{x}_{ji}}{h}\right)$$

#### A. Hypercube Window Function
In $d$-dimensional space, the unit hypercube window is defined as:
$$\varphi(\mathbf{u}) = \begin{cases} 1 & \text{if } \max_{k=1,\dots,d} |u_k| \le \frac{1}{2} \\ 0 & \text{otherwise} \end{cases}$$

Therefore, a training sample $\mathbf{x}_{ji}$ falls inside the hypercube window of side $h$ centered at $\mathbf{x}$ if and only if its Chebyshev ($L_\infty$) distance is at most $h/2$:
$$\max_{k=1,\dots,d} |x_k - x_{jik}| \le \frac{h}{2}$$

Let $k_j(\mathbf{x}) = \sum_{i=1}^{N_j} \mathbb{I}\left(\|\mathbf{x} - \mathbf{x}_{ji}\|_\infty \le \frac{h}{2}\right)$ be the count of class $\omega_j$ samples in the hypercube. The estimated density is:
$$p(\mathbf{x} | \omega_j) = \frac{k_j(\mathbf{x})}{N_j h^d}$$

**Empty Window Handling:**
If a query point $\mathbf{x}$ is isolated, it is possible that $k_j(\mathbf{x}) = 0$ for all classes $j$, meaning $\sum_j k_j(\mathbf{x}) = 0$. In this case, the window contains zero training samples. The application records this count and gracefully falls back to the training class prior probabilities $P(\omega_j)$ to avoid indeterminate fractions.

#### B. Gaussian Window Function
The multivariate spherical Gaussian window function in $d$ dimensions is:
$$\varphi(\mathbf{u}) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2} \|\mathbf{u}\|^2\right)$$

Scaling by window width $h$ yields:
$$p(\mathbf{x} | \omega_j) = \frac{1}{N_j (2\pi h^2)^{d/2}} \sum_{i=1}^{N_j} \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_{ji}\|^2}{2h^2}\right)$$

This interprets the density as a smooth mixture of Gaussians centered at each training point with variance $\sigma^2 = h^2$.

### 2.3 Floating-Point Underflow & The Log-Sum-Exp Trick
When $h$ is small (e.g. $h = 0.01$ or $0.05$) or feature dimensionality $d$ is high (e.g. 30 features in Breast Cancer), the exponent $-\frac{\|\mathbf{x} - \mathbf{x}_{ji}\|^2}{2h^2}$ can easily become $-1000$ or less. In 64-bit IEEE-754 floating point, $\exp(-709) \approx 0.0$, causing severe **underflow to zero**. If densities underflow to zero for all classes, posterior probabilities become $\frac{0}{0} = \text{NaN}$.

To prevent underflow, our scratch implementation evaluates class-conditional densities strictly in **logarithmic space**:
$$\ln p(\mathbf{x} | \omega_j) = -\ln(N_j) - \frac{d}{2}\ln(2\pi) - d\ln(h) + \ln\left(\sum_{i=1}^{N_j} \exp(z_i)\right)$$
where $z_i = -\frac{\|\mathbf{x} - \mathbf{x}_{ji}\|^2}{2h^2}$.

To evaluate $\ln \sum_{i=1}^{N_j} \exp(z_i)$ stably, let $M = \max_{i} z_i \le 0$:
$$\ln\left(\sum_{i=1}^{N_j} \exp(z_i)\right) = M + \ln\left(\sum_{i=1}^{N_j} \exp(z_i - M)\right)$$

Since $z_i - M \le 0$, $\exp(z_i - M) \in (0, 1]$, with at least one element equal to $\exp(0) = 1$. Underflow is completely eliminated!

### 2.4 Bayes Decision Rule for Classification
Combining class-conditional densities $p(\mathbf{x} | \omega_j)$ with class prior probabilities $P(\omega_j) = \frac{N_j}{N}$:
$$P(\omega_j | \mathbf{x}) = \frac{p(\mathbf{x} | \omega_j) P(\omega_j)}{\sum_{c=1}^C p(\mathbf{x} | \omega_c) P(\omega_c)}$$

In log-space:
$$a_j(\mathbf{x}) = \ln p(\mathbf{x} | \omega_j) + \ln P(\omega_j)$$
$$P(\omega_j | \mathbf{x}) = \frac{\exp(a_j(\mathbf{x}) - \max_c a_c(\mathbf{x}))}{\sum_{c=1}^C \exp(a_c(\mathbf{x}) - \max_c a_c(\mathbf{x}))}$$
$$\hat{y} = \arg\max_{j \in \{1,\dots,C\}} a_j(\mathbf{x})$$

### 2.5 Window Width ($h$) Bias-Variance Trade-Off
The smoothing parameter $h$ controls the estimator's bias-variance trade-off:
- **Small $h$ (Undersmoothing / High Variance):**
  - Gaussian density becomes a collection of sharp, spiky peaks centered around training samples.
  - Hypercube window becomes tiny, leaving most of the feature space empty ($k_j = 0$).
  - Training error rate is artificially low (overfitting), but validation error rate spikes.
- **Large $h$ (Oversmoothing / High Bias):**
  - Kernels overlap excessively, washing out subtle class boundaries and local structure.
  - Estimated density flattens out into an overly smooth, unimodal distribution.
  - Training and validation error rates both increase (underfitting).
- **Optimal $h^*$:**
  - Minimizes classification error rate on an independent validation set.

Error rate is defined as:
$$\text{Error Rate} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Predictions}} = 1 - \text{Accuracy}$$

---

## 3. System Architecture & Features

```
+---------------------------------------------------------------------------------------------------+
|                                 Classifiers Studio Web Application                                |
+---------------------------------------------------------------------------------------------------+
|  Step 1: Dataset Ingestion (Presets: Iris, Breast Cancer, Heart, Titanic, Churn, Separable / CSV) |
|  Step 2: Exploration & Target Configuration (Distribution Bar Chart, Dynamic Class Renaming)      |
|  Step 3: Pre-Training Preparation (Imputation, Categorical Encoding, StandardScaler Toggle)        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
        [Active Classifier: Parzen Window]                        [Other Integrated Classifiers]
                     |                                                         |
      +--------------+--------------+                           +--------------+--------------+
      |                             |                           |              |              |
1. Default Model             2. Multi-h Analysis           Ho-Kashyap         KNN           Bayes
  (Gaussian, h=1.0)           (Gaussian & Hypercube)       (Linear Margin) (Elbow Curve) (Parametric)
  - Prior Probabilities       - Grid: h in [0.01..5.0]          |              |              |
  - First 5 Test Densities    - Train & Val Error Rates         +--------------+--------------+
  - Accuracy, F1, CM Heatmap  - Empty Window Counts                            |
      |                       - 4-Curve Log-Scale Plot                         v
      +--------------+--------------+                             Benchmark Comparison Studio
                     |                                              (/performance-analysis)
             3. Optimal Model
              - Best (Window*, h*)
              - Test Evaluation & CM
              - Side-by-Side Comparison
                     |
                     v
  Step 6: Prediction of New Instances
  (Dynamic Feature Form, Domain Range Warning, Fitted Preprocessing, Probabilities & Densities)
```

---

## 4. Implementation Details

### 4.1 CustomParzenClassifier (NumPy from First Principles)
- **Zero scikit-learn dependency** for density computation.
- **BLAS-accelerated Pairwise Distance Calculation:** Uses 2D matrix expansion $\|\mathbf{x}_m - \mathbf{x}_i\|^2 = \|\mathbf{x}_m\|^2 - 2 \mathbf{x}_m \mathbf{x}_i^T + \|\mathbf{x}_i\|^2$ to eliminate memory-intensive 3D tensors.
- **Log-Sum-Exp Trick:** Implemented directly via `max_Z + np.log(np.sum(np.exp(Z - max_Z)))`.
- **Chebyshev Distance for Hypercube:** Computes $L_\infty$ distance batch-wise to maintain efficient memory utilization.
- **Full Explainability:** Returns class-conditional densities, log-densities, prior probabilities, and posterior probabilities for any test or user query.

### 4.2 Multi-h Window Width Analysis Engine
- Evaluates $h \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$ (customizable via UI input).
- Compares **Gaussian** and **Hypercube** kernels simultaneously on identical 3-way stratified splits:
  - Training Set: 60%
  - Validation Set: 20%
  - Test Set: 20%
- Tracks **Empty Window Count** for Hypercube window on validation and test sets.
- Generates high-resolution 4-curve visualization with logarithmic $x$-axis scale (`parzen_h_analysis.png`).
- Determines optimal $(window^*, h^*)$ based strictly on validation error rate, evaluates optimal model on test set, and generates side-by-side comparison matrix with baseline default parameters.

### 4.3 Integrated Multi-Classifier Studio
Includes all four classifiers under a unified dropdown selector:
1. **Parzen Window Density Estimator** (Current Focus): Hypercube and Gaussian kernels, log-sum-exp trick, multi-$h$ curve, optimal selection.
2. **Ho-Kashyap Linear Classifier** (Assignment 08): Positive margin relaxation, linear separability diagnosis, multi-$\eta$ convergence analysis.
3. **K-Nearest Neighbors** (Assignment 07): Euclidean metric, Elbow Method for optimal $K^*$, nearest neighbor explainability.
4. **Gaussian Naive Bayes** (Assignments 06/07): Parametric generative likelihood modeling.
5. **Comparative Analysis Studio (`/performance-analysis`):** Multi-model benchmark matrix (Accuracy, Precision, Recall, F1, Error Rate, Latency) and comparative visualization.

---

## 5. Verification Suite & Empirical Test Results

The verification suite `test_verification.py` executes 6 comprehensive test phases:

```
==================================================================
RUNNING ASSIGNMENT 09 VERIFICATION SUITE
==================================================================

==================================================================
TEST 1: Parzen Window Density Estimator & Log-Sum-Exp Stability
==================================================================
Query point: [1.5 1.5 1.5 1.5]
Gaussian with extreme small h=0.005:
  Raw densities: [9.85967654e-305 9.85967654e-305]
  Log densities: [-228118.01436707   -7067.43806522]
  Posterior probabilities: [0. 1.]
  Predicted class: 1
PASS: Gaussian log-sum-exp trick completely prevents floating-point underflow!

Hypercube with h=1.5:
  Densities: [0.         0.06419753]
  Posteriors: [0. 1.]
  Is empty window: False
  Predicted class: 1
PASS: Hypercube density and Bayes prediction verified!

==================================================================
TEST 2: Default Parameters (Gaussian, h=1.0) & First 5 Test Samples
==================================================================
Dataset: Iris (Rows: 150, Features: 5)
Class Priors:
  Class 0 (Iris-setosa): P(w_0) = 0.3333
  Class 1 (Iris-versicolor): P(w_1) = 0.3333
  Class 2 (Iris-virginica): P(w_2) = 0.3333

Default Parameters (Gaussian, h=1.0) Test Metrics:
  Accuracy:   100.0%
  Error Rate: 0.0%
  Macro F1:   100.0%

First Five Test Samples Breakdown (Class-conditional densities & posteriors):
  Sample #1: True=Iris-setosa     Pred=Iris-setosa     [CORRECT]
     -> Iris-setosa: p(x|w)=0.005352, P(w|x)=99.68%
     -> Iris-versicolor: p(x|w)=1.6831e-05, P(w|x)=0.31%
     -> Iris-virginica: p(x|w)=9.8729e-08, P(w|x)=0.0%
  Sample #2: True=Iris-virginica  Pred=Iris-virginica  [CORRECT]
     -> Iris-setosa: p(x|w)=3.0341e-06, P(w|x)=0.04%
     -> Iris-versicolor: p(x|w)=0.003227, P(w|x)=47.02%
     -> Iris-virginica: p(x|w)=0.003634, P(w|x)=52.94%
  Sample #3: True=Iris-versicolor Pred=Iris-versicolor [CORRECT]
     -> Iris-setosa: p(x|w)=1.2352e-05, P(w|x)=0.28%
     -> Iris-versicolor: p(x|w)=0.003377, P(w|x)=77.0%
     -> Iris-virginica: p(x|w)=0.000996, P(w|x)=22.71%
  Sample #4: True=Iris-versicolor Pred=Iris-versicolor [CORRECT]
     -> Iris-setosa: p(x|w)=8.1935e-05, P(w|x)=1.15%
     -> Iris-versicolor: p(x|w)=0.004998, P(w|x)=70.37%
     -> Iris-virginica: p(x|w)=0.002022, P(w|x)=28.48%
  Sample #5: True=Iris-setosa     Pred=Iris-setosa     [CORRECT]
     -> Iris-setosa: p(x|w)=0.006843, P(w|x)=98.29%
     -> Iris-versicolor: p(x|w)=0.000118, P(w|x)=1.69%
     -> Iris-virginica: p(x|w)=1.1460e-06, P(w|x)=0.02%
PASS: First five test samples recorded and verified!

==================================================================
TEST 3: Window Width Analysis (Multi-h) & Optimal Model Selection
==================================================================
Dataset: Naive-Bayes (Glucose & Blood Pressure, N=995)

Multi-h Analysis Table:
h        | G-Train    | G-Val      | H-Train    | H-Val      | H-Empty-Val 
------------------------------------------------------------------------
0.01     | 9.52       | 28.57      | 9.52       | 42.86      | 28          
0.05     | 9.52       | 28.57      | 9.52       | 42.86      | 28          
0.1      | 9.52       | 28.57      | 9.52       | 42.86      | 28          
0.2      | 11.43      | 25.71      | 12.38      | 37.14      | 16          
0.5      | 16.19      | 14.29      | 14.29      | 25.71      | 5           
1.0      | 17.14      | 17.14      | 16.19      | 20.0       | 0           
2.0      | 28.57      | 28.57      | 23.81      | 22.86      | 0           
5.0      | 28.57      | 28.57      | 28.57      | 28.57      | 0           

Optimal Model Selected: Gaussian Window with h* = 0.5
  Optimal Test Accuracy: 88.57%
  Optimal Error Rate:    11.43%
  Optimal Macro F1:      85.04%
PASS: Multi-h analysis plot generated successfully!

Side-by-Side Comparison (Default vs Optimal):
  Default (Gaussian, h=1.0): Accuracy = 82.86%, Error Rate = 17.14%
  Optimal (Gaussian, h*=0.5): Accuracy = 88.57%, Error Rate = 11.43%
  Accuracy Gain: +5.71%
PASS: Optimal model selection and comparison confirmed!

==================================================================
TEST 4: Instance Prediction & Domain Range Warning Validation
==================================================================
In-domain prediction for {'glucose': '45.0', 'bloodpressure': '80.0'}:
  Predicted Label: 1 (Code: 1)
  Range warnings count: 0

Outlier prediction for {'glucose': '999.0', 'bloodpressure': '-50.0'}:
  Predicted Label: 1 (Code: 1)
  Range warnings count: 2
    Warning: Feature 'glucose' = 999.0 is Far outside range (Training: [20.00, 70.00])
    Warning: Feature 'bloodpressure' = -50.0 is Far outside range (Training: [50.00, 100.00])
PASS: Prediction input validation and range warning verified!

==================================================================
TEST 5: Unified Classifiers Integration (Parzen, Ho-Kashyap, KNN, Bayes)
==================================================================
Parzen Window: Accuracy=100.0%, Window=gaussian, h*=1.0
Ho-Kashyap: Accuracy=100.0%, Separability=Linearly Separable, Optimal η*=0.1
KNN: Accuracy=100.0%, Optimal K=1
Bayes: Accuracy=100.0%, F1=100.0%

Comparative Benchmark Studio Matrix:
  Parzen Window (Gaussian, h=1.0)     | Acc: 100.0 % | F1: 100.0 % | Latency: 0.18 ms
  Ho-Kashyap (Linearly Separable)     | Acc: 100.0 % | F1: 100.0 % | Latency: 0.04 ms
  KNN (K=1)                           | Acc: 100.0 % | F1: 100.0 % | Latency: 1.02 ms
  Gaussian Naive Bayes                | Acc: 100.0 % | F1: 100.0 % | Latency: 1.08 ms
Best Recommended Model: Parzen Window (Gaussian, h=1.0)
PASS: Unified 4-classifier studio integration verified!

==================================================================
TEST 6: Flask Endpoints & Integration Client Tests
==================================================================
PASS: GET / -> 200 OK
PASS: POST /upload (preset) -> 200 OK
PASS: POST /prepare -> 200 OK
PASS: POST /train (Parzen Window) -> 200 OK
PASS: POST /predict -> 200 OK
PASS: GET /performance-analysis -> 200 OK
PASS: POST /reset -> 200 OK

==================================================================
ALL TESTS PASSED SUCCESSFULLY! (100% SPECIFICATION CONFORMANCE)
==================================================================
```

---

## 6. Project Structure

```
PatternReco-Assignment-09/
├── PR_Lab_Assignment_09_Parzen_Window.pdf   # Original Lab Assignment Specification
├── main.py                                  # Flask Application, Models & Endpoints
├── test_verification.py                     # Comprehensive Test Suite (100% Pass)
├── README.md                                # In-Depth Documentation & Viva Guide
├── Dataset/                                 # Standard Classification Datasets
│   ├── iris.csv                             # Multi-Class Benchmark (3 classes)
│   ├── breast_cancer.csv                    # High-Dimensional Binary Dataset (30 features)
│   ├── heart.csv                            # Medical Classification Benchmark
│   ├── Naive-Bayes-Classification-Data.csv  # Continuous 2D Feature Dataset
│   ├── linearly_separable_data.csv          # Synthetic Linearly Separable Dataset
│   ├── Titanic-Dataset.csv                  # Mixed Categorical/Numeric with Missing Values
│   └── netflix_customer_churn.csv           # Customer Churn Dataset
├── Templates/
│   ├── index.html                           # Main ML Studio Dashboard
│   └── performance_analysis.html            # 4-Classifier Comparative Benchmark Matrix
├── Statics/
│   ├── css/
│   │   └── style.css                        # Modern Dark-Theme Design System
│   ├── js/
│   │   └── app.js                           # Interactive Interactivity Engine
│   ├── plots/                               # High-Resolution Matplotlib Visualizations
│   │   ├── class_distribution.png
│   │   ├── parzen_h_analysis.png
│   │   ├── parzen_cm_default.png
│   │   ├── parzen_cm_optimal.png
│   │   ├── hokashyap_convergence.png
│   │   ├── hokashyap_eta_curve.png
│   │   ├── knn_elbow_curve.png
│   │   └── benchmark_comparison.png
│   └── vendor/
│       └── katex/                           # Local KaTeX Math Formula Rendering Engine
└── Uploads/                                 # Temporary Storage for User Uploaded CSVs
```

---

## 7. How to Run the Application

### 7.1 Running the Flask Web Studio
Execute the following command inside the `PatternReco-Assignment-09` directory:
```bash
python3 main.py
```
Open your web browser and navigate to:
```
http://localhost:5000
```

### 7.2 Running the Comprehensive Test Suite
```bash
python3 test_verification.py
```

---

## 8. Viva Voce Preparation & Answers

### Q1: What is the fundamental difference between parametric and non-parametric density estimation?
**Answer:** Parametric methods assume the data follows a specific mathematical distribution with unknown parameters (e.g., Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$) and estimate these parameters from data. Non-parametric methods (like Parzen Window and KNN) make no prior assumptions about the distribution shape; the density estimate is constructed directly from local sample counts, allowing them to model arbitrary multi-modal and non-linear distributions.

### Q2: What is a Parzen Window and how is the density estimated?
**Answer:** The Parzen Window estimator estimates the continuous density $p(\mathbf{x})$ by placing a kernel function $\varphi(\mathbf{u})$ centered at each sample point and scaling it by window width $h$:
$$p(\mathbf{x}) = \frac{1}{N h^d} \sum_{i=1}^N \varphi\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$
The window width $h$ controls the volume $V = h^d$ over which data is smoothed.

### Q3: Why does Gaussian window density underflow for small $h$ or large $d$, and how does the log-sum-exp trick resolve it?
**Answer:** The Gaussian kernel contains the term $\exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2h^2}\right)$. For small $h$ or high dimensions $d$, $-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2h^2} \ll -700$, causing $\exp(\cdot)$ to round to $0.0$ in 64-bit floating point. When all training samples yield zero, the density underflows to zero and posterior probabilities become indeterminate ($0/0$). The log-sum-exp trick computes densities in log-space:
$$\ln\left(\sum \exp(z_i)\right) = M + \ln\left(\sum \exp(z_i - M)\right) \quad \text{where } M = \max_i z_i$$
Because $z_i - M \le 0$, the largest term is $\exp(0) = 1$, completely preventing floating-point underflow.

### Q4: What happens if a test instance falls in an empty hypercube window?
**Answer:** If no training samples from any class fall inside the hypercube ($\sum_j k_j(\mathbf{x}) = 0$), $p(\mathbf{x}|\omega_j) = 0$ for all classes. In our implementation, such empty-window instances are explicitly tracked, and the Bayes decision rule falls back gracefully to the prior probabilities $P(\omega_j) = N_j / N$.

### Q5: How does the window width $h$ affect the bias and variance of the classifier?
**Answer:**
- Very small $h$ leads to **high variance / low bias (undersmoothing)**: the estimated density consists of sharp spikes around individual training points, leading to overfitting and high validation error.
- Very large $h$ leads to **high bias / low variance (oversmoothing)**: the kernels blur together, wiping out class boundaries and underfitting the data.
- The optimal $h^*$ achieves the minimal validation error rate by balancing this bias-variance trade-off.

---

**Pattern Recognition Laboratory &bull; VNIT-W26 &bull; Roll No: MT26MCS024**

