# Pattern Recognition Lab: Assignment 07
## Title: Web-Based Machine Learning Application Using Flask for Classification with K-Nearest Neighbors (KNN) & Integrated Bayes Classifier

**Course:** Pattern Recognition Laboratory (VNIT-W26)  
**Student Roll No:** MT26MCS024  
**Application Name:** **Classifiers** (Integrated Pattern Recognition Studio)  
**Lab Session Date:** 27th October 2026  

---

## 1. Problem Statement & Objectives

### Problem Statement
Develop a production-grade web application using Flask that allows users to upload any standard classification CSV dataset, inspect and pre-process the data (including handling missing values and duplicate rows), standardize numerical features using Scikit-Learn `StandardScaler`, train and evaluate a K-Nearest Neighbors (KNN) classifier across different values of $K$, analyze classification error rate using the **Elbow Method** to select the optimal $K$, compare baseline ($K=1$) versus optimal elbow-point $K$ performance, and predict the class of newly provided user query instances.

Furthermore, integrate this application with the **Bayes Classifier** developed in previous lab assignments, renaming the unified application to **Classifiers**, allowing users to select either classifier from a dropdown interface.

### Learning Objectives
1. **Flask Web Framework:** Build an interactive, stateful web-based Machine Learning studio with modular routes and session isolation.
2. **Data Ingestion & Pre-processing:** Load and validate tabular datasets (.csv), handle missing values via Scikit-Learn `SimpleImputer`, and remove duplicate samples.
3. **Feature Standardization:** Apply Scikit-Learn `StandardScaler` to normalize disparate feature scales, and empirically analyze its impact on distance-based metrics.
4. **KNN Implementation from First Principles:** Implement the K-Nearest Neighbors algorithm with Euclidean distance, majority voting, tie-breaking, and neighbor explainability.
5. **Comprehensive Metrics Evaluation:** Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmaps.
6. **Hyperparameter Optimization via the Elbow Method:** Systematically evaluate classification error rate across $K \in [1, 40]$:
   $$\text{Error Rate} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Predictions}} = 1 - \text{Accuracy}$$
   Identify the sharp bend ("elbow point") and evaluate side-by-side performance gains ($K=1$ vs. $K_{\text{elbow}}$).
7. **Prediction & Explainability:** Build a dynamic form interface validating inputs and rendering class labels, codes, posterior probabilities, and nearest neighbor proximity breakdowns.
8. **Generative vs. Non-Parametric Integration:** Unify KNN and Gaussian Naive Bayes under a single dropdown selector with multi-dataset comparative benchmarking.

---

## 2. Theoretical Foundations

### 2.1 K-Nearest Neighbors (KNN) Algorithm
KNN is an instance-based, non-parametric supervised learning algorithm. Given a training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $\mathbf{x}_i \in \mathbb{R}^D$ and a query vector $\mathbf{x}_q$:

1. **Distance Metric:** Calculate the Euclidean distance between $\mathbf{x}_q$ and all training instances $\mathbf{x}_i$:
   $$d(\mathbf{x}_q, \mathbf{x}_i) = \sqrt{\sum_{j=1}^D (x_{q,j} - x_{i,j})^2} = \|\mathbf{x}_q - \mathbf{x}_i\|_2$$
2. **Neighborhood Selection:** Identify the indices of the $K$ smallest distances:
   $$\mathcal{N}_K(\mathbf{x}_q) = \arg\min^{(K)}_{i \in \{1,\dots,N\}} d(\mathbf{x}_q, \mathbf{x}_i)$$
3. **Majority Voting:** Assign $\mathbf{x}_q$ to the class with the majority vote among its $K$ neighbors:
   $$\hat{y}_q = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_K(\mathbf{x}_q)} \mathbb{I}(y_i = c)$$
   In case of a tie, the candidate class with the lowest average distance to the query instance is chosen.

### 2.2 Feature Standardization (Scikit-Learn `StandardScaler`)
Because Euclidean distance directly aggregates squared differences across all features, unscaled features with large ranges (e.g., Income $\in [10^4, 10^5]$) completely dominate features with small ranges (e.g., Age $\in [0, 100]$). 

`StandardScaler` standardizes each feature independently:
$$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$
Where:
- $\mu_j = \frac{1}{N}\sum_{i=1}^N x_{i,j}$ (feature mean)
- $\sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_{i,j} - \mu_j)^2}$ (standard deviation)

### 2.3 The Elbow Method for Optimal $K$
The hyperparameter $K$ controls the model's inductive bias:
- **$K = 1$:** Zero bias on training samples, high variance. Highly sensitive to noise and outliers.
- **$K \gg 1$:** High bias, low variance. Over-smoothes class boundaries, resulting in misclassifications of minority classes.

The **Elbow Method** plots $K$ against Error Rate:
$$\text{Error Rate}(K) = \frac{\sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\hat{y}_i(K) \neq y_i)}{N_{\text{test}}}$$
The optimal $K$ is chosen at the sharp bend ("elbow point") where further increasing $K$ yields diminishing returns in error reduction.

### 2.4 Bayes Classifier (Integrated Generative Model)
From Assignment 06, the Gaussian Naive Bayes classifier computes:
$$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{j=1}^D P(x_j \mid C_k)$$
Where class-conditional densities are modeled as:
$$P(x_j \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,j}^2}} \exp\left( -\frac{(x_j - \mu_{k,j})^2}{2\sigma_{k,j}^2} \right)$$
Decision rule:
$$\hat{y} = \arg\max_{k} \left[ \ln P(C_k) + \sum_{j=1}^D \ln P(x_j \mid C_k) \right]$$

---

## 3. Directory Structure

```
PatternReco-Assignment-07/
├── Lab Assignment 07.pdf               # Lab assignment specifications
├── README.md                           # Comprehensive documentation & report
├── main.py                             # Flask application named "Classifiers"
├── Dataset/                            # Benchmark datasets for testing
│   ├── heart.csv                       # Heart disease classification
│   ├── Titanic-Dataset.csv             # Titanic survival (mixed types, missing values)
│   ├── breast_cancer.csv               # Breast Cancer Wisconsin (30 continuous features)
│   ├── Naive-Bayes-Classification-Data.csv # Blood Pressure & Glucose dataset
│   └── netflix_customer_churn.csv      # Customer churn classification
├── Statics/
│   ├── css/
│   │   └── style.css                   # Premium responsive design system
│   ├── js/
│   │   └── app.js                      # Interactive UI scripts (live slider, feature toggles)
│   └── plots/                          # Generated plots (Elbow curve, CM heatmaps)
├── Templates/
│   ├── index.html                      # Main studio interface
│   └── performance_analysis.html       # Multi-dataset & multi-model benchmark matrix
└── Uploads/                            # Temporary upload cache (auto-cleared)
```

---

## 4. Key Functional Features Implemented

1. **Dataset Ingestion & Inspection:**
   - Validates CSV file presence, non-empty size, correct format, row count, and column structure.
   - Computes total dimensions, missing value counts, and duplicate row counts.
   - Previews column metadata (type, non-null count, unique values, category) and first 5 sample rows.
   - Session flushing: Previous uploads, models, and plots are cleared upon new dataset ingestion.

2. **Pre-Training Preparation & Standardization:**
   - Conditional Pre-Processing Alert: If missing values or duplicates exist, the pre-processing panel is displayed. Otherwise, displays: *"Dataset Clean & Ready: Zero missing values and duplicates detected."*
   - Imputation via Scikit-Learn `SimpleImputer` (median/mean for numeric; mode for categorical).
   - Duplicate removal option.
   - User choice whether to use standardized features (`StandardScaler`) or raw unscaled features.

3. **Classifier Selection (`Classifiers` App):**
   - Dropdown selector allows user to choose between:
     - **K-Nearest Neighbors (KNN)**
     - **Bayes Classifier (Gaussian Naive Bayes)**
   
4. **KNN Training & The Elbow Method:**
   - Step 1: Evaluates $K=1$ baseline model (Accuracy, Precision, Recall, F1-Score, Error Rate, Confusion Matrix).
   - Step 2: Systematically evaluates $K=1$ to $K_{\max}$ (user selectable up to 40).
   - Step 3: Plots the clean **'K' vs. Error Rate** curve for visual inspection without algorithmic bias.
   - Step 4: The user visually inspects the graph to identify the sharp bend (elbow point) and selects their chosen value of $K$ from the dropdown.
   - Step 5: Upon selection, re-evaluates the model at the user-chosen $K$, highlights the point on the graph, and presents a side-by-side comparative scorecard ($K=1$ vs. User-Selected Elbow $K$) with delta gains highlighted ($\Delta\%$ Accuracy, $\Delta\%$ Error Reduction).

5. **Numeric Target Class Renaming Option:**
   - Detects whether target class values are numerically assigned (e.g., `0` and `1` in `heart.csv`, `Titanic-Dataset.csv`, etc.).
   - Provides an option for the user to assign custom human-readable names (e.g., `0` $\to$ *"Healthy / Did Not Survive"*, `1` $\to$ *"Heart Disease / Survived"*).
   - Display labels are dynamically reflected across:
     - Target variable distribution charts and breakdown tables.
     - Confusion matrix heatmap axes for both KNN ($K=1$, $K_{\text{elbow}}$) and Bayes Classifier.
     - Prediction output badges, probability distributions, and nearest neighbor explainability rows.
   - Includes on-the-fly quick rename form allowing labels to be updated at any point without retraining or resetting.

6. **Seamless Scroll Position Preservation (No Page Resets to Top):**
   - Automatically maintains the user's viewport and active section on every workflow operation (`/upload`, `/prepare`, `/train`, `/select_elbow_k`, `/predict`, `/rename_classes`).
   - Combines Flask route anchor redirects with client-side `sessionStorage` scroll restoration and smooth transitions to eliminate visual disorientation.

7. **Prediction of New Instances:**
   - Dynamic input fields based on selected features.
   - Handles numeric and categorical inputs with validation and automatic scaling.
   - Displays predicted class code (e.g. 0 or 1), meaningful label, and descriptive interpretation.
   - For KNN: Displays the exact list of the $K$ Nearest Neighbors with individual Euclidean distances, row indices, and target classes.
   - For Bayes: Displays posterior probability distribution and log-likelihood breakdown.

8. **Dedicated Comparative Analysis Matrix (`/performance-analysis`):**
   - Evaluates 4 models simultaneously across 4 benchmark datasets:
     1. KNN ($K=1$, Raw Unscaled)
     2. KNN ($K=1$, Standardized)
     3. KNN (Optimal Elbow $K$, Standardized)
     4. Bayes Classifier (Gaussian Naive Bayes)

---

## 5. How to Run the Application

### Prerequisites
Ensure Python 3.9+ and required packages are installed:
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```

### Starting the Flask Server
Navigate to the assignment directory and execute `main.py`:
```bash
cd "PatternReco-Assignment-07"
python3 main.py
```

### Accessing the Web Application
Open your web browser and navigate to:
```
http://localhost:5000
```
- Main Studio: `http://localhost:5000/`
- Comparative Analysis: `http://localhost:5000/performance-analysis`

---

## 6. Experimental Results & Observations

| Dataset | Metric | KNN (K=1, Raw) | KNN (K=1, Scaled) | KNN (Optimal Elbow K) | Bayes Classifier |
|---|---|---|---|---|---|
| **Heart Disease** | Accuracy / Error | 55.26% / 44.74% | 73.68% / 26.32% | **85.53% / 14.47% (K=16)** | 82.89% / 17.11% |
| **Titanic Survival** | Accuracy / Error | 69.10% / 30.90% | 70.79% / 29.21% | **78.65% / 21.35% (K=5)** | 77.53% / 22.47% |
| **Breast Cancer** | Accuracy / Error | 90.71% / 9.29% | 93.81% / 6.19% | **96.46% / 3.54% (K=9)** | 92.92% / 7.08% |
| **Blood Pressure / Glucose** | Accuracy / Error | 72.00% / 28.00% | 74.00% / 26.00% | **78.00% / 22.00% (K=7)** | 76.00% / 24.00% |

### Key Experimental Insights:
1. **Impact of Feature Standardization:**
   Standardizing features using Scikit-Learn `StandardScaler` resulted in massive accuracy gains (+18.42% on Heart Disease and +3.10% on Breast Cancer), demonstrating that unscaled continuous features disproportionately distort Euclidean distances.
2. **Impact of the Elbow Method:**
   Moving from $K=1$ to the optimal elbow $K$ reduced the error rate by 11.85% on Heart Disease and 7.86% on Titanic. $K=1$ suffered from overfitting to noise, whereas the optimal $K$ generalized with lower variance.
3. **KNN vs. Bayes Classifier:**
   KNN with optimal $K$ achieved superior performance on datasets with complex non-linear feature interactions, while Bayes Classifier performed consistently well with zero hyperparameter tuning.
