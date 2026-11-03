import os
import json
import shutil
import signal
import threading
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'Templates'),
    static_folder=os.path.join(BASE_DIR, 'Statics')
)
app.secret_key = 'classifiers_hokashyap_assignment_08_secret'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'Uploads')
app.config['DATASET_FOLDER'] = os.path.join(BASE_DIR, 'Dataset')
app.config['PLOT_FOLDER'] = os.path.join(BASE_DIR, 'Statics', 'plots')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)


# =====================================================================
# NUMERIC VALUE & CLASS NORMALIZATION HELPERS
# =====================================================================

def normalize_class_label(val):
    """Normalizes class representation (e.g. 0.0 -> '0', '1' -> '1', 'No' -> 'No')."""
    s = str(val).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def is_numeric_value(val):
    """Checks whether a value or string can be safely parsed as a numeric float."""
    try:
        float(str(val).strip())
        return True
    except (ValueError, TypeError):
        return False


# =====================================================================
# CUSTOM LABEL ENCODER
# =====================================================================

class CustomLabelEncoder:
    """Custom Label Encoder implemented from scratch with custom display names support."""
    def __init__(self):
        self.classes_ = []
        self.mapping = {}
        self.inverse_mapping = {}
        self.display_names = {}

    def fit(self, values):
        unique_set = list(set(normalize_class_label(v) for v in values))
        try:
            # Sort numerically if all values represent numbers
            unique_vals = sorted(unique_set, key=lambda x: float(x))
        except (ValueError, TypeError):
            unique_vals = sorted(unique_set)

        self.classes_ = unique_vals
        self.mapping = {val: idx for idx, val in enumerate(unique_vals)}
        self.inverse_mapping = {idx: val for idx, val in enumerate(unique_vals)}
        return self

    def set_display_names(self, name_dict):
        """Allows overriding class codes (e.g. '0', '1') with user-defined names."""
        for k, v in name_dict.items():
            norm_k = normalize_class_label(k)
            if str(v).strip():
                self.display_names[norm_k] = str(v).strip()

    def get_display_name(self, raw_or_idx):
        if isinstance(raw_or_idx, (int, np.integer)):
            raw_str = self.inverse_mapping.get(int(raw_or_idx), str(raw_or_idx))
        else:
            raw_str = normalize_class_label(raw_or_idx)
        return self.display_names.get(raw_str, raw_str)

    def fit_transform(self, values):
        self.fit(values)
        return np.array([self.mapping[normalize_class_label(v)] for v in values])

    def transform(self, values):
        encoded = []
        for v in values:
            s_val = normalize_class_label(v)
            if s_val in self.mapping:
                encoded.append(self.mapping[s_val])
            else:
                encoded.append(0)
        return np.array(encoded)

    def inverse_transform(self, indices):
        return [self.inverse_mapping.get(int(idx), str(idx)) for idx in indices]


# =====================================================================
# CUSTOM HO-KASHYAP CLASSIFIER (IMPLEMENTED FROM SCRATCH USING NUMPY)
# =====================================================================

class CustomHoKashyapClassifier:
    """
    Ho-Kashyap Linear Classification Procedure implemented from scratch using NumPy.
    
    Finds a weight vector a = [a_0, a_1, ..., a_d]^T and a positive margin vector b > 0
    such that Y a = b > 0 in the augmented normalized feature space:
      - Augmented feature vectors: y_aug = [1, x_1, x_2, ..., x_d]^T (a_0 is bias threshold)
      - Class normalization: y_i = z_i * y_aug_i, where z_i in {+1, -1}
      - Design matrix Y: shape (N, d+1)
      - Pseudo-inverse: Y_pinv = (Y^T Y)^(-1) Y^T
      - Error vector: e(k) = Y a(k) - b(k)
      - Modified positive error: e+(k) = 0.5 * (e(k) + |e(k)|) = max(0, e(k))
      - Margin update: b(k+1) = b(k) + 2 * eta * e+(k)
      - Weight update: a(k+1) = Y_pinv * b(k+1)
      
    Tracks squared error ||e(k)||^2 and misclassified training samples (a^T y_i <= 0).
    Tests for linear separability vs non-separability (e(k) <= 0 indicates non-separability).
    """
    def __init__(self, learning_rate=0.1, max_iter=500, tol=1e-5, b_init=1.0):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.b_init = float(b_init)
        self.a_ = None               # Final weight vector a
        self.b_ = None               # Final margin vector b
        self.history_ = {
            'iteration': [],
            'squared_error': [],
            'misclassified': [],
            'b_min': [],
            'b_max': []
        }
        self.converged_ = False
        self.separability_ = 'Unknown'
        self.iterations_run_ = 0
        self.stop_reason_ = ''
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Ho-Kashyap procedure requires a binary classification problem (exactly 2 classes). Found {len(classes)} classes: {classes}")

        self.classes_ = classes
        # Map: classes_[1] -> +1, classes_[0] -> -1
        z = np.where(y == self.classes_[1], 1.0, -1.0)[:, None]

        N, d = X.shape
        # Augmented feature matrix: [1, x_1, x_2, ..., x_d]
        X_aug = np.hstack([np.ones((N, 1), dtype=np.float64), X])
        Y = z * X_aug  # Shape: (N, d+1)

        # Moore-Penrose pseudo-inverse of Y
        Y_pinv = np.linalg.pinv(Y)
        b = np.full(N, self.b_init, dtype=np.float64)
        a = Y_pinv @ b

        self.history_ = {
            'iteration': [],
            'squared_error': [],
            'misclassified': [],
            'b_min': [],
            'b_max': []
        }
        prev_err = float('inf')
        stagnant_count = 0

        for k in range(1, self.max_iter + 1):
            e = Y @ a - b
            sq_err = float(np.sum(e ** 2))
            # Misclassified training samples: samples with a^T y_i <= 0
            # Since row i of Y is y_i^T, (Y @ a)_i is a^T y_i
            margins = Y @ a
            misclassified = int(np.sum(margins <= 0))
            b_min = float(np.min(b))
            b_max = float(np.max(b))

            self.history_['iteration'].append(k)
            self.history_['squared_error'].append(round(sq_err, 6))
            self.history_['misclassified'].append(misclassified)
            self.history_['b_min'].append(round(b_min, 6))
            self.history_['b_max'].append(round(b_max, 6))
            self.iterations_run_ = k

            # Stopping Criterion 1: Convergence / Separable Solution Found
            if sq_err <= self.tol or (misclassified == 0 and sq_err < 1e-2):
                self.converged_ = True
                self.separability_ = 'Linearly Separable'
                self.stop_reason_ = f"Converged at iteration {k} with squared error {sq_err:.6e} and 0 misclassifications."
                break

            # Stopping Criterion 2: Linear Non-Separability Guarantee (Theorem: e(k) <= 0 with e != 0)
            if np.all(e <= 1e-8) and sq_err > self.tol:
                self.converged_ = False
                self.separability_ = 'Linearly Non-Separable'
                self.stop_reason_ = f"Linear non-separability condition e(k) <= 0 satisfied at iteration {k}."
                break

            # Stopping Criterion 3: Error Plateau / Numerical Stagnation
            if abs(prev_err - sq_err) < 1e-7:
                stagnant_count += 1
                if stagnant_count >= 5:
                    self.converged_ = False
                    self.separability_ = 'Linearly Non-Separable (Plateau Reached)'
                    self.stop_reason_ = f"Error plateau reached at iteration {k} (consecutive error change < 1e-7)."
                    break
            else:
                stagnant_count = 0

            prev_err = sq_err

            # Ho-Kashyap margin & weight update
            # e+(k) = 0.5 * (e + |e|) = max(0, e)
            e_plus = np.maximum(0.0, e)
            b = b + 2.0 * self.learning_rate * e_plus
            a = Y_pinv @ b

        if not self.converged_ and self.separability_ == 'Unknown':
            if misclassified == 0:
                self.converged_ = True
                self.separability_ = 'Linearly Separable'
                self.stop_reason_ = f"Separating hyperplane found (0 training misclassifications)."
            else:
                self.separability_ = 'Linearly Non-Separable / Inconclusive'
                self.stop_reason_ = f"Maximum iterations ({self.max_iter}) reached without zero training error."

        self.a_ = a
        self.b_ = b
        return self

    def decision_function(self, X):
        """Computes linear discriminant g(x) = a^T x_aug = a_0 + sum(a_j * x_j)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_aug = np.hstack([np.ones((len(X), 1), dtype=np.float64), X])
        return X_aug @ self.a_

    def predict(self, X):
        """Classifies samples: class 1 if g(x) > 0 else class 0."""
        scores = self.decision_function(X)
        return np.where(scores > 0, self.classes_[1], self.classes_[0])

    def predict_proba_pseudo(self, X):
        """Computes sigmoid-calibrated pseudo-probabilities from discriminant scores."""
        scores = self.decision_function(X)
        probs_class1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        return np.column_stack([1.0 - probs_class1, probs_class1])

    def get_weight_summary(self, feature_names=None):
        """Returns structured weight vector components including bias and feature weights."""
        if self.a_ is None:
            return {}
        bias = float(self.a_[0])
        weights = [float(w) for w in self.a_[1:]]
        if feature_names is None or len(feature_names) != len(weights):
            feature_names = [f"x_{i+1}" for i in range(len(weights))]

        feat_weights = []
        for name, w in zip(feature_names, weights):
            feat_weights.append({
                'feature': name,
                'weight': round(w, 5),
                'abs_weight': round(abs(w), 5),
                'sign': '+' if w >= 0 else '-'
            })

        return {
            'bias_a0': round(bias, 5),
            'feature_weights': feat_weights,
            'raw_vector': [round(float(v), 5) for v in self.a_]
        }

    def get_margin_summary(self):
        """Returns min, max, mean, and positivity status of the final margin vector b."""
        if self.b_ is None:
            return {}
        return {
            'min_b': round(float(np.min(self.b_)), 5),
            'max_b': round(float(np.max(self.b_)), 5),
            'mean_b': round(float(np.mean(self.b_)), 5),
            'length': len(self.b_),
            'all_positive': bool(np.all(self.b_ > 0))
        }

    def explain_instance(self, x, feature_names=None):
        """Provides an explainability breakdown for a single input vector x."""
        x = np.asarray(x, dtype=np.float64).flatten()
        bias = float(self.a_[0])
        feature_contributions = []
        total_score = bias

        for idx, (val, w) in enumerate(zip(x, self.a_[1:])):
            contrib = float(w * val)
            total_score += contrib
            fname = feature_names[idx] if (feature_names and idx < len(feature_names)) else f"Feature {idx+1}"
            feature_contributions.append({
                'feature': fname,
                'input_value': round(float(val), 4),
                'weight': round(float(w), 5),
                'contribution': round(contrib, 4)
            })

        pred_class = int(self.classes_[1]) if total_score > 0 else int(self.classes_[0])
        # Logistic confidence
        conf_prob = 1.0 / (1.0 + np.exp(-np.clip(abs(total_score), -50, 50)))
        confidence_pct = round(conf_prob * 100.0, 2)

        return {
            'bias': round(bias, 5),
            'discriminant_score': round(total_score, 4),
            'predicted_class': pred_class,
            'confidence': confidence_pct,
            'feature_contributions': feature_contributions,
            'is_positive': total_score > 0
        }


# =====================================================================
# CUSTOM K-NEAREST NEIGHBORS CLASSIFIER (FROM ASSIGNMENT 07)
# =====================================================================

class CustomKNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from first principles.
    Uses Euclidean distance metric, majority voting, tie-breaking by average distance,
    and returns neighbor indices and individual distances for full explainability.
    """
    def __init__(self, n_neighbors=1):
        self.n_neighbors = max(1, int(n_neighbors))
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _predict_single(self, x):
        diff = self.X_train - x
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        k = min(self.n_neighbors, len(self.X_train))
        neighbor_indices = np.argsort(distances)[:k]
        neighbor_labels = self.y_train[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]

        # Majority vote
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = counts.max()
        candidate_labels = unique_labels[counts == max_count]

        if len(candidate_labels) > 1:
            # Tie-breaking: pick candidate with smallest average neighbor distance
            best_label = candidate_labels[0]
            min_avg_d = float('inf')
            for lbl in candidate_labels:
                avg_d = np.mean(neighbor_distances[neighbor_labels == lbl])
                if avg_d < min_avg_d:
                    min_avg_d = avg_d
                    best_label = lbl
            pred_label = best_label
        else:
            pred_label = candidate_labels[0]

        vote_counts = {int(c): int(np.sum(neighbor_labels == c)) for c in self.classes_}
        vote_proportions = {int(c): float(vote_counts[int(c)] / k) for c in self.classes_}

        return pred_label, vote_proportions, neighbor_indices, neighbor_distances

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = []
        for x in X:
            pred_label, _, _, _ = self._predict_single(x)
            preds.append(pred_label)
        return np.array(preds)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probas = []
        for x in X:
            _, probs, _, _ = self._predict_single(x)
            probas.append([probs[int(c)] for c in self.classes_])
        return np.array(probas)

    def explain_instance(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        pred_label, probs, indices, dists = self._predict_single(x)
        neighbor_details = []
        for idx, dist in zip(indices, dists):
            neighbor_details.append({
                'train_index': int(idx),
                'distance': round(float(dist), 4),
                'class': int(self.y_train[idx])
            })
        return {
            'predicted_class': int(pred_label),
            'probabilities': probs,
            'neighbors': neighbor_details
        }


# =====================================================================
# CUSTOM BAYES CLASSIFIER (FROM ASSIGNMENTS 06/07)
# =====================================================================

class PatternClassifiers:
    """
    Custom Naive Bayes Classifier implemented from first principles.
    Calculates class priors and class-conditional Gaussian density parameters.
    """
    def __init__(self, eps=1e-9):
        self.classes = []
        self.features = []
        self.priors = {}
        self.parameters = {}
        self.eps = eps

    def displayParam(self):
        return {
            'priors': self.priors,
            'parameters': self.parameters
        }

    def bayesianClassifier(self, x, mean, variance):
        if variance <= 0:
            variance = self.eps
        coef = 1.0 / np.sqrt(2.0 * np.pi * variance)
        exp_term = np.exp(-((x - mean) ** 2) / (2.0 * variance))
        return max(coef * exp_term, 1e-300)

    def fit(self, featureDF, classifications):
        if isinstance(featureDF, np.ndarray):
            featureDF = pd.DataFrame(featureDF, columns=[f'f_{i}' for i in range(featureDF.shape[1])])
        self.features = list(featureDF.columns)
        class_series = pd.Series(classifications).reset_index(drop=True)
        featureDF = featureDF.reset_index(drop=True)
        self.classes = sorted(list(class_series.unique()))
        total_records = len(class_series)

        self.priors = {}
        self.parameters = {}

        for c in self.classes:
            X_c = featureDF[class_series == c]
            self.priors[c] = len(X_c) / total_records
            self.parameters[c] = {}

            for f in self.features:
                mean_val = float(X_c[f].mean())
                var_val = float(X_c[f].var(ddof=1))
                if np.isnan(var_val) or var_val < self.eps:
                    var_val = self.eps
                self.parameters[c][f] = {
                    'mean': round(mean_val, 4),
                    'variance': round(var_val, 6)
                }

    def predict_single(self, row):
        if isinstance(row, np.ndarray):
            row = pd.Series(row, index=self.features)
        log_probs = {}
        for c in self.classes:
            lp = np.log(max(self.priors[c], self.eps))
            for f in self.features:
                mean = self.parameters[c][f]['mean']
                var = self.parameters[c][f]['variance']
                val = float(row[f])
                likelihood = self.bayesianClassifier(val, mean, var)
                lp += np.log(likelihood)
            log_probs[c] = lp
        return max(log_probs, key=log_probs.get)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.features)
        return [self.predict_single(row) for _, row in X.iterrows()]

    def predict_proba_single(self, row):
        if isinstance(row, np.ndarray):
            row = pd.Series(row, index=self.features)
        log_probs = []
        for c in self.classes:
            lp = np.log(max(self.priors[c], self.eps))
            for f in self.features:
                mean = self.parameters[c][f]['mean']
                var = self.parameters[c][f]['variance']
                val = float(row[f])
                likelihood = self.bayesianClassifier(val, mean, var)
                lp += np.log(likelihood)
            log_probs.append(lp)
        log_probs = np.array(log_probs)
        max_lp = np.max(log_probs)
        probs = np.exp(log_probs - max_lp)
        normalized = probs / np.sum(probs)
        return {int(c): float(normalized[i]) for i, c in enumerate(self.classes)}


# =====================================================================
# EVALUATION METRICS & DATA SPLIT UTILITIES
# =====================================================================

def custom_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    rng = np.random.RandomState(random_state)
    y_list = list(y)
    classes = sorted(list(set(y_list)))

    train_indices = []
    test_indices = []

    if stratify:
        for c in classes:
            c_indices = [i for i, val in enumerate(y_list) if val == c]
            rng.shuffle(c_indices)
            n_test = max(1, int(round(len(c_indices) * test_size)))
            test_indices.extend(c_indices[:n_test])
            train_indices.extend(c_indices[n_test:])
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        indices = list(range(len(y_list)))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(y_list) * test_size)))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    y_train = np.array([y_list[i] for i in train_indices])
    y_test = np.array([y_list[i] for i in test_indices])

    return X_train, X_test, y_train, y_test


def custom_confusion_matrix(y_true, y_pred, labels):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = [[0 for _ in range(k)] for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    return np.array(cm)


def compute_classification_metrics(y_true, y_pred, labels):
    cm = custom_confusion_matrix(y_true, y_pred, labels)
    total_samples = len(y_true)
    correct = sum(cm[i][i] for i in range(len(labels)))
    incorrect = total_samples - correct
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    error_rate = incorrect / total_samples if total_samples > 0 else 0.0

    # Macro & class-specific metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in range(len(labels)):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(labels))) - tp
        fn = sum(cm[i][j] for j in range(len(labels))) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_per_class.append(prec)
        recall_per_class.append(rec)
        f1_per_class.append(f1)

    macro_precision = float(np.mean(precision_per_class))
    macro_recall = float(np.mean(recall_per_class))
    macro_f1 = float(np.mean(f1_per_class))

    return {
        'accuracy': round(accuracy * 100.0, 2),
        'error_rate': round(error_rate * 100.0, 2),
        'precision': round(macro_precision * 100.0, 2),
        'recall': round(macro_recall * 100.0, 2),
        'f1_score': round(macro_f1 * 100.0, 2),
        'confusion_matrix': cm.tolist(),
        'labels': [int(l) for l in labels],
        'total_samples': total_samples,
        'correct': correct,
        'incorrect': incorrect
    }


# =====================================================================
# MATPLOTLIB VISUALIZATION PLOT GENERATORS
# =====================================================================

def plot_hokashyap_convergence(history, save_path):
    """
    Generates two plots required by Assignment 08:
      1. Iteration k vs. ||e(k)||^2 (Squared Error)
      2. Iteration k vs. Number of misclassified training samples
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0f172a')
    iters = history['iteration']
    sq_errs = history['squared_error']
    misclass = history['misclassified']

    # Subplot 1: Iteration vs Squared Error
    ax1.set_facecolor('#1e293b')
    ax1.plot(iters, sq_errs, color='#38bdf8', linewidth=2.5, marker='o', markersize=4, label=r'$\|e(k)\|^2$')
    ax1.set_title(r'Iteration $k$ vs. Squared Error $\|e(k)\|^2$', fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
    ax1.set_xlabel('Iteration $k$', fontsize=11, color='#cbd5e1')
    ax1.set_ylabel(r'Squared Error $\|e(k)\|^2$', fontsize=11, color='#cbd5e1')
    ax1.tick_params(colors='#94a3b8', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3, color='#475569')
    for spine in ax1.spines.values():
        spine.set_color('#334155')

    # Subplot 2: Iteration vs Misclassified Samples
    ax2.set_facecolor('#1e293b')
    ax2.plot(iters, misclass, color='#f43f5e', linewidth=2.5, marker='s', markersize=4, label='Misclassified')
    ax2.set_title(r'Iteration $k$ vs. Misclassified Training Samples ($a^T y_i \leq 0$)', fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
    ax2.set_xlabel('Iteration $k$', fontsize=11, color='#cbd5e1')
    ax2.set_ylabel('Number of Misclassified Samples', fontsize=11, color='#cbd5e1')
    ax2.tick_params(colors='#94a3b8', labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3, color='#475569')
    for spine in ax2.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def plot_eta_vs_iterations(eta_results, optimal_eta, save_path):
    """
    Plots eta vs. Number of iterations and marks the chosen optimal eta.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    etas = [r['eta'] for r in eta_results]
    iters = [r['iterations'] for r in eta_results]

    ax.plot(etas, iters, color='#818cf8', linewidth=2.5, marker='o', markersize=7, label=r'$\eta$ vs. Iterations')

    # Highlight optimal eta
    opt_result = next((r for r in eta_results if abs(r['eta'] - optimal_eta) < 1e-5), None)
    if opt_result:
        ax.plot([opt_result['eta']], [opt_result['iterations']], marker='*', markersize=16,
                color='#f59e0b', markeredgecolor='#ffffff', markeredgewidth=1.5,
                label=rf"Optimal $\eta$ = {optimal_eta:.2f} (Iters: {opt_result['iterations']}, Err: {opt_result['error_rate']}%)")
        ax.annotate(
            rf"Optimal $\eta$ = {optimal_eta:.2f}" + f"\n{opt_result['iterations']} iters, {opt_result['error_rate']}% err",
            xy=(opt_result['eta'], opt_result['iterations']),
            xytext=(opt_result['eta'], opt_result['iterations'] + max(2, max(iters) * 0.1)),
            arrowprops=dict(facecolor='#f59e0b', shrink=0.08, width=2, headwidth=8),
            fontsize=10, fontweight='bold', color='#fde68a',
            bbox=dict(boxstyle="round,pad=0.4", fc="#334155", ec="#f59e0b", lw=1.5)
        )

    ax.set_title(r'Learning Rate $\eta$ vs. Number of Iterations to Convergence', fontsize=13, fontweight='bold', color='#f8fafc', pad=14)
    ax.set_xlabel(r'Learning Rate ($\eta$)', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Iterations Run', fontsize=11, color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, color='#475569')
    ax.legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=10)

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def plot_knn_elbow(k_values, error_rates, selected_elbow_k, save_path):
    """Generates the K vs Error Rate Elbow curve for KNN."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(k_values, error_rates, color='#38bdf8', linewidth=2.5, marker='o', markersize=6, label='Error Rate (1 - Accuracy)')

    if selected_elbow_k in k_values:
        idx = k_values.index(selected_elbow_k)
        err = error_rates[idx]
        ax.plot([selected_elbow_k], [err], marker='*', markersize=16, color='#f59e0b',
                markeredgecolor='#ffffff', markeredgewidth=1.5, label=f'Chosen Elbow K={selected_elbow_k} ({err}%)')
        ax.annotate(
            f"Elbow Point: K={selected_elbow_k}\nError Rate: {err:.2f}%",
            xy=(selected_elbow_k, err),
            xytext=(selected_elbow_k + 2, err + max(1.5, max(error_rates) * 0.08)),
            arrowprops=dict(facecolor='#f59e0b', shrink=0.08, width=2, headwidth=8),
            fontsize=10, fontweight='bold', color='#fde68a',
            bbox=dict(boxstyle="round,pad=0.4", fc="#334155", ec="#f59e0b", lw=1.5)
        )

    ax.set_title('K-Value vs. Classification Error Rate (The Elbow Method)', fontsize=13, fontweight='bold', color='#f8fafc', pad=14)
    ax.set_xlabel('Number of Neighbors (K)', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('Error Rate (%)', fontsize=11, color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, color='#475569')
    ax.legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=10)

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def plot_confusion_matrix_heatmap(cm, display_labels, title, save_path):
    """Draws a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='mako', cbar=False,
        xticklabels=display_labels, yticklabels=display_labels,
        annot_kws={'size': 14, 'weight': 'bold', 'color': '#ffffff'},
        ax=ax, linewidths=1.5, linecolor='#0f172a'
    )

    ax.set_title(title, fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, color='#cbd5e1', labelpad=8)
    ax.set_ylabel('Actual True Label', fontsize=11, color='#cbd5e1', labelpad=8)
    ax.tick_params(colors='#cbd5e1', labelsize=10)

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


def plot_class_distribution_bar(class_counts, display_labels, save_path):
    """Generates target class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    colors = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6'][:len(class_counts)]
    bars = ax.bar(display_labels, class_counts, color=colors, edgecolor='#ffffff', linewidth=1.2, width=0.45)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + max(1, max(class_counts) * 0.02),
                f'{int(h)}', ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=11)

    ax.set_title('Target Class Distribution', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_ylabel('Sample Count', fontsize=10, color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#475569')

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


# =====================================================================
# APPLICATION STATE CONTROLLER
# =====================================================================

class AppState:
    """Manages session data, models, metrics, and workflow state."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_df = None
        self.processed_df = None
        self.filename = None
        self.summary = {}
        self.column_metadata = []
        self.column_unique_map = {}

        self.target_col = None
        self.feature_cols = []
        self.encoder = CustomLabelEncoder()

        # Preprocessing status
        self.has_missing = False
        self.has_duplicates = False
        self.has_categorical = False
        self.requires_preprocessing = False
        self.preprocessing_done = False
        self.use_standardization = False

        # Train/Test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_stats = {}  # for validation: min, max, mean, std

        # Active Classifier selection: 'hokashyap', 'knn', or 'bayes'
        self.active_classifier = 'hokashyap'

        # Ho-Kashyap specific results
        self.hk_default_model = None
        self.hk_default_results = None
        self.hk_eta_results = []
        self.hk_optimal_eta = 0.1
        self.hk_optimal_model = None
        self.hk_active_model = None  # pointer to either default or adopted optimal model
        self.hk_separability_status = None
        self.hk_convergence_plot = None
        self.hk_eta_plot = None
        self.hk_cm_plot = None

        # KNN specific results
        self.knn_max_k = 40
        self.knn_k1_results = None
        self.knn_elbow_curve = []
        self.knn_selected_elbow_k = None
        self.knn_optimal_results = None
        self.knn_model = None
        self.knn_elbow_plot = None
        self.knn_cm_k1_plot = None
        self.knn_cm_optimal_plot = None

        # Bayes specific results
        self.bayes_model = None
        self.bayes_results = None
        self.bayes_cm_plot = None

        # Prediction state
        self.last_prediction = None

    def load_dataset(self, file_path, original_filename):
        self.reset()
        self.filename = original_filename
        df = pd.read_csv(file_path)
        self.raw_df = df.copy()

        rows, cols = df.shape
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        col_meta = []
        unique_map = {}
        has_cat = False

        for col in df.columns:
            series = df[col]
            n_unique = series.nunique(dropna=True)
            non_null = int(series.count())
            null_count = int(series.isnull().sum())
            dtype_str = str(series.dtype)

            # Check if values are numeric
            non_null_samples = series.dropna().head(100)
            all_numeric = True
            for val in non_null_samples:
                if not is_numeric_value(val):
                    all_numeric = False
                    break

            if not all_numeric or 'object' in dtype_str or 'category' in dtype_str:
                col_type = 'Categorical'
                has_cat = True
            elif n_unique == 2:
                col_type = 'Binary'
            elif n_unique < 15:
                col_type = 'Discrete'
            else:
                col_type = 'Continuous Numeric'

            unique_vals = []
            if n_unique <= 20:
                unique_vals = [normalize_class_label(v) for v in sorted(series.dropna().unique(), key=lambda x: str(x))]

            unique_map[col] = {
                'type': col_type,
                'is_numeric': all_numeric,
                'unique_count': n_unique,
                'unique_values': unique_vals
            }

            col_meta.append({
                'name': col,
                'dtype': dtype_str,
                'non_null': non_null,
                'null_count': null_count,
                'unique': n_unique,
                'category': col_type,
                'is_numeric': all_numeric
            })

        self.has_missing = (missing_count > 0)
        self.has_duplicates = (duplicate_count > 0)
        self.has_categorical = has_cat
        self.requires_preprocessing = (self.has_missing or self.has_duplicates or self.has_categorical)

        self.summary = {
            'filename': original_filename,
            'rows': rows,
            'cols': cols,
            'missing_cells': missing_count,
            'duplicate_rows': duplicate_count,
            'has_missing': self.has_missing,
            'has_duplicates': self.has_duplicates,
            'has_categorical': self.has_categorical,
            'requires_preprocessing': self.requires_preprocessing,
            'preview_rows': df.head(5).to_dict(orient='records'),
            'columns': list(df.columns)
        }
        self.column_metadata = col_meta
        self.column_unique_map = unique_map

        # Pre-select potential target column: look for 'target', 'label', 'class', 'survived', etc.
        candidates = ['target', 'target_class', 'label', 'class', 'survived', 'outcome', 'diabetes', 'num']
        selected_target = None
        for cand in candidates:
            for c in df.columns:
                if c.lower() == cand:
                    selected_target = c
                    break
            if selected_target:
                break
        if not selected_target:
            # Default to the last column
            selected_target = df.columns[-1]

        self.target_col = selected_target
        # Default features: all other columns
        self.feature_cols = [c for c in df.columns if c != selected_target]

    def configure_features(self, target_col, selected_features, class_display_names=None):
        if not target_col or target_col not in self.raw_df.columns:
            raise ValueError("Invalid target variable selected.")
        if not selected_features:
            raise ValueError("At least one feature column must be selected.")

        self.target_col = target_col
        self.feature_cols = [c for c in selected_features if c in self.raw_df.columns and c != target_col]

        if not self.feature_cols:
            raise ValueError("No valid features selected for training.")

        # Initialize encoder for target
        self.encoder.fit(self.raw_df[self.target_col].dropna().values)
        if class_display_names:
            self.encoder.set_display_names(class_display_names)

    def update_class_names(self, rename_dict):
        self.encoder.set_display_names(rename_dict)

    def execute_preprocessing(self, remove_duplicates=False, imputation_strategy='median',
                              categorical_strategy='encode', use_standardization=True):
        df = self.raw_df.copy()

        # 1. Remove duplicates if requested
        if remove_duplicates and self.has_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        # 2. Extract target
        y_raw = df[self.target_col].values
        # Encode target
        y = self.encoder.fit_transform(y_raw)

        # 3. Extract and process features
        X_df = df[self.feature_cols].copy()

        # Check for categorical columns among selected features
        cat_cols = []
        num_cols = []
        for col in self.feature_cols:
            col_info = self.column_unique_map.get(col, {})
            if not col_info.get('is_numeric', True):
                cat_cols.append(col)
            else:
                num_cols.append(col)

        # Handle Categorical Columns
        if cat_cols:
            if categorical_strategy == 'exclude':
                # Drop categorical columns from training features
                X_df = X_df[num_cols].copy()
                self.feature_cols = num_cols
                if not self.feature_cols:
                    raise ValueError("All selected features were categorical and excluded. Select at least one numeric feature.")
            else:
                # Encode categorical columns using pandas get_dummies / label encoding
                for c in cat_cols:
                    if X_df[c].isnull().any():
                        mode_val = X_df[c].mode()[0] if not X_df[c].mode().empty else 'Missing'
                        X_df[c] = X_df[c].fillna(mode_val)
                    le = CustomLabelEncoder()
                    X_df[c] = le.fit_transform(X_df[c].astype(str).values)

        # Impute Missing Values in numeric features
        if X_df.isnull().any().any():
            imputer = SimpleImputer(strategy=imputation_strategy)
            imputed_vals = imputer.fit_transform(X_df.values)
            X_df = pd.DataFrame(imputed_vals, columns=X_df.columns)

        # Train/Test Split (80/20 Stratified)
        X_train_raw, X_test_raw, y_train, y_test = custom_train_test_split(
            X_df.values, y, test_size=0.2, random_state=42, stratify=True
        )

        # Store training feature statistics for validation
        self.feature_stats = {}
        for idx, col in enumerate(self.feature_cols):
            col_train = X_train_raw[:, idx]
            self.feature_stats[col] = {
                'min': float(np.min(col_train)),
                'max': float(np.max(col_train)),
                'mean': float(np.mean(col_train)),
                'std': float(np.std(col_train)) if float(np.std(col_train)) > 1e-6 else 1.0
            }

        # Standardization Choice (Scikit-Learn StandardScaler)
        self.use_standardization = use_standardization
        if use_standardization:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(X_train_raw)
            self.X_test = self.scaler.transform(X_test_raw)
        else:
            self.scaler = None
            self.X_train = X_train_raw
            self.X_test = X_test_raw

        self.y_train = y_train
        self.y_test = y_test
        self.preprocessing_done = True

        # Generate target class distribution plot
        plot_path = os.path.join(app.config['PLOT_FOLDER'], 'target_distribution.png')
        unique_classes, counts = np.unique(y, return_counts=True)
        display_labels = [self.encoder.get_display_name(c) for c in unique_classes]
        plot_class_distribution_bar(counts, display_labels, plot_path)

        return {
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features_count': len(self.feature_cols),
            'use_standardization': self.use_standardization
        }

    # =====================================================================
    # TRAINING PROCEDURES: HO-KASHYAP, KNN, BAYES
    # =====================================================================

    def train_hokashyap(self, default_eta=0.1, max_iter=500, tol=1e-5, b_init=1.0, selected_etas=None):
        """
        Executes the Ho-Kashyap procedure:
          1. Trains default model with default_eta and records:
             - Weight vector a, min/max margin b
             - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
             - Convergence plots: Iteration vs ||e||^2, Iteration vs Misclassifications
          2. Runs multi-eta convergence analysis for eta in [0.1, ..., 0.9]
          3. Plots eta vs iterations and selects the optimal eta
        """
        if not self.preprocessing_done:
            raise RuntimeError("Data pre-processing must be completed prior to training.")

        if len(self.encoder.classes_) != 2:
            raise ValueError(f"Ho-Kashyap procedure is a binary linear classifier and requires exactly 2 classes. Current target has {len(self.encoder.classes_)} classes.")

        self.active_classifier = 'hokashyap'

        # 1. Fit Default Ho-Kashyap Classifier
        default_model = CustomHoKashyapClassifier(
            learning_rate=default_eta,
            max_iter=max_iter,
            tol=tol,
            b_init=b_init
        )
        default_model.fit(self.X_train, self.y_train)
        self.hk_default_model = default_model
        self.hk_active_model = default_model

        # Evaluate on Test Set
        y_test_pred = default_model.predict(self.X_test)
        metrics = compute_classification_metrics(self.y_test, y_test_pred, labels=[0, 1])

        weight_summary = default_model.get_weight_summary(self.feature_cols)
        margin_summary = default_model.get_margin_summary()

        self.hk_default_results = {
            'metrics': metrics,
            'weight_summary': weight_summary,
            'margin_summary': margin_summary,
            'iterations_run': default_model.iterations_run_,
            'converged': default_model.converged_,
            'separability': default_model.separability_,
            'stop_reason': default_model.stop_reason_,
            'default_eta': default_eta
        }
        self.hk_separability_status = default_model.separability_

        # Plot 1: Iteration vs ||e(k)||^2 and Iteration vs Misclassified
        conv_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'hk_convergence.png')
        plot_hokashyap_convergence(default_model.history_, conv_plot_path)
        self.hk_convergence_plot = 'plots/hk_convergence.png'

        # Plot 2: Test Confusion Matrix Heatmap
        cm_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'hk_confusion_matrix.png')
        disp_labels = [self.encoder.get_display_name(0), self.encoder.get_display_name(1)]
        plot_confusion_matrix_heatmap(
            np.array(metrics['confusion_matrix']),
            disp_labels,
            f"Ho-Kashyap Test Confusion Matrix (η={default_eta})",
            cm_plot_path
        )
        self.hk_cm_plot = 'plots/hk_confusion_matrix.png'

        # 2. Multi-Eta Analysis (Study convergence for different learning rates from 0.1 to 0.9)
        if selected_etas is None or len(selected_etas) == 0:
            selected_etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            selected_etas = sorted([float(e) for e in selected_etas])

        eta_results = []
        best_candidate = None
        best_score_tuple = (float('inf'), float('inf'), float('inf'))  # (error_rate, iterations, sq_err)

        for eta_val in selected_etas:
            m = CustomHoKashyapClassifier(
                learning_rate=eta_val,
                max_iter=max_iter,
                tol=tol,
                b_init=b_init
            )
            m.fit(self.X_train, self.y_train)
            pred_m = m.predict(self.X_test)
            m_metrics = compute_classification_metrics(self.y_test, pred_m, labels=[0, 1])
            final_sq_err = m.history_['squared_error'][-1] if m.history_['squared_error'] else 0.0
            final_mis = m.history_['misclassified'][-1] if m.history_['misclassified'] else 0

            entry = {
                'eta': eta_val,
                'iterations': m.iterations_run_,
                'converged': m.converged_,
                'separability': m.separability_,
                'squared_error': final_sq_err,
                'training_misclassified': final_mis,
                'accuracy': m_metrics['accuracy'],
                'error_rate': m_metrics['error_rate'],
                'precision': m_metrics['precision'],
                'recall': m_metrics['recall'],
                'f1_score': m_metrics['f1_score'],
                'model': m
            }
            eta_results.append(entry)

            # Selection criterion: Fewest iterations with the lowest error rate
            score_tuple = (entry['error_rate'], entry['iterations'], final_sq_err)
            if score_tuple < best_score_tuple:
                best_score_tuple = score_tuple
                best_candidate = entry

        self.hk_eta_results = eta_results
        self.hk_optimal_eta = best_candidate['eta']
        self.hk_optimal_model = best_candidate['model']

        # Plot 3: Eta vs Number of Iterations
        eta_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'hk_eta_iterations.png')
        plot_eta_vs_iterations(eta_results, self.hk_optimal_eta, eta_plot_path)
        self.hk_eta_plot = 'plots/hk_eta_iterations.png'

        return self.hk_default_results

    def adopt_optimal_eta(self):
        """Switches the active prediction model to the chosen optimal eta model."""
        if not self.hk_optimal_model:
            raise RuntimeError("Optimal eta model not available.")
        self.hk_active_model = self.hk_optimal_model
        return self.hk_optimal_eta

    def train_knn(self, max_k=40, use_standardization=True):
        """Trains KNN with K=1 baseline and computes Elbow curve up to max_k."""
        if not self.preprocessing_done:
            raise RuntimeError("Data pre-processing must be completed prior to training.")

        self.active_classifier = 'knn'
        self.knn_max_k = min(int(max_k), len(self.X_train) - 1)
        self.knn_max_k = max(self.knn_max_k, 5)

        # Baseline K=1
        k1_clf = CustomKNNClassifier(n_neighbors=1)
        k1_clf.fit(self.X_train, self.y_train)
        y_test_pred_k1 = k1_clf.predict(self.X_test)
        self.knn_k1_results = compute_classification_metrics(self.y_test, y_test_pred_k1, labels=self.encoder.classes_)

        # Evaluate K from 1 to max_k
        k_values = list(range(1, self.knn_max_k + 1))
        error_rates = []
        elbow_curve_data = []

        for k in k_values:
            clf = CustomKNNClassifier(n_neighbors=k)
            clf.fit(self.X_train, self.y_train)
            preds = clf.predict(self.X_test)
            m = compute_classification_metrics(self.y_test, preds, labels=self.encoder.classes_)
            err = m['error_rate']
            error_rates.append(err)
            elbow_curve_data.append({
                'k': k,
                'error_rate': err,
                'accuracy': m['accuracy'],
                'f1_score': m['f1_score']
            })

        self.knn_elbow_curve = elbow_curve_data

        # Default optimal K: K with minimum error rate (first minimum)
        min_err = min(error_rates)
        default_elbow_k = k_values[error_rates.index(min_err)]
        self.knn_selected_elbow_k = default_elbow_k

        # Plot Elbow Curve
        elbow_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'knn_elbow_curve.png')
        plot_knn_elbow(k_values, error_rates, default_elbow_k, elbow_plot_path)
        self.knn_elbow_plot = 'plots/knn_elbow_curve.png'

        # Evaluate default elbow K model
        self.evaluate_optimal_knn(default_elbow_k)

        # Plot K=1 confusion matrix
        cm_k1_path = os.path.join(app.config['PLOT_FOLDER'], 'knn_cm_k1.png')
        disp_labels = [self.encoder.get_display_name(c) for c in self.encoder.classes_]
        plot_confusion_matrix_heatmap(
            np.array(self.knn_k1_results['confusion_matrix']),
            disp_labels,
            "KNN Baseline Confusion Matrix (K=1)",
            cm_k1_path
        )
        self.knn_cm_k1_plot = 'plots/knn_cm_k1.png'

    def evaluate_optimal_knn(self, chosen_k):
        self.knn_selected_elbow_k = int(chosen_k)
        clf = CustomKNNClassifier(n_neighbors=self.knn_selected_elbow_k)
        clf.fit(self.X_train, self.y_train)
        self.knn_model = clf

        preds = clf.predict(self.X_test)
        self.knn_optimal_results = compute_classification_metrics(self.y_test, preds, labels=self.encoder.classes_)

        # Re-plot elbow curve with updated highlight
        k_values = [d['k'] for d in self.knn_elbow_curve]
        error_rates = [d['error_rate'] for d in self.knn_elbow_curve]
        elbow_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'knn_elbow_curve.png')
        plot_knn_elbow(k_values, error_rates, self.knn_selected_elbow_k, elbow_plot_path)

        # Plot Optimal K confusion matrix
        cm_opt_path = os.path.join(app.config['PLOT_FOLDER'], 'knn_cm_optimal.png')
        disp_labels = [self.encoder.get_display_name(c) for c in self.encoder.classes_]
        plot_confusion_matrix_heatmap(
            np.array(self.knn_optimal_results['confusion_matrix']),
            disp_labels,
            f"KNN Optimal Confusion Matrix (K={self.knn_selected_elbow_k})",
            cm_opt_path
        )
        self.knn_cm_optimal_plot = 'plots/knn_cm_optimal.png'

    def train_bayes(self):
        """Trains Gaussian Naive Bayes classifier."""
        if not self.preprocessing_done:
            raise RuntimeError("Data pre-processing must be completed prior to training.")

        self.active_classifier = 'bayes'
        bayes = PatternClassifiers()
        bayes.fit(self.X_train, self.y_train)
        self.bayes_model = bayes

        y_test_pred = bayes.predict(self.X_test)
        self.bayes_results = compute_classification_metrics(self.y_test, y_test_pred, labels=self.encoder.classes_)

        cm_bayes_path = os.path.join(app.config['PLOT_FOLDER'], 'bayes_confusion_matrix.png')
        disp_labels = [self.encoder.get_display_name(c) for c in self.encoder.classes_]
        plot_confusion_matrix_heatmap(
            np.array(self.bayes_results['confusion_matrix']),
            disp_labels,
            "Gaussian Naive Bayes Test Confusion Matrix",
            cm_bayes_path
        )
        self.bayes_cm_plot = 'plots/bayes_confusion_matrix.png'

    # =====================================================================
    # PREDICTION OF NEW USER INSTANCES
    # =====================================================================

    def predict_new_instance(self, input_dict):
        """
        Validates input values (non-empty, numeric, out-of-range warnings)
        and predicts class using active classifier (Ho-Kashyap weight vector a, KNN, or Bayes).
        """
        raw_values = []
        validation_warnings = []

        # Validate features
        for col in self.feature_cols:
            val_str = str(input_dict.get(col, '')).strip()
            if val_str == '':
                raise ValueError(f"Feature '{col}' cannot be empty. Please provide a numeric value.")
            if not is_numeric_value(val_str):
                raise ValueError(f"Feature '{col}' must be a valid numeric float. Received '{val_str}'.")

            val_float = float(val_str)
            raw_values.append(val_float)

            # Check range warning (outside training bounds)
            stats = self.feature_stats.get(col)
            if stats:
                col_min = stats['min']
                col_max = stats['max']
                col_range = col_max - col_min
                buffer = max(col_range * 0.25, stats['std'] * 2.0)
                if val_float < (col_min - buffer) or val_float > (col_max + buffer):
                    validation_warnings.append(
                        f"Notice: Input for '{col}' ({val_float}) is outside typical training range [{col_min:.2f}, {col_max:.2f}]."
                    )

        raw_vector = np.array(raw_values, dtype=np.float64).reshape(1, -1)

        # Apply scaling if active
        if self.use_standardization and self.scaler is not None:
            scaled_vector = self.scaler.transform(raw_vector)
        else:
            scaled_vector = raw_vector

        # Classify based on active model
        if self.active_classifier == 'hokashyap':
            model = self.hk_active_model or self.hk_default_model
            if model is None or model.a_ is None:
                raise RuntimeError("Ho-Kashyap classifier is not trained yet.")

            explanation = model.explain_instance(scaled_vector[0], feature_names=self.feature_cols)
            pred_code = explanation['predicted_class']
            pred_label = self.encoder.get_display_name(pred_code)
            disc_score = explanation['discriminant_score']
            confidence = explanation['confidence']

            desc_text = (
                f"Computed linear discriminant score g(x) = {disc_score:.4f}. "
                f"Since g(x) {' > 0' if disc_score > 0 else ' <= 0'}, the query point lies on the "
                f"{'positive' if disc_score > 0 else 'negative'} half-space of the separating hyperplane a^T x_aug = 0, "
                f"classifying it as Class {pred_code} ('{pred_label}')."
            )

            result = {
                'classifier': 'Ho-Kashyap Linear Classifier',
                'predicted_class_code': pred_code,
                'predicted_class_label': pred_label,
                'confidence': confidence,
                'discriminant_score': disc_score,
                'description': desc_text,
                'explanation': explanation,
                'warnings': validation_warnings,
                'raw_inputs': dict(zip(self.feature_cols, raw_values))
            }

        elif self.active_classifier == 'knn':
            model = self.knn_model or CustomKNNClassifier(n_neighbors=1).fit(self.X_train, self.y_train)
            exp = model.explain_instance(scaled_vector[0])
            pred_code = exp['predicted_class']
            pred_label = self.encoder.get_display_name(pred_code)
            confidence = round(exp['probabilities'].get(pred_code, 1.0) * 100.0, 2)

            # Map neighbor classes to display names
            for nb in exp['neighbors']:
                nb['class_label'] = self.encoder.get_display_name(nb['class'])

            desc_text = (
                f"Classified as Class {pred_code} ('{pred_label}') based on majority voting of "
                f"{len(exp['neighbors'])} nearest neighbors with {confidence}% agreement."
            )

            result = {
                'classifier': f"K-Nearest Neighbors (K={model.n_neighbors})",
                'predicted_class_code': pred_code,
                'predicted_class_label': pred_label,
                'confidence': confidence,
                'description': desc_text,
                'knn_neighbors': exp['neighbors'],
                'probabilities': {self.encoder.get_display_name(c): round(p * 100.0, 1) for c, p in exp['probabilities'].items()},
                'warnings': validation_warnings,
                'raw_inputs': dict(zip(self.feature_cols, raw_values))
            }

        else:  # Bayes
            if self.bayes_model is None:
                raise RuntimeError("Bayes classifier is not trained yet.")
            pred_code = self.bayes_model.predict_single(scaled_vector[0])
            pred_label = self.encoder.get_display_name(pred_code)
            probs = self.bayes_model.predict_proba_single(scaled_vector[0])
            confidence = round(probs.get(pred_code, 0.0) * 100.0, 2)

            desc_text = (
                f"Classified as Class {pred_code} ('{pred_label}') with posterior probability "
                f"P(C|x) = {confidence}% under Gaussian class-conditional density assumption."
            )

            result = {
                'classifier': 'Gaussian Naive Bayes',
                'predicted_class_code': pred_code,
                'predicted_class_label': pred_label,
                'confidence': confidence,
                'description': desc_text,
                'probabilities': {self.encoder.get_display_name(c): round(p * 100.0, 2) for c, p in probs.items()},
                'warnings': validation_warnings,
                'raw_inputs': dict(zip(self.feature_cols, raw_values))
            }

        self.last_prediction = result
        return result


# Global app state instance
app_instance = AppState()


# =====================================================================
# FLASK WEB APPLICATION ROUTES
# =====================================================================

@app.route('/', methods=['GET'])
def index():
    sample_datasets = []
    dataset_dir = app.config['DATASET_FOLDER']
    if os.path.exists(dataset_dir):
        for f in sorted(os.listdir(dataset_dir)):
            if f.endswith('.csv'):
                sample_datasets.append(f)

    return render_template(
        'index.html',
        app_state=app_instance,
        sample_datasets=sample_datasets,
        column_unique_map_json=json.dumps(app_instance.column_unique_map)
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if not file.filename.lower().endswith('.csv'):
            flash("Invalid format. Please upload a standard classification CSV dataset.", "error")
            return redirect(url_for('index', _anchor='step-upload'))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            app_instance.load_dataset(save_path, filename)
            flash(f"Dataset '{filename}' successfully ingested ({app_instance.summary['rows']} rows, {app_instance.summary['cols']} columns). Previous session state flushed.", "success")
        except Exception as e:
            flash(f"Failed to ingest dataset: {str(e)}", "error")
            return redirect(url_for('index', _anchor='step-upload'))

    elif 'sample_dataset' in request.form and request.form.get('sample_dataset'):
        sample_name = request.form.get('sample_dataset')
        sample_path = os.path.join(app.config['DATASET_FOLDER'], sample_name)
        if os.path.exists(sample_path):
            try:
                app_instance.load_dataset(sample_path, sample_name)
                flash(f"Benchmark dataset '{sample_name}' loaded successfully ({app_instance.summary['rows']} rows, {app_instance.summary['cols']} columns).", "success")
            except Exception as e:
                flash(f"Error loading benchmark dataset: {str(e)}", "error")
        else:
            flash(f"Benchmark file '{sample_name}' not found.", "error")
    else:
        flash("Please upload a CSV file or select a benchmark dataset.", "error")

    return redirect(url_for('index', _anchor='step-upload'))


@app.route('/prepare', methods=['POST'])
def prepare():
    target_col = request.form.get('target_col')
    selected_features = request.form.getlist('features')
    remove_duplicates = 'remove_duplicates' in request.form
    imputation_strategy = request.form.get('imputation_strategy', 'median')
    categorical_strategy = request.form.get('categorical_strategy', 'encode')
    use_standardization = 'use_standardization' in request.form

    rename_dict = {}
    for key, val in request.form.items():
        if key.startswith('rename_class_'):
            cls_code = key.replace('rename_class_', '')
            if val.strip():
                rename_dict[cls_code] = val.strip()

    if not target_col:
        flash("Please select a target variable for classification.", "error")
        return redirect(url_for('index', _anchor='step-preparation'))

    try:
        app_instance.configure_features(target_col, selected_features, class_display_names=rename_dict)
        res = app_instance.execute_preprocessing(
            remove_duplicates=remove_duplicates,
            imputation_strategy=imputation_strategy,
            categorical_strategy=categorical_strategy,
            use_standardization=use_standardization
        )
        flash(
            f"Pre-processing completed: {res['train_samples']} training samples, "
            f"{res['test_samples']} test samples across {res['features_count']} numeric features. "
            f"Feature Standardization: {'Enabled (StandardScaler)' if res['use_standardization'] else 'Disabled (Raw Unscaled)'}.",
            "success"
        )
    except Exception as e:
        flash(f"Preparation Error: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-preparation'))


@app.route('/rename_classes', methods=['POST'])
def rename_classes():
    rename_dict = {}
    for key, val in request.form.items():
        if key.startswith('rename_class_'):
            cls_code = key.replace('rename_class_', '')
            if val.strip():
                rename_dict[cls_code] = val.strip()

    if rename_dict:
        try:
            app_instance.update_class_names(rename_dict)
            flash("Target class display labels updated successfully across charts, confusion matrices, and predictions.", "success")
        except Exception as e:
            flash(f"Error updating class labels: {str(e)}", "error")
    else:
        flash("No class label changes provided.", "info")

    return redirect(url_for('index', _anchor='step-preparation'))


@app.route('/train', methods=['POST'])
def train():
    classifier_choice = request.form.get('classifier_choice', 'hokashyap')
    use_standardization = 'use_standardization' in request.form

    # Re-apply standardization setting if toggled
    if app_instance.preprocessing_done and app_instance.use_standardization != use_standardization:
        app_instance.execute_preprocessing(use_standardization=use_standardization)

    try:
        if classifier_choice == 'hokashyap':
            default_eta = float(request.form.get('default_eta', 0.1))
            max_iter = int(request.form.get('max_iter', 500))
            tol = float(request.form.get('tol', 1e-5))
            b_init = float(request.form.get('b_init', 1.0))
            
            # Selected etas for multi-learning-rate exploration
            selected_etas_raw = request.form.getlist('selected_etas')
            selected_etas = [float(e) for e in selected_etas_raw] if selected_etas_raw else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            results = app_instance.train_hokashyap(
                default_eta=default_eta,
                max_iter=max_iter,
                tol=tol,
                b_init=b_init,
                selected_etas=selected_etas
            )
            flash(
                f"Ho-Kashyap Procedure Executed! Status: {results['separability']}. "
                f"Ran {results['iterations_run']} iterations (Final ||e||²: {results['margin_summary']['min_b']} <= b <= {results['margin_summary']['max_b']}). "
                f"Test Accuracy: {results['metrics']['accuracy']}%, Error Rate: {results['metrics']['error_rate']}%. "
                f"Optimal η across multi-rate analysis: η*={app_instance.hk_optimal_eta:.2f}.",
                "success"
            )

        elif classifier_choice == 'knn':
            max_k = int(request.form.get('max_k', 40))
            app_instance.train_knn(max_k=max_k, use_standardization=use_standardization)
            flash(
                f"KNN evaluated for K=1 (Baseline Error: {app_instance.knn_k1_results['error_rate']}%) and across K=1 to {app_instance.knn_max_k}. "
                f"Optimal elbow point K={app_instance.knn_selected_elbow_k} evaluated with Error Rate: {app_instance.knn_optimal_results['error_rate']}%.",
                "info"
            )

        else:  # Bayes
            app_instance.train_bayes()
            flash(
                f"Gaussian Naive Bayes Trained: Test Accuracy: {app_instance.bayes_results['accuracy']}%, "
                f"F1-Score: {app_instance.bayes_results['f1_score']}%.",
                "success"
            )
    except Exception as e:
        flash(f"Training failed: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-training'))


@app.route('/select_elbow_k', methods=['POST'])
def select_elbow_k():
    chosen_k = request.form.get('chosen_k')
    if not chosen_k:
        flash("Please select a valid K from the dropdown.", "error")
        return redirect(url_for('index', _anchor='step-training'))

    try:
        app_instance.evaluate_optimal_knn(int(chosen_k))
        flash(
            f"Elbow Point K={app_instance.knn_selected_elbow_k} evaluated! "
            f"Accuracy: {app_instance.knn_optimal_results['accuracy']}%, Error Rate: {app_instance.knn_optimal_results['error_rate']}%.",
            "success"
        )
    except Exception as e:
        flash(f"Error evaluating chosen K: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-training'))


@app.route('/adopt_optimal_eta', methods=['POST'])
def adopt_optimal_eta():
    try:
        opt_eta = app_instance.adopt_optimal_eta()
        flash(f"Optimal learning rate model (η = {opt_eta:.2f}) adopted as the active classifier for inference!", "success")
    except Exception as e:
        flash(f"Error adopting optimal model: {str(e)}", "error")
    return redirect(url_for('index', _anchor='step-training'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for col in app_instance.feature_cols:
            input_data[col] = request.form.get(f'feat_{col}', '')

        pred = app_instance.predict_new_instance(input_data)
        flash(
            f"Prediction successful ({pred['classifier']}): Predicted Class {pred['predicted_class_code']} "
            f"('{pred['predicted_class_label']}') with {pred['confidence']}% confidence.",
            "success"
        )
    except Exception as e:
        flash(f"Prediction Error: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-prediction'))


@app.route('/reset', methods=['POST'])
def reset_app():
    app_instance.reset()
    flash("Session reset. Previous dataset, models, and evaluation plots cleared.", "info")
    return redirect(url_for('index'))


@app.route('/performance-analysis')
def performance_analysis():
    """
    Dedicated Multi-Dataset & Multi-Model Comparative Analysis Studio.
    Benchmarking:
      1. Ho-Kashyap (Raw Unscaled)
      2. Ho-Kashyap (Standardized)
      3. KNN (K=1, Standardized)
      4. KNN (Optimal Elbow K, Standardized)
      5. Bayes Classifier (Gaussian Naive Bayes)
    Across available benchmark datasets.
    """
    dataset_dir = app.config['DATASET_FOLDER']
    benchmark_files = [
        'linearly_separable_data.csv',
        'heart.csv',
        'Titanic-Dataset.csv',
        'breast_cancer.csv',
        'Naive-Bayes-Classification-Data.csv'
    ]

    analysis_results = []

    for fname in benchmark_files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            continue

        try:
            df = pd.read_csv(fpath)
            # Determine target column
            target_candidates = ['target_class', 'target', 'label', 'class', 'survived', 'diabetes']
            t_col = None
            for tc in target_candidates:
                for c in df.columns:
                    if c.lower() == tc:
                        t_col = c
                        break
                if t_col:
                    break
            if not t_col:
                t_col = df.columns[-1]

            # Preprocess features (numeric only)
            feat_cols = [c for c in df.columns if c != t_col]
            num_feats = []
            for col in feat_cols:
                if is_numeric_value(df[col].dropna().iloc[0]) if not df[col].dropna().empty else False:
                    num_feats.append(col)

            if not num_feats:
                continue

            # Target encoding
            encoder = CustomLabelEncoder()
            y = encoder.fit_transform(df[t_col].dropna().values)
            if len(encoder.classes_) != 2:
                continue

            X_df = df[num_feats].copy()
            if X_df.isnull().any().any():
                imputer = SimpleImputer(strategy='median')
                X_df = pd.DataFrame(imputer.fit_transform(X_df.values), columns=num_feats)

            # Train/Test Split
            X_tr_raw, X_te_raw, y_tr, y_te = custom_train_test_split(
                X_df.values[:len(y)], y, test_size=0.2, random_state=42, stratify=True
            )

            # Scaled versions
            scaler = StandardScaler()
            X_tr_scl = scaler.fit_transform(X_tr_raw)
            X_te_scl = scaler.transform(X_te_raw)

            # 1. Ho-Kashyap (Raw Unscaled)
            hk_raw = CustomHoKashyapClassifier(learning_rate=0.1, max_iter=300)
            hk_raw.fit(X_tr_raw, y_tr)
            hk_raw_preds = hk_raw.predict(X_te_raw)
            m_hk_raw = compute_classification_metrics(y_te, hk_raw_preds, labels=[0, 1])

            # 2. Ho-Kashyap (Standardized)
            hk_scl = CustomHoKashyapClassifier(learning_rate=0.1, max_iter=300)
            hk_scl.fit(X_tr_scl, y_tr)
            hk_scl_preds = hk_scl.predict(X_te_scl)
            m_hk_scl = compute_classification_metrics(y_te, hk_scl_preds, labels=[0, 1])

            # 3. KNN (K=1, Standardized)
            knn_k1 = CustomKNNClassifier(n_neighbors=1)
            knn_k1.fit(X_tr_scl, y_tr)
            knn_k1_preds = knn_k1.predict(X_te_scl)
            m_knn_k1 = compute_classification_metrics(y_te, knn_k1_preds, labels=[0, 1])

            # 4. KNN (Optimal Elbow K, Standardized)
            best_k = 1
            best_knn_err = m_knn_k1['error_rate']
            best_m_knn = m_knn_k1
            for k_test in range(2, min(25, len(X_tr_scl))):
                knn_k = CustomKNNClassifier(n_neighbors=k_test)
                knn_k.fit(X_tr_scl, y_tr)
                p = knn_k.predict(X_te_scl)
                m = compute_classification_metrics(y_te, p, labels=[0, 1])
                if m['error_rate'] < best_knn_err:
                    best_knn_err = m['error_rate']
                    best_k = k_test
                    best_m_knn = m

            # 5. Bayes Classifier (Naive Bayes)
            bayes = PatternClassifiers()
            bayes.fit(X_tr_scl, y_tr)
            bayes_preds = bayes.predict(X_te_scl)
            m_bayes = compute_classification_metrics(y_te, bayes_preds, labels=[0, 1])

            analysis_results.append({
                'dataset_name': fname,
                'samples': len(df),
                'features': len(num_feats),
                'hk_separability': hk_scl.separability_,
                'hk_raw': {
                    'acc': m_hk_raw['accuracy'],
                    'err': m_hk_raw['error_rate'],
                    'f1': m_hk_raw['f1_score'],
                    'iters': hk_raw.iterations_run_
                },
                'hk_scl': {
                    'acc': m_hk_scl['accuracy'],
                    'err': m_hk_scl['error_rate'],
                    'f1': m_hk_scl['f1_score'],
                    'iters': hk_scl.iterations_run_
                },
                'knn_k1': {
                    'acc': m_knn_k1['accuracy'],
                    'err': m_knn_k1['error_rate'],
                    'f1': m_knn_k1['f1_score']
                },
                'knn_opt': {
                    'k': best_k,
                    'acc': best_m_knn['accuracy'],
                    'err': best_m_knn['error_rate'],
                    'f1': best_m_knn['f1_score']
                },
                'bayes': {
                    'acc': m_bayes['accuracy'],
                    'err': m_bayes['error_rate'],
                    'f1': m_bayes['f1_score']
                }
            })

        except Exception as e:
            continue

    return render_template('performance_analysis.html', benchmark_results=analysis_results)


@app.route('/shutdown', methods=['POST'])
def shutdown():
    def kill_server():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=kill_server).start()
    return "Classifiers Studio is safely shutting down. You may close this tab."


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"=== Starting Classifiers Studio (Assignment 08) on http://localhost:{port} ===")
    app.run(host='0.0.0.0', port=port, debug=False)
