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
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# =====================================================================
# FLASK CONFIGURATION & DIRECTORY SETUP
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'Templates'),
    static_folder=os.path.join(BASE_DIR, 'Statics')
)
app.secret_key = 'pattern_recognition_lab_assignment_09_parzen_window_secret'
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
        self.reverse_mapping = {}
        self.display_names = {}

    def fit(self, y):
        raw_vals = [normalize_class_label(v) for v in y]
        unique_vals = sorted(list(set(raw_vals)))
        self.classes_ = unique_vals
        self.mapping = {val: idx for idx, val in enumerate(unique_vals)}
        self.reverse_mapping = {idx: val for idx, val in enumerate(unique_vals)}
        if not self.display_names:
            self.display_names = {idx: str(val) for idx, val in enumerate(unique_vals)}
        return self

    def set_display_name(self, code, name):
        self.display_names[int(code)] = str(name).strip()

    def get_display_name(self, code):
        return self.display_names.get(int(code), str(self.reverse_mapping.get(int(code), code)))

    def transform(self, y):
        raw_vals = [normalize_class_label(v) for v in y]
        return np.array([self.mapping[val] for val in raw_vals], dtype=np.int64)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_codes):
        return [self.reverse_mapping[int(c)] for c in y_codes]

    def get_classes_info(self):
        return [
            {
                'code': idx,
                'raw_label': self.reverse_mapping[idx],
                'display_name': self.display_names.get(idx, str(self.reverse_mapping[idx]))
            }
            for idx in range(len(self.classes_))
        ]


# =====================================================================
# CUSTOM PARZEN WINDOW CLASSIFIER (IMPLEMENTED FROM SCRATCH USING NUMPY)
# =====================================================================

class CustomParzenClassifier:
    """
    Non-parametric Parzen Window density estimation classifier implemented
    strictly from scratch using NumPy.

    Mathematical Formulation:
      - Class-conditional probability density estimate:
          p(x | w_j) = (1 / (N_j * V_n)) * sum_{i=1}^{N_j} phi((x - x_{ji}) / h)
        where:
          - h is the window width parameter.
          - V_n = h^d is the hypercube volume in d-dimensional space.
          - phi(u) is the window function:
              * Hypercube: phi(u) = 1 if max_k |u_k| <= 0.5, else 0
              * Gaussian: phi(u) = (1 / (2*pi)^(d/2)) * exp(-0.5 * ||u||^2)

      - Log-Sum-Exp Trick for Gaussian Window:
          With small h or large d, Gaussian densities may underflow standard IEEE-754
          floating point. To prevent underflow, we evaluate in log-space:
            ln p(x | w_j) = -ln(N_j) - (d/2)*ln(2*pi) - d*ln(h) + LSE_{i}(-||x - x_{ji}||^2 / (2*h^2))
          where LSE(z) = max(z) + ln sum(exp(z - max(z))).

      - Bayes Decision Rule:
          Combine class-conditional densities with class prior probabilities P(w_j):
            P(w_j | x) = p(x | w_j) * P(w_j) / sum_c (p(x | w_c) * P(w_c))
          Predicted class:
            y_hat = argmax_j [ ln p(x | w_j) + ln P(w_j) ]

      - Hypercube Empty Window Handling:
          Tracks samples where no training samples of any class fall inside the window
          (sum_j k_j = 0). For empty-window queries, falls back gracefully to prior probabilities.
    """
    def __init__(self, window='gaussian', h=1.0):
        self.window = window.lower()
        if self.window not in ('gaussian', 'hypercube'):
            raise ValueError(f"Unsupported window function '{self.window}'. Choose 'gaussian' or 'hypercube'.")
        self.h = max(1e-6, float(h))
        self.classes_ = None
        self.d_ = None
        self.priors_ = {}
        self.X_by_class_ = {}
        self.N_by_class_ = {}
        self.total_train_samples_ = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.d_ = X.shape[1]
        self.total_train_samples_ = len(y)

        self.priors_ = {}
        self.X_by_class_ = {}
        self.N_by_class_ = {}

        for c in self.classes_:
            Xc = X[y == c]
            self.X_by_class_[c] = Xc
            self.N_by_class_[c] = len(Xc)
            self.priors_[c] = len(Xc) / float(self.total_train_samples_)

        return self

    def _compute_gaussian(self, X):
        """
        Computes Gaussian window class-conditional log-densities and densities
        using the numerically stable log-sum-exp trick with batched distance evaluation.
        """
        M = X.shape[0]
        C = len(self.classes_)
        log_densities = np.zeros((M, C), dtype=np.float64)
        densities = np.zeros((M, C), dtype=np.float64)

        # Precompute squared norms of query points: shape (M, 1)
        x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
        batch_size = 200

        for idx, c in enumerate(self.classes_):
            Xc = self.X_by_class_[c]
            # Cap reference points to 2500 per class for memory safety & responsive evaluation
            if len(Xc) > 2500:
                rng = np.random.RandomState(42)
                sub_idx = rng.choice(len(Xc), 2500, replace=False)
                Xc_ref = Xc[sub_idx]
            else:
                Xc_ref = Xc
            Nc_ref = len(Xc_ref)

            xc_norm_sq = np.sum(Xc_ref ** 2, axis=1, keepdims=True).T
            const_term = -np.log(Nc_ref) - (self.d_ / 2.0) * np.log(2.0 * np.pi) - self.d_ * np.log(self.h)

            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)
                X_batch = X[start:end]
                x_norm_sq_batch = x_norm_sq[start:end]

                # BLAS-accelerated pairwise squared Euclidean distances: shape (batch_len, Nc_ref)
                dist_sq = np.maximum(0.0, x_norm_sq_batch - 2.0 * (X_batch @ Xc_ref.T) + xc_norm_sq)

                # Exponents: z_i = - ||x - x_i||^2 / (2 * h^2)
                Z = -dist_sq / (2.0 * (self.h ** 2))

                # Log-sum-exp along axis 1 (across training samples in class c)
                max_Z = np.max(Z, axis=1, keepdims=True)
                lse = max_Z.squeeze(axis=1) + np.log(np.sum(np.exp(Z - max_Z), axis=1))

                # ln p(x | w_j) = -ln(N_j) - (d/2)*ln(2*pi) - d*ln(h) + lse
                lp = const_term + lse
                log_densities[start:end, idx] = lp

                # Safe density conversion for presentation (clipped to prevent overflow/underflow warnings)
                densities[start:end, idx] = np.exp(np.clip(lp, -700.0, 700.0))

        # Bayes posterior probabilities computed strictly in log space
        log_priors = np.array([np.log(self.priors_[c]) for c in self.classes_], dtype=np.float64)
        log_joint = log_densities + log_priors[np.newaxis, :]  # shape: (M, C)

        max_lj = np.max(log_joint, axis=1, keepdims=True)
        exp_joint = np.exp(log_joint - max_lj)
        posteriors = exp_joint / np.sum(exp_joint, axis=1, keepdims=True)

        preds = self.classes_[np.argmax(log_joint, axis=1)]
        empty_windows = np.zeros(M, dtype=bool)

        return preds, posteriors, densities, log_densities, empty_windows

    def _compute_hypercube(self, X):
        """
        Computes Hypercube window class-conditional densities and counts:
          phi(u) = 1 if max_k |u_k| <= 0.5, else 0
          Condition: max_k |x_k - x_{ji, k}| <= h / 2.
        Tracks empty windows (sum_c k_c == 0).
        """
        M = X.shape[0]
        C = len(self.classes_)
        densities = np.zeros((M, C), dtype=np.float64)
        log_densities = np.full((M, C), -np.inf, dtype=np.float64)
        counts = np.zeros((M, C), dtype=np.int64)

        vol = self.h ** self.d_
        half_h = self.h / 2.0
        batch_size = 200

        for idx, c in enumerate(self.classes_):
            Xc = self.X_by_class_[c]
            # Cap reference points to 2500 per class for memory safety & responsive evaluation
            if len(Xc) > 2500:
                rng = np.random.RandomState(42)
                sub_idx = rng.choice(len(Xc), 2500, replace=False)
                Xc_ref = Xc[sub_idx]
            else:
                Xc_ref = Xc
            Nc_ref = len(Xc_ref)

            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)
                X_batch = X[start:end]
                # Chebyshev (L-infinity) distance check via 2D boolean reduction: avoids large 3D array allocations
                inside = np.ones((end - start, Nc_ref), dtype=bool)
                for k in range(self.d_):
                    inside &= (np.abs(X_batch[:, k:k+1] - Xc_ref[:, k:k+1].T) <= half_h)
                cnt = np.sum(inside, axis=1)
                counts[start:end, idx] = cnt

            densities[:, idx] = counts[:, idx] / (Nc_ref * vol)
            non_zero = counts[:, idx] > 0
            log_densities[non_zero, idx] = (
                np.log(counts[non_zero, idx]) - np.log(Nc_ref) - self.d_ * np.log(self.h)
            )

        total_counts = np.sum(counts, axis=1)  # shape: (M,)
        empty_windows = (total_counts == 0)

        posteriors = np.zeros((M, C), dtype=np.float64)
        preds = np.empty(M, dtype=self.classes_.dtype)
        priors_arr = np.array([self.priors_[c] for c in self.classes_], dtype=np.float64)

        for m in range(M):
            if empty_windows[m]:
                # Fallback to class priors when no training samples fall inside the hypercube
                posteriors[m] = priors_arr
                preds[m] = self.classes_[np.argmax(priors_arr)]
            else:
                posteriors[m] = counts[m] / float(total_counts[m])
                # Bayes classification: argmax count (with prior weighting tie-breaking)
                weighted_counts = counts[m].astype(np.float64) * priors_arr
                preds[m] = self.classes_[np.argmax(weighted_counts)]

        return preds, posteriors, densities, log_densities, empty_windows

    def predict_details(self, X):
        """Returns predictions, posteriors, densities, log-densities, and empty-window flags."""
        X = np.asarray(X, dtype=np.float64)
        if self.window == 'gaussian':
            return self._compute_gaussian(X)
        return self._compute_hypercube(X)

    def predict(self, X):
        preds, _, _, _, _ = self.predict_details(X)
        return preds

    def predict_proba(self, X):
        _, posteriors, _, _, _ = self.predict_details(X)
        return posteriors

    def explain_instance(self, x):
        """Provides an in-depth breakdown for a single input instance."""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        preds, posteriors, densities, log_densities, empty_windows = self.predict_details(x)

        pred_code = int(preds[0])
        class_breakdown = []

        for idx, c in enumerate(self.classes_):
            c_int = int(c)
            p_dens = float(densities[0, idx])
            lp_dens = float(log_densities[0, idx])
            post = float(posteriors[0, idx])
            prior = float(self.priors_[c])

            class_breakdown.append({
                'class_code': c_int,
                'prior': round(prior, 4),
                'density': p_dens,
                'density_str': f"{p_dens:.4e}" if (0 < p_dens < 1e-4 or p_dens > 1e4) else f"{p_dens:.6f}",
                'log_density': round(lp_dens, 4) if np.isfinite(lp_dens) else "-inf",
                'posterior': round(post, 4),
                'posterior_percent': round(post * 100.0, 2)
            })

        return {
            'predicted_class': pred_code,
            'window': self.window,
            'h': self.h,
            'is_empty_window': bool(empty_windows[0]),
            'classes': class_breakdown
        }


# =====================================================================
# CUSTOM HO-KASHYAP LINEAR CLASSIFIER (INTEGRATED FROM ASSIGNMENT 08)
# =====================================================================

class CustomHoKashyapClassifier:
    """
    Ho-Kashyap Linear Classification Procedure implemented from scratch using NumPy.
    Finds weight vector a = [a_0, a_1, ..., a_d]^T and margin vector b > 0
    such that Y a = b > 0 in augmented normalized space.
    """
    def __init__(self, learning_rate=0.1, max_iter=500, tol=1e-5, b_init=1.0):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.b_init = float(b_init)
        self.a_ = None
        self.b_ = None
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
            raise ValueError(f"Ho-Kashyap procedure requires a binary classification problem (exactly 2 classes). Found {len(classes)} classes.")

        self.classes_ = classes
        z = np.where(y == self.classes_[1], 1.0, -1.0)[:, None]

        N, d = X.shape
        X_aug = np.hstack([np.ones((N, 1), dtype=np.float64), X])
        Y = z * X_aug

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

            if sq_err <= self.tol or (misclassified == 0 and sq_err < 1e-2):
                self.converged_ = True
                self.separability_ = 'Linearly Separable'
                self.stop_reason_ = f"Converged at iteration {k} with squared error {sq_err:.6e} and 0 misclassifications."
                break

            if np.all(e <= 1e-8) and sq_err > self.tol:
                self.converged_ = False
                self.separability_ = 'Linearly Non-Separable'
                self.stop_reason_ = f"Linear non-separability condition e(k) <= 0 satisfied at iteration {k}."
                break

            if abs(prev_err - sq_err) < 1e-7:
                stagnant_count += 1
                if stagnant_count >= 5:
                    self.converged_ = False
                    self.separability_ = 'Linearly Non-Separable (Plateau Reached)'
                    self.stop_reason_ = f"Error plateau reached at iteration {k}."
                    break
            else:
                stagnant_count = 0

            prev_err = sq_err
            e_plus = np.maximum(0.0, e)
            b = b + 2.0 * self.learning_rate * e_plus
            a = Y_pinv @ b

        if not self.converged_ and self.separability_ == 'Unknown':
            if misclassified == 0:
                self.converged_ = True
                self.separability_ = 'Linearly Separable'
                self.stop_reason_ = "Separating hyperplane found (0 training misclassifications)."
            else:
                self.separability_ = 'Linearly Non-Separable / Inconclusive'
                self.stop_reason_ = f"Maximum iterations ({self.max_iter}) reached without zero error."

        self.a_ = a
        self.b_ = b
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        X_aug = np.hstack([np.ones((N, 1), dtype=np.float64), X])
        scores = X_aug @ self.a_
        preds = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        return preds

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        X_aug = np.hstack([np.ones((N, 1), dtype=np.float64), X])
        return X_aug @ self.a_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs_class_1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
        probs_class_0 = 1.0 - probs_class_1
        return np.column_stack([probs_class_0, probs_class_1])


# =====================================================================
# CUSTOM KNN CLASSIFIER (INTEGRATED FROM ASSIGNMENTS 07/08)
# =====================================================================

class CustomKNNClassifier:
    """K-Nearest Neighbors Classifier implemented from first principles."""
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

        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = counts.max()
        candidate_labels = unique_labels[counts == max_count]

        if len(candidate_labels) > 1:
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
        return np.array([self._predict_single(x)[0] for x in X])

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
# CUSTOM BAYES CLASSIFIER (INTEGRATED FROM ASSIGNMENTS 06/07/08)
# =====================================================================

class PatternClassifiers:
    """Gaussian Naive Bayes Classifier implemented from scratch."""
    def __init__(self, eps=1e-9):
        self.classes = []
        self.features = []
        self.priors = {}
        self.parameters = {}
        self.eps = eps

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

def custom_train_val_test_split(X, y, test_size=0.2, val_size=0.2, random_state=42, stratify=True):
    """
    Performs a stratified 3-way split into Training (1 - test_size - val_size),
    Validation (val_size), and Testing (test_size) sets.
    """
    rng = np.random.RandomState(random_state)
    y_list = list(y)
    classes = sorted(list(set(y_list)))

    train_indices = []
    val_indices = []
    test_indices = []

    if stratify:
        for c in classes:
            c_indices = [i for i, val in enumerate(y_list) if val == c]
            rng.shuffle(c_indices)
            n_total = len(c_indices)
            if n_total <= 3:
                n_test = 1 if n_total >= 3 else 0
                n_val = 1 if n_total >= 2 else 0
            else:
                n_test = max(1, int(round(n_total * test_size)))
                n_val = max(1, int(round(n_total * val_size)))
                if n_test + n_val >= n_total:
                    n_test = max(1, (n_total - 1) // 2)
                    n_val = max(0, n_total - 1 - n_test)

            test_idx = c_indices[:n_test]
            val_idx = c_indices[n_test:n_test + n_val]
            train_idx = c_indices[n_test + n_val:]

            test_indices.extend(test_idx)
            val_indices.extend(val_idx)
            train_indices.extend(train_idx)
    else:
        all_indices = list(range(len(y)))
        rng.shuffle(all_indices)
        n_total = len(all_indices)
        n_test = int(round(n_total * test_size))
        n_val = int(round(n_total * val_size))

        test_indices = all_indices[:n_test]
        val_indices = all_indices[n_test:n_test + n_val]
        train_indices = all_indices[n_test + n_val:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices], X[val_indices], X[test_indices],
        np.array(y)[train_indices], np.array(y)[val_indices], np.array(y)[test_indices]
    )


def custom_confusion_matrix(y_true, y_pred, labels):
    """Computes confusion matrix where cm[i, j] is count of true class i predicted as class j."""
    n_labels = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def compute_classification_metrics(y_true, y_pred, labels):
    """
    Computes Accuracy, Precision, Recall, F1-Score (macro-averaged for multi-class),
    Error Rate, and per-class breakdown.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_samples = len(y_true)

    cm = custom_confusion_matrix(y_true, y_pred, labels)
    correct_predictions = int(np.trace(cm))
    incorrect_predictions = int(total_samples - correct_predictions)

    accuracy = float(correct_predictions / total_samples) if total_samples > 0 else 0.0
    error_rate = float(incorrect_predictions / total_samples) if total_samples > 0 else 0.0

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for i, lbl in enumerate(labels):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        tn = float(total_samples - (tp + fp + fn))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class_metrics[lbl] = {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'support': int(tp + fn)
        }

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions,
        'accuracy': round(accuracy * 100.0, 2),
        'error_rate': round(error_rate * 100.0, 2),
        'precision': round(macro_precision * 100.0, 2),
        'recall': round(macro_recall * 100.0, 2),
        'f1_score': round(macro_f1 * 100.0, 2),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class_metrics
    }


# =====================================================================
# PLOTTING ROUTINES (HIGH-RESOLUTION VISUALIZATIONS)
# =====================================================================

def plot_parzen_window_analysis(h_values, results_gaussian, results_hypercube, optimal_h, optimal_window, save_path):
    """
    Plots h vs Error Rate comparing Gaussian and Hypercube window functions
    (training and validation error rates of both on the SAME graph with logarithmic scale for h).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    h_arr = np.array(h_values, dtype=np.float64)

    g_train = [r['train_error_rate'] for r in results_gaussian]
    g_val = [r['val_error_rate'] for r in results_gaussian]
    h_train = [r['train_error_rate'] for r in results_hypercube]
    h_val = [r['val_error_rate'] for r in results_hypercube]

    # Gaussian Curves (Cyan/Blue palette)
    ax.plot(h_arr, g_train, marker='o', linewidth=2.4, markersize=7, color='#38bdf8', label='Gaussian (Train Error Rate)')
    ax.plot(h_arr, g_val, marker='s', linewidth=2.4, markersize=7, linestyle='--', color='#818cf8', label='Gaussian (Validation Error Rate)')

    # Hypercube Curves (Orange/Rose palette)
    ax.plot(h_arr, h_train, marker='^', linewidth=2.4, markersize=7, color='#fb923c', label='Hypercube (Train Error Rate)')
    ax.plot(h_arr, h_val, marker='D', linewidth=2.4, markersize=7, linestyle='--', color='#f43f5e', label='Hypercube (Validation Error Rate)')

    # Mark optimal point
    if optimal_window == 'gaussian':
        opt_val = [r['val_error_rate'] for r in results_gaussian if abs(r['h'] - optimal_h) < 1e-6][0]
        opt_color = '#818cf8'
    else:
        opt_val = [r['val_error_rate'] for r in results_hypercube if abs(r['h'] - optimal_h) < 1e-6][0]
        opt_color = '#f43f5e'

    ax.scatter([optimal_h], [opt_val], color='#22c55e', s=220, zorder=6, edgecolors='#ffffff', linewidth=2.2, label=f'Optimal (h*={optimal_h}, {optimal_window.capitalize()})')
    ax.annotate(
        f"Optimal: h*={optimal_h}\nVal Err={opt_val:.2f}%",
        xy=(optimal_h, opt_val),
        xytext=(optimal_h * 1.3, opt_val + (3.0 if opt_val < 50 else -7.0)),
        arrowprops=dict(facecolor='#22c55e', shrink=0.08, width=1.5, headwidth=6),
        color='#f8fafc',
        fontweight='bold',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f172a", ec="#22c55e", lw=1.5)
    )

    ax.set_xscale('log')
    ax.set_title('Window Width Analysis: h vs. Classification Error Rate', fontsize=14, fontweight='bold', color='#f8fafc', pad=14)
    ax.set_xlabel('Window Width (h) [Logarithmic Scale]', fontsize=11, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel('Classification Error Rate (%)', fontsize=11, fontweight='bold', color='#cbd5e1')

    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.35, color='#64748b', which='both')
    for spine in ax.spines.values():
        spine.set_color('#334155')

    legend = ax.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155', fontsize=9.5, loc='best')
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_heatmap(cm, display_labels, title, save_path):
    """Renders high-resolution confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=display_labels, yticklabels=display_labels,
        cbar=True, ax=ax, linewidths=1.2, linecolor='#0f172a',
        annot_kws={"size": 12, "weight": "bold", "color": "#0f172a"}
    )

    ax.set_title(title, fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel('True Ground Truth Label', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=9)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#94a3b8')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def plot_class_distribution_bar(class_counts, display_labels, save_path):
    """Generates bar chart of class distribution in dataset."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.bar(display_labels, class_counts, color='#6366f1', edgecolor='#818cf8', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(class_counts) * 0.02),
                f'{int(yval)}', ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=9.5)

    ax.set_title('Dataset Class Distribution', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_ylabel('Sample Count', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=9.5)
    ax.grid(axis='y', linestyle=':', alpha=0.35, color='#64748b')
    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def plot_hokashyap_convergence(history, save_path):
    """Plots iteration vs squared error and iteration vs misclassified samples for Ho-Kashyap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    fig.patch.set_facecolor('#0f172a')

    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)
        ax.grid(True, linestyle=':', alpha=0.35, color='#64748b')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    iters = history['iteration']
    ax1.plot(iters, history['squared_error'], color='#6366f1', linewidth=2.2)
    ax1.set_title('Iteration k vs. Squared Error ||e(k)||^2', fontsize=11, fontweight='bold', color='#f8fafc')
    ax1.set_xlabel('Iteration (k)', fontsize=10, color='#cbd5e1')
    ax1.set_ylabel('Squared Error ||e(k)||^2', fontsize=10, color='#cbd5e1')

    ax2.plot(iters, history['misclassified'], color='#f59e0b', linewidth=2.2)
    ax2.set_title('Iteration k vs. Misclassified Samples', fontsize=11, fontweight='bold', color='#f8fafc')
    ax2.set_xlabel('Iteration (k)', fontsize=10, color='#cbd5e1')
    ax2.set_ylabel('Misclassified Count (a^T y <= 0)', fontsize=10, color='#cbd5e1')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def plot_eta_vs_iterations(eta_results, optimal_eta, save_path):
    """Plots eta vs number of iterations for Ho-Kashyap."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    etas = [r['eta'] for r in eta_results]
    iters = [r['iterations'] for r in eta_results]

    ax.plot(etas, iters, marker='o', linewidth=2.4, color='#10b981', markersize=8, label='Iterations Run')
    for r in eta_results:
        ax.annotate(f"{r['iterations']}", (r['eta'], r['iterations']),
                    textcoords="offset points", xytext=(0, 8), ha='center',
                    color='#f8fafc', fontweight='bold', fontsize=9)

    ax.axvline(x=optimal_eta, color='#f59e0b', linestyle='--', linewidth=1.8, label=f'Optimal η* = {optimal_eta:.2f}')
    ax.set_title('Learning Rate (η) vs. Iterations to Convergence', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_xlabel('Learning Rate (η)', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel('Total Iterations', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.35, color='#64748b')

    legend = ax.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155', fontsize=9)
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def plot_knn_elbow(k_values, error_rates, selected_elbow_k, save_path):
    """Plots K vs Error Rate for KNN Elbow analysis."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(k_values, error_rates, marker='o', linewidth=2.2, color='#38bdf8', markersize=5.5, label='Validation Error Rate')
    if selected_elbow_k in k_values:
        idx = k_values.index(selected_elbow_k)
        ax.scatter([selected_elbow_k], [error_rates[idx]], color='#ef4444', s=160, zorder=5, label=f'Optimal Elbow K={selected_elbow_k}')

    ax.set_title('KNN Elbow Analysis: Value of K vs. Error Rate', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_xlabel('Number of Neighbors (K)', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel('Classification Error Rate (%)', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.tick_params(colors='#94a3b8', labelsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.35, color='#64748b')

    legend = ax.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155', fontsize=9)
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def plot_comparative_benchmark(results_list, save_path):
    """Plots comparative bar chart of Accuracy and F1-Score across all integrated classifiers."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    names = [r['name'] for r in results_list]
    accs = [r['accuracy'] for r in results_list]
    f1s = [r['f1_score'] for r in results_list]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#6366f1', edgecolor='#818cf8')
    rects2 = ax.bar(x + width/2, f1s, width, label='F1-Score (%)', color='#10b981', edgecolor='#34d399')

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#f8fafc')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#f8fafc')

    ax.set_title('Unified Classifiers Benchmark Comparison', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel('Score (%)', fontsize=10, fontweight='bold', color='#cbd5e1')
    ax.set_ylim(0, 115)
    ax.tick_params(colors='#94a3b8', labelsize=9.5)
    ax.grid(axis='y', linestyle=':', alpha=0.35, color='#64748b')

    legend = ax.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155', fontsize=9)
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


# =====================================================================
# APPLICATION STATE MANAGER (SINGLETON STATEFUL DATA STORE)
# =====================================================================

class AppState:
    """Manages application state, dataset lifecycle, models, metrics, and caching."""
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

        # Preprocessing status & flags
        self.has_missing = False
        self.has_duplicates = False
        self.has_categorical = False
        self.requires_preprocessing = False
        self.preprocessing_done = False
        self.use_standardization = False

        # Train / Val / Test splits (60% Train, 20% Val, 20% Test)
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.feature_stats = {}

        # Active Classifier selection: 'parzen', 'hokashyap', 'knn', or 'bayes'
        self.active_classifier = 'parzen'

        # Parzen Window results
        self.parzen_default_model = None
        self.parzen_default_results = None
        self.parzen_first_five_test = []
        self.parzen_multi_h_results_gaussian = []
        self.parzen_multi_h_results_hypercube = []
        self.parzen_optimal_h = 1.0
        self.parzen_optimal_window = 'gaussian'
        self.parzen_optimal_model = None
        self.parzen_optimal_results = None
        self.parzen_comparison = None
        self.parzen_analysis_plot = None
        self.parzen_cm_default_plot = None
        self.parzen_cm_optimal_plot = None

        # Ho-Kashyap results
        self.hk_default_model = None
        self.hk_default_results = None
        self.hk_eta_results = []
        self.hk_optimal_eta = 0.1
        self.hk_optimal_model = None
        self.hk_active_model = None
        self.hk_separability_status = None
        self.hk_convergence_plot = None
        self.hk_eta_plot = None
        self.hk_cm_plot = None

        # KNN results
        self.knn_max_k = 40
        self.knn_k1_results = None
        self.knn_elbow_curve = []
        self.knn_selected_elbow_k = None
        self.knn_optimal_results = None
        self.knn_model = None
        self.knn_elbow_plot = None
        self.knn_cm_k1_plot = None
        self.knn_cm_optimal_plot = None

        # Bayes results
        self.bayes_model = None
        self.bayes_results = None
        self.bayes_cm_plot = None

        # Instance prediction state
        self.last_prediction = None

        # Reclaim system memory
        import gc
        gc.collect()

    def load_dataset(self, file_path, original_filename):
        self.reset()
        self.filename = original_filename

        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except Exception:
                df = pd.read_csv(file_path, sep=None, engine='python')

        df = df.dropna(how='all')
        df.columns = [str(c).strip() for c in df.columns]
        self.raw_df = df.copy()

        rows, cols = df.shape
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        # Smart detection of target column:
        target_candidate = None
        priority_keywords = ['heartdisease', 'target', 'target_class', 'label', 'class', 'species', 'outcome', 'diagnosis', 'survived', 'churn', 'status']
        # 1. First check for exact keyword match
        for kw in priority_keywords:
            for c in df.columns:
                if c.lower() == kw:
                    target_candidate = c
                    break
            if target_candidate:
                break

        # 2. Substring match
        if not target_candidate:
            for kw in priority_keywords:
                for c in df.columns:
                    if kw in c.lower():
                        target_candidate = c
                        break
                if target_candidate:
                    break

        # 3. Fallback to last column
        if not target_candidate:
            target_candidate = df.columns[-1]

        self.target_col = target_candidate

        col_meta = []
        col_unique = {}
        has_cat_features = False

        for col in df.columns:
            non_null = df[col].dropna()
            numeric_flag = pd.api.types.is_numeric_dtype(df[col])

            if not numeric_flag and len(non_null) > 0:
                numeric_flag = all(is_numeric_value(val) for val in non_null.head(100))

            if not numeric_flag and col != target_candidate:
                has_cat_features = True

            unique_vals = [normalize_class_label(v) for v in sorted(list(non_null.unique()))]
            col_unique[col] = unique_vals

            col_meta.append({
                'name': col,
                'dtype': str(df[col].dtype),
                'is_numeric': numeric_flag,
                'missing_count': int(df[col].isnull().sum()),
                'unique_count': len(unique_vals),
                'unique_values': unique_vals[:15]
            })

        self.has_missing = missing_count > 0
        self.has_duplicates = duplicate_count > 0
        self.has_categorical = has_cat_features
        self.requires_preprocessing = self.has_missing or self.has_categorical
        self.column_metadata = col_meta
        self.column_unique_map = col_unique

        self.summary = {
            'filename': original_filename,
            'rows': rows,
            'cols': cols,
            'missing_count': missing_count,
            'duplicate_count': duplicate_count,
            'preview': df.head(8).to_dict(orient='records'),
            'columns': list(df.columns),
            'default_target': target_candidate
        }

        # Always auto-execute preprocessing so dataset is immediately prepared and usable
        self.execute_preprocessing(
            target_col=self.target_col,
            impute_strategy='mean',
            cat_handling='encode',
            use_standardization=False
        )

    def execute_preprocessing(self, target_col=None, impute_strategy='mean',
                              cat_handling='encode', use_standardization=False,
                              excluded_features=None):
        if self.raw_df is None:
            raise ValueError("No dataset loaded.")

        if target_col is not None:
            self.target_col = target_col
        if excluded_features is None:
            excluded_features = []

        df = self.raw_df.copy()
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")

        # Drop duplicates if present
        df = df.drop_duplicates().reset_index(drop=True)

        # Drop rows where target is missing
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        # Stratified subsampling for responsiveness and memory safety if dataset > 5000 rows
        if len(df) > 5000:
            try:
                sampled_indices = []
                for c in df[self.target_col].unique():
                    c_indices = df[df[self.target_col] == c].index.values
                    c_target_count = max(2, int(round(5000 * len(c_indices) / len(df))))
                    c_sampled = np.random.RandomState(42).choice(
                        c_indices, size=min(len(c_indices), c_target_count), replace=False
                    )
                    sampled_indices.extend(c_sampled)
                df = df.loc[sampled_indices].reset_index(drop=True)
            except Exception:
                df = df.sample(n=5000, random_state=42).reset_index(drop=True)

        target_series = df[self.target_col]
        if target_series.nunique() < 2:
            raise ValueError(f"Target column '{self.target_col}' must contain at least 2 distinct classes.")

        feature_df = df.drop(columns=[self.target_col] + [col for col in excluded_features if col in df.columns])

        # Drop columns with 100% missing values
        feature_df = feature_df.dropna(axis=1, how='all')

        # Categorical feature handling
        numeric_cols = []
        cat_cols = []
        for col in feature_df.columns:
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                numeric_cols.append(col)
            else:
                cat_cols.append(col)

        if cat_handling == 'exclude':
            feature_df = feature_df[numeric_cols]
        else:
            for col in cat_cols:
                non_null = feature_df[col].dropna()
                unique_vals = sorted(list(non_null.unique()))
                val_to_code = {v: i for i, v in enumerate(unique_vals)}
                feature_df[col] = feature_df[col].map(val_to_code)

        # Impute missing feature values
        if feature_df.isnull().sum().sum() > 0:
            if impute_strategy == 'drop':
                valid_mask = feature_df.notnull().all(axis=1)
                feature_df = feature_df[valid_mask].reset_index(drop=True)
                target_series = target_series[valid_mask].reset_index(drop=True)
            else:
                imputer = SimpleImputer(strategy=impute_strategy)
                feature_df = pd.DataFrame(imputer.fit_transform(feature_df), columns=feature_df.columns)

        if len(feature_df.columns) == 0:
            raise ValueError("Dataset has no valid feature columns left after exclusions.")

        self.feature_cols = list(feature_df.columns)
        self.processed_df = feature_df.copy()

        # Fit label encoder
        self.encoder.fit(target_series.values)
        y_encoded = self.encoder.transform(target_series.values)
        X_vals = self.processed_df.values.astype(np.float64)

        # Compute feature statistics for domain range validation
        self.feature_stats = {}
        for idx, col in enumerate(self.feature_cols):
            vals = X_vals[:, idx]
            self.feature_stats[col] = {
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)) if np.std(vals) > 0 else 1.0
            }

        # Stratified 3-way split: 60% Train, 20% Val, 20% Test
        X_tr, X_va, X_te, y_tr, y_va, y_te = custom_train_val_test_split(
            X_vals, y_encoded, test_size=0.2, val_size=0.2, random_state=42, stratify=True
        )

        self.use_standardization = bool(use_standardization)
        if self.use_standardization:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(X_tr)
            self.X_val = self.scaler.transform(X_va)
            self.X_test = self.scaler.transform(X_te)
        else:
            self.scaler = None
            self.X_train = X_tr
            self.X_val = X_va
            self.X_test = X_te

        self.y_train = y_tr
        self.y_val = y_va
        self.y_test = y_te
        self.preprocessing_done = True

        # Generate class distribution plot
        class_codes = list(range(len(self.encoder.classes_)))
        class_labels = [self.encoder.get_display_name(c) for c in class_codes]
        counts = [int(np.sum(y_encoded == c)) for c in class_codes]
        cd_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'class_distribution.png')
        plot_class_distribution_bar(counts, class_labels, cd_plot_path)

    # -----------------------------------------------------------------
    # PARZEN WINDOW CLASSIFIER TRAINING & ANALYSIS
    # -----------------------------------------------------------------

    def train_parzen(self, h_values=None, use_standardization=None):
        if not self.preprocessing_done:
            raise ValueError("Preprocessing must be completed before training.")

        if use_standardization is not None and use_standardization != self.use_standardization:
            self.execute_preprocessing(use_standardization=use_standardization)

        if h_values is None or len(h_values) == 0:
            h_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

        h_values = sorted([float(h) for h in h_values if float(h) > 0])
        labels = list(range(len(self.encoder.classes_)))
        display_labels = [self.encoder.get_display_name(c) for c in labels]

        # 1. APPLY DEFAULT PARAMETERS (Gaussian Window, h = 1.0)
        default_clf = CustomParzenClassifier(window='gaussian', h=1.0)
        default_clf.fit(self.X_train, self.y_train)
        self.parzen_default_model = default_clf

        y_test_pred_def, posteriors_def, densities_def, log_densities_def, _ = default_clf.predict_details(self.X_test)
        self.parzen_default_results = compute_classification_metrics(self.y_test, y_test_pred_def, labels)

        # Record first five test samples details
        first_five = []
        n_five = min(5, len(self.X_test))
        for i in range(n_five):
            dens_dict = {}
            post_dict = {}
            for c_idx, c in enumerate(labels):
                dens_val = float(densities_def[i, c_idx])
                post_val = float(posteriors_def[i, c_idx])
                dens_str = f"{dens_val:.4e}" if (0 < dens_val < 1e-4 or dens_val > 1e4) else f"{dens_val:.6f}"
                dens_dict[display_labels[c_idx]] = dens_str
                post_dict[display_labels[c_idx]] = round(post_val * 100.0, 2)

            true_c = int(self.y_test[i])
            pred_c = int(y_test_pred_def[i])
            first_five.append({
                'sample_id': i + 1,
                'true_class': self.encoder.get_display_name(true_c),
                'pred_class': self.encoder.get_display_name(pred_c),
                'is_correct': (true_c == pred_c),
                'densities': dens_dict,
                'posteriors': post_dict
            })
        self.parzen_first_five_test = first_five

        # Generate Default Confusion Matrix Heatmap
        cm_def_path = os.path.join(app.config['PLOT_FOLDER'], 'parzen_cm_default.png')
        plot_confusion_matrix_heatmap(
            np.array(self.parzen_default_results['confusion_matrix']),
            display_labels,
            'Parzen Window Confusion Matrix (Default: Gaussian, h=1.0)',
            cm_def_path
        )
        self.parzen_cm_default_plot = 'plots/parzen_cm_default.png'

        # 2. WINDOW WIDTH ANALYSIS: Evaluate multi-h for Gaussian and Hypercube
        results_gaussian = []
        results_hypercube = []

        best_val_err = float('inf')
        best_window = 'gaussian'
        best_h = 1.0

        # Rapid evaluation subset for training error during multi-h sweep (preserves class distribution)
        if len(self.X_train) > 600:
            rng_eval = np.random.RandomState(42)
            eval_indices = []
            for c in np.unique(self.y_train):
                c_idx = np.where(self.y_train == c)[0]
                c_take = max(2, int(round(600 * len(c_idx) / len(self.y_train))))
                eval_indices.extend(rng_eval.choice(c_idx, min(len(c_idx), c_take), replace=False))
            X_tr_eval = self.X_train[eval_indices]
            y_tr_eval = self.y_train[eval_indices]
        else:
            X_tr_eval = self.X_train
            y_tr_eval = self.y_train

        for win in ['gaussian', 'hypercube']:
            for h in h_values:
                clf = CustomParzenClassifier(window=win, h=h)
                clf.fit(self.X_train, self.y_train)

                # Predict on train and val sets
                tr_preds, _, _, _, empty_tr = clf.predict_details(X_tr_eval)
                va_preds, _, _, _, empty_va = clf.predict_details(self.X_val)

                tr_err = float(np.mean(tr_preds != y_tr_eval) * 100.0)
                va_err = float(np.mean(va_preds != self.y_val) * 100.0)
                empty_count_va = int(np.sum(empty_va)) if win == 'hypercube' else 0

                entry = {
                    'window': win,
                    'h': h,
                    'train_error_rate': round(tr_err, 2),
                    'val_error_rate': round(va_err, 2),
                    'empty_windows_val': empty_count_va
                }

                if win == 'gaussian':
                    results_gaussian.append(entry)
                else:
                    results_hypercube.append(entry)

                # Optimal selection criterion: lowest validation error rate
                if va_err < best_val_err:
                    best_val_err = va_err
                    best_window = win
                    best_h = h
                elif abs(va_err - best_val_err) < 1e-5:
                    # Tie breaking: pick Gaussian, or h closest to 1.0
                    if win == 'gaussian' and best_window == 'hypercube':
                        best_window = win
                        best_h = h
                    elif win == best_window and abs(h - 1.0) < abs(best_h - 1.0):
                        best_h = h

        self.parzen_multi_h_results_gaussian = results_gaussian
        self.parzen_multi_h_results_hypercube = results_hypercube
        self.parzen_optimal_h = best_h
        self.parzen_optimal_window = best_window

        # 3. FIT & EVALUATE OPTIMAL MODEL
        optimal_clf = CustomParzenClassifier(window=best_window, h=best_h)
        optimal_clf.fit(self.X_train, self.y_train)
        self.parzen_optimal_model = optimal_clf

        y_test_pred_opt, _, _, _, empty_opt_test = optimal_clf.predict_details(self.X_test)
        self.parzen_optimal_results = compute_classification_metrics(self.y_test, y_test_pred_opt, labels)
        self.parzen_optimal_results['empty_windows_test'] = int(np.sum(empty_opt_test)) if best_window == 'hypercube' else 0

        # Generate Optimal Confusion Matrix Heatmap
        cm_opt_path = os.path.join(app.config['PLOT_FOLDER'], 'parzen_cm_optimal.png')
        plot_confusion_matrix_heatmap(
            np.array(self.parzen_optimal_results['confusion_matrix']),
            display_labels,
            f"Parzen Window Confusion Matrix (Optimal: {best_window.capitalize()}, h={best_h})",
            cm_opt_path
        )
        self.parzen_cm_optimal_plot = 'plots/parzen_cm_optimal.png'

        # Generate Multi-h Analysis Plot
        analysis_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'parzen_h_analysis.png')
        plot_parzen_window_analysis(
            h_values, results_gaussian, results_hypercube, best_h, best_window, analysis_plot_path
        )
        self.parzen_analysis_plot = 'plots/parzen_h_analysis.png'

        # 4. SIDE-BY-SIDE COMPARISON: Default vs Optimal
        self.parzen_comparison = {
            'default': {
                'window': 'Gaussian',
                'h': 1.0,
                'accuracy': self.parzen_default_results['accuracy'],
                'error_rate': self.parzen_default_results['error_rate'],
                'precision': self.parzen_default_results['precision'],
                'recall': self.parzen_default_results['recall'],
                'f1_score': self.parzen_default_results['f1_score']
            },
            'optimal': {
                'window': best_window.capitalize(),
                'h': best_h,
                'accuracy': self.parzen_optimal_results['accuracy'],
                'error_rate': self.parzen_optimal_results['error_rate'],
                'precision': self.parzen_optimal_results['precision'],
                'recall': self.parzen_optimal_results['recall'],
                'f1_score': self.parzen_optimal_results['f1_score']
            },
            'delta': {
                'accuracy': round(self.parzen_optimal_results['accuracy'] - self.parzen_default_results['accuracy'], 2),
                'error_rate': round(self.parzen_optimal_results['error_rate'] - self.parzen_default_results['error_rate'], 2),
                'f1_score': round(self.parzen_optimal_results['f1_score'] - self.parzen_default_results['f1_score'], 2)
            }
        }

        self.active_classifier = 'parzen'
        return {
            'default_results': self.parzen_default_results,
            'optimal_results': self.parzen_optimal_results,
            'optimal_h': best_h,
            'optimal_window': best_window
        }

    # -----------------------------------------------------------------
    # HO-KASHYAP PROCEDURE (INTEGRATED FROM ASSIGNMENT 08)
    # -----------------------------------------------------------------

    def train_hokashyap(self, default_eta=0.1, max_iter=500, tol=1e-5, b_init=1.0, selected_etas=None):
        if not self.preprocessing_done:
            raise ValueError("Preprocessing required before training.")

        labels = list(range(len(self.encoder.classes_)))
        if len(labels) != 2:
            raise ValueError(f"Ho-Kashyap procedure requires a binary dataset (exactly 2 classes). Found {len(labels)} classes.")

        display_labels = [self.encoder.get_display_name(c) for c in labels]

        # Combine train + val for HK binary classifier
        X_tr = np.vstack([self.X_train, self.X_val])
        y_tr = np.concatenate([self.y_train, self.y_val])

        # Default model fit
        hk_default = CustomHoKashyapClassifier(learning_rate=default_eta, max_iter=max_iter, tol=tol, b_init=b_init)
        hk_default.fit(X_tr, y_tr)
        self.hk_default_model = hk_default
        self.hk_active_model = hk_default
        self.hk_separability_status = hk_default.separability_

        y_test_pred = hk_default.predict(self.X_test)
        self.hk_default_results = compute_classification_metrics(self.y_test, y_test_pred, labels)

        # Multi-eta analysis
        if selected_etas is None:
            selected_etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        eta_results = []
        best_acc = -1.0
        min_iters = float('inf')
        opt_eta = default_eta

        for eta_val in selected_etas:
            m = CustomHoKashyapClassifier(learning_rate=eta_val, max_iter=max_iter, tol=tol, b_init=b_init)
            m.fit(X_tr, y_tr)
            preds = m.predict(self.X_test)
            metrics = compute_classification_metrics(self.y_test, preds, labels)

            entry = {
                'eta': round(eta_val, 2),
                'iterations': m.iterations_run_,
                'converged': m.converged_,
                'separability': m.separability_,
                'accuracy': metrics['accuracy'],
                'error_rate': metrics['error_rate'],
                'f1_score': metrics['f1_score']
            }
            eta_results.append(entry)

            if (metrics['accuracy'] > best_acc) or (abs(metrics['accuracy'] - best_acc) < 1e-4 and m.iterations_run_ < min_iters):
                best_acc = metrics['accuracy']
                min_iters = m.iterations_run_
                opt_eta = eta_val

        self.hk_eta_results = eta_results
        self.hk_optimal_eta = opt_eta

        # Generate HK plots
        conv_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'hokashyap_convergence.png')
        plot_hokashyap_convergence(hk_default.history_, conv_plot_path)
        self.hk_convergence_plot = 'plots/hokashyap_convergence.png'

        eta_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'hokashyap_eta_curve.png')
        plot_eta_vs_iterations(eta_results, opt_eta, eta_plot_path)
        self.hk_eta_plot = 'plots/hokashyap_eta_curve.png'

        cm_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'hokashyap_cm.png')
        plot_confusion_matrix_heatmap(
            np.array(self.hk_default_results['confusion_matrix']),
            display_labels,
            'Ho-Kashyap Linear Classifier Confusion Matrix',
            cm_plot_path
        )
        self.hk_cm_plot = 'plots/hokashyap_cm.png'

        self.active_classifier = 'hokashyap'
        return {
            'metrics': self.hk_default_results,
            'separability': self.hk_separability_status,
            'iterations_run': hk_default.iterations_run_,
            'optimal_eta': opt_eta
        }

    # -----------------------------------------------------------------
    # KNN CLASSIFIER (INTEGRATED FROM ASSIGNMENTS 07/08)
    # -----------------------------------------------------------------

    def train_knn(self, max_k=40):
        if not self.preprocessing_done:
            raise ValueError("Preprocessing required before training.")

        labels = list(range(len(self.encoder.classes_)))
        display_labels = [self.encoder.get_display_name(c) for c in labels]

        self.knn_max_k = min(max_k, len(self.X_train) - 1)
        k_values = list(range(1, self.knn_max_k + 1))
        error_rates = []

        best_k = 1
        lowest_err = float('inf')

        for k in k_values:
            clf = CustomKNNClassifier(n_neighbors=k)
            clf.fit(self.X_train, self.y_train)
            preds = clf.predict(self.X_val)
            err = float(np.mean(preds != self.y_val) * 100.0)
            error_rates.append(round(err, 2))
            if err < lowest_err:
                lowest_err = err
                best_k = k

        self.knn_elbow_curve = [{'k': k, 'error_rate': err} for k, err in zip(k_values, error_rates)]
        self.knn_selected_elbow_k = best_k

        # Fit K=1 baseline
        k1_clf = CustomKNNClassifier(n_neighbors=1).fit(self.X_train, self.y_train)
        y_test_k1 = k1_clf.predict(self.X_test)
        self.knn_k1_results = compute_classification_metrics(self.y_test, y_test_k1, labels)

        # Fit optimal elbow K model
        opt_clf = CustomKNNClassifier(n_neighbors=best_k).fit(self.X_train, self.y_train)
        self.knn_model = opt_clf
        y_test_opt = opt_clf.predict(self.X_test)
        self.knn_optimal_results = compute_classification_metrics(self.y_test, y_test_opt, labels)

        # Generate KNN plots
        elbow_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'knn_elbow_curve.png')
        plot_knn_elbow(k_values, error_rates, best_k, elbow_plot_path)
        self.knn_elbow_plot = 'plots/knn_elbow_curve.png'

        cm_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'knn_optimal_cm.png')
        plot_confusion_matrix_heatmap(
            np.array(self.knn_optimal_results['confusion_matrix']),
            display_labels,
            f'KNN Optimal Model (K={best_k}) Confusion Matrix',
            cm_plot_path
        )
        self.knn_cm_optimal_plot = 'plots/knn_optimal_cm.png'

        self.active_classifier = 'knn'
        return {
            'k1_results': self.knn_k1_results,
            'optimal_results': self.knn_optimal_results,
            'selected_k': best_k
        }

    # -----------------------------------------------------------------
    # BAYES CLASSIFIER (INTEGRATED FROM ASSIGNMENTS 06/07/08)
    # -----------------------------------------------------------------

    def train_bayes(self):
        if not self.preprocessing_done:
            raise ValueError("Preprocessing required before training.")

        labels = list(range(len(self.encoder.classes_)))
        display_labels = [self.encoder.get_display_name(c) for c in labels]

        # Combine train + val for Bayes
        X_tr = np.vstack([self.X_train, self.X_val])
        y_tr = np.concatenate([self.y_train, self.y_val])

        bayes = PatternClassifiers()
        bayes.fit(X_tr, y_tr)
        self.bayes_model = bayes

        preds = bayes.predict(self.X_test)
        self.bayes_results = compute_classification_metrics(self.y_test, preds, labels)

        cm_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'bayes_cm.png')
        plot_confusion_matrix_heatmap(
            np.array(self.bayes_results['confusion_matrix']),
            display_labels,
            'Gaussian Naive Bayes Confusion Matrix',
            cm_plot_path
        )
        self.bayes_cm_plot = 'plots/bayes_cm.png'

        self.active_classifier = 'bayes'
        return self.bayes_results

    # -----------------------------------------------------------------
    # INSTANCE PREDICTION ENGINE WITH INPUT VALIDATION & OUTLIER ALERTS
    # -----------------------------------------------------------------

    def predict_new_instance(self, input_dict):
        if not self.preprocessing_done:
            raise ValueError("No model trained yet.")

        # 1. Validation: non-empty, numeric, and training domain range check
        raw_values = []
        domain_warnings = []

        for col in self.feature_cols:
            val_str = str(input_dict.get(col, '')).strip()
            if val_str == '':
                raise ValueError(f"Feature '{col}' cannot be empty.")
            try:
                val_float = float(val_str)
            except ValueError:
                raise ValueError(f"Feature '{col}' must be a valid numeric number (received '{val_str}').")

            raw_values.append(val_float)

            # Domain range validation
            stats = self.feature_stats.get(col)
            if stats:
                f_min, f_max = stats['min'], stats['max']
                if val_float < f_min or val_float > f_max:
                    domain_warnings.append({
                        'feature': col,
                        'value': val_float,
                        'training_range': f"[{f_min:.2f}, {f_max:.2f}]",
                        'status': 'Far outside range' if (val_float < f_min - 2 * stats['std'] or val_float > f_max + 2 * stats['std']) else 'Outside range'
                    })

        raw_vec = np.array(raw_values, dtype=np.float64).reshape(1, -1)

        # Apply pre-processing fitted scaler if enabled
        if self.use_standardization and self.scaler is not None:
            proc_vec = self.scaler.transform(raw_vec)
        else:
            proc_vec = raw_vec

        # Predict based on active classifier
        clf_type = self.active_classifier

        if clf_type == 'parzen':
            model = self.parzen_optimal_model or self.parzen_default_model
            if model is None:
                raise ValueError("Parzen Window classifier not trained.")
            explanation = model.explain_instance(proc_vec)
            pred_code = explanation['predicted_class']
            display_name = self.encoder.get_display_name(pred_code)

            result = {
                'classifier': 'Parzen Window Density Estimator',
                'predicted_code': pred_code,
                'predicted_label': display_name,
                'window': explanation['window'].capitalize(),
                'h': explanation['h'],
                'is_empty_window': explanation['is_empty_window'],
                'class_probabilities': [
                    {
                        'class_code': c['class_code'],
                        'display_name': self.encoder.get_display_name(c['class_code']),
                        'prior': c['prior'],
                        'density': c['density_str'],
                        'log_density': c['log_density'],
                        'posterior': c['posterior'],
                        'posterior_percent': c['posterior_percent']
                    }
                    for c in explanation['classes']
                ],
                'domain_warnings': domain_warnings,
                'raw_inputs': {col: raw_values[i] for i, col in enumerate(self.feature_cols)}
            }

        elif clf_type == 'hokashyap':
            model = self.hk_active_model or self.hk_default_model
            if model is None:
                raise ValueError("Ho-Kashyap classifier not trained.")
            pred_code = int(model.predict(proc_vec)[0])
            score = float(model.decision_function(proc_vec)[0])
            proba = model.predict_proba(proc_vec)[0]
            display_name = self.encoder.get_display_name(pred_code)

            result = {
                'classifier': 'Ho-Kashyap Linear Classifier',
                'predicted_code': pred_code,
                'predicted_label': display_name,
                'decision_score': round(score, 4),
                'class_probabilities': [
                    {
                        'class_code': int(c),
                        'display_name': self.encoder.get_display_name(int(c)),
                        'posterior': round(float(proba[i]), 4),
                        'posterior_percent': round(float(proba[i]) * 100.0, 2)
                    }
                    for i, c in enumerate(model.classes_)
                ],
                'domain_warnings': domain_warnings,
                'raw_inputs': {col: raw_values[i] for i, col in enumerate(self.feature_cols)}
            }

        elif clf_type == 'knn':
            model = self.knn_model or CustomKNNClassifier(1).fit(self.X_train, self.y_train)
            exp = model.explain_instance(proc_vec)
            pred_code = exp['predicted_class']
            display_name = self.encoder.get_display_name(pred_code)

            result = {
                'classifier': f"K-Nearest Neighbors (K={model.n_neighbors})",
                'predicted_code': pred_code,
                'predicted_label': display_name,
                'class_probabilities': [
                    {
                        'class_code': int(c),
                        'display_name': self.encoder.get_display_name(int(c)),
                        'posterior': round(float(exp['probabilities'][int(c)]), 4),
                        'posterior_percent': round(float(exp['probabilities'][int(c)]) * 100.0, 2)
                    }
                    for c in model.classes_
                ],
                'neighbors': exp['neighbors'][:5],
                'domain_warnings': domain_warnings,
                'raw_inputs': {col: raw_values[i] for i, col in enumerate(self.feature_cols)}
            }

        else:  # bayes
            model = self.bayes_model
            if model is None:
                raise ValueError("Bayes classifier not trained.")
            probs_map = model.predict_proba_single(proc_vec[0])
            pred_code = max(probs_map, key=probs_map.get)
            display_name = self.encoder.get_display_name(pred_code)

            result = {
                'classifier': 'Gaussian Naive Bayes',
                'predicted_code': pred_code,
                'predicted_label': display_name,
                'class_probabilities': [
                    {
                        'class_code': int(c),
                        'display_name': self.encoder.get_display_name(int(c)),
                        'posterior': round(float(probs_map[int(c)]), 4),
                        'posterior_percent': round(float(probs_map[int(c)]) * 100.0, 2)
                    }
                    for c in model.classes
                ],
                'domain_warnings': domain_warnings,
                'raw_inputs': {col: raw_values[i] for i, col in enumerate(self.feature_cols)}
            }

        self.last_prediction = result
        return result

    # -----------------------------------------------------------------
    # COMPARATIVE PERFORMANCE ANALYSIS (ALL 4 CLASSIFIERS BENCHMARK)
    # -----------------------------------------------------------------

    def run_performance_analysis(self):
        if not self.preprocessing_done:
            raise ValueError("Preprocessing required before running performance analysis.")

        labels = list(range(len(self.encoder.classes_)))
        benchmark_results = []

        # 1. Parzen Window Classifier
        t0 = time.time()
        parzen_clf = self.parzen_optimal_model
        if parzen_clf is None:
            parzen_clf = CustomParzenClassifier(window='gaussian', h=1.0).fit(self.X_train, self.y_train)
        parzen_preds = parzen_clf.predict(self.X_test)
        t_parzen = round((time.time() - t0) * 1000.0, 2)
        m_parzen = compute_classification_metrics(self.y_test, parzen_preds, labels)

        benchmark_results.append({
            'name': f"Parzen Window ({parzen_clf.window.capitalize()}, h={parzen_clf.h})",
            'type': 'Parzen Window',
            'accuracy': m_parzen['accuracy'],
            'error_rate': m_parzen['error_rate'],
            'precision': m_parzen['precision'],
            'recall': m_parzen['recall'],
            'f1_score': m_parzen['f1_score'],
            'latency_ms': t_parzen,
            'status': 'Evaluated'
        })

        # 2. Ho-Kashyap Linear Classifier (if binary)
        if len(labels) == 2:
            t0 = time.time()
            hk_clf = self.hk_default_model
            if hk_clf is None:
                X_tr = np.vstack([self.X_train, self.X_val])
                y_tr = np.concatenate([self.y_train, self.y_val])
                hk_clf = CustomHoKashyapClassifier(learning_rate=0.1).fit(X_tr, y_tr)
            hk_preds = hk_clf.predict(self.X_test)
            t_hk = round((time.time() - t0) * 1000.0, 2)
            m_hk = compute_classification_metrics(self.y_test, hk_preds, labels)

            benchmark_results.append({
                'name': f"Ho-Kashyap ({hk_clf.separability_})",
                'type': 'Ho-Kashyap',
                'accuracy': m_hk['accuracy'],
                'error_rate': m_hk['error_rate'],
                'precision': m_hk['precision'],
                'recall': m_hk['recall'],
                'f1_score': m_hk['f1_score'],
                'latency_ms': t_hk,
                'status': 'Evaluated'
            })
        else:
            benchmark_results.append({
                'name': "Ho-Kashyap Linear Classifier",
                'type': 'Ho-Kashyap',
                'accuracy': 0.0,
                'error_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'latency_ms': 0.0,
                'status': f"Skipped (Requires binary dataset, found {len(labels)} classes)"
            })

        # 3. K-Nearest Neighbors Classifier
        t0 = time.time()
        knn_clf = self.knn_model
        if knn_clf is None:
            knn_clf = CustomKNNClassifier(n_neighbors=self.knn_selected_elbow_k or 3).fit(self.X_train, self.y_train)
        knn_preds = knn_clf.predict(self.X_test)
        t_knn = round((time.time() - t0) * 1000.0, 2)
        m_knn = compute_classification_metrics(self.y_test, knn_preds, labels)

        benchmark_results.append({
            'name': f"KNN (K={knn_clf.n_neighbors})",
            'type': 'KNN',
            'accuracy': m_knn['accuracy'],
            'error_rate': m_knn['error_rate'],
            'precision': m_knn['precision'],
            'recall': m_knn['recall'],
            'f1_score': m_knn['f1_score'],
            'latency_ms': t_knn,
            'status': 'Evaluated'
        })

        # 4. Gaussian Naive Bayes Classifier
        t0 = time.time()
        bayes_clf = self.bayes_model
        if bayes_clf is None:
            X_tr = np.vstack([self.X_train, self.X_val])
            y_tr = np.concatenate([self.y_train, self.y_val])
            bayes_clf = PatternClassifiers()
            bayes_clf.fit(X_tr, y_tr)
        bayes_preds = bayes_clf.predict(self.X_test)
        t_bayes = round((time.time() - t0) * 1000.0, 2)
        m_bayes = compute_classification_metrics(self.y_test, bayes_preds, labels)

        benchmark_results.append({
            'name': "Gaussian Naive Bayes",
            'type': 'Bayes',
            'accuracy': m_bayes['accuracy'],
            'error_rate': m_bayes['error_rate'],
            'precision': m_bayes['precision'],
            'recall': m_bayes['recall'],
            'f1_score': m_bayes['f1_score'],
            'latency_ms': t_bayes,
            'status': 'Evaluated'
        })

        # Generate comparative chart
        evaluated_only = [r for r in benchmark_results if r['status'] == 'Evaluated']
        chart_path = os.path.join(app.config['PLOT_FOLDER'], 'benchmark_comparison.png')
        plot_comparative_benchmark(evaluated_only, chart_path)

        # Recommendation
        best_model = max(evaluated_only, key=lambda x: (x['accuracy'], x['f1_score']))

        return {
            'benchmark_table': benchmark_results,
            'benchmark_chart': 'plots/benchmark_comparison.png',
            'best_model': best_model
        }


app_instance = AppState()


# =====================================================================
# FLASK WEB CONTROLLER & ROUTING ENDPOINTS
# =====================================================================

@app.route('/', methods=['GET'])
def index():
    # List pre-bundled datasets
    available_datasets = []
    if os.path.exists(app.config['DATASET_FOLDER']):
        for f in sorted(os.listdir(app.config['DATASET_FOLDER'])):
            if f.endswith('.csv'):
                fpath = os.path.join(app.config['DATASET_FOLDER'], f)
                available_datasets.append({
                    'name': f,
                    'size_kb': round(os.path.getsize(fpath) / 1024, 1)
                })

    return render_template(
        'index.html',
        app_state=app_instance,
        available_datasets=available_datasets,
        cache_buster=int(time.time())
    )


@app.route('/upload', methods=['POST'])
def upload():
    source = request.form.get('source', 'upload')

    if source == 'preset':
        preset_filename = request.form.get('preset_dataset') or request.form.get('sample_dataset')
        if not preset_filename:
            flash("Please choose a valid pre-bundled dataset.", "error")
            return redirect(url_for('index'))
        file_path = os.path.join(app.config['DATASET_FOLDER'], preset_filename)
        if not os.path.exists(file_path):
            flash(f"Pre-bundled dataset '{preset_filename}' not found.", "error")
            return redirect(url_for('index'))
        display_name = preset_filename
    else:
        file = request.files.get('dataset_file') or request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected for upload. Please browse a CSV file.", "error")
            return redirect(url_for('index'))

        if not file.filename.lower().endswith('.csv'):
            flash(f"Invalid file '{file.filename}'. Please upload a CSV (.csv) file.", "error")
            return redirect(url_for('index'))

        display_name = secure_filename(file.filename) or 'uploaded_dataset.csv'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], display_name)
        file.save(file_path)

    try:
        app_instance.load_dataset(file_path, display_name)
        flash(
            f"Dataset '{display_name}' loaded successfully! ({app_instance.summary['rows']} rows, {app_instance.summary['cols']} columns).",
            "success"
        )
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-summary'))


@app.route('/prepare', methods=['POST'])
def prepare():
    target_col = request.form.get('target_col')
    impute_strategy = request.form.get('impute_strategy', 'mean')
    cat_handling = request.form.get('cat_handling', 'encode')
    use_standardization = 'use_standardization' in request.form
    excluded_features = request.form.getlist('excluded_features')

    try:
        app_instance.execute_preprocessing(
            target_col=target_col,
            impute_strategy=impute_strategy,
            cat_handling=cat_handling,
            use_standardization=use_standardization,
            excluded_features=excluded_features
        )
        msg = f"Data pre-processed successfully! Target: '{app_instance.target_col}', Features: {len(app_instance.feature_cols)}."
        if use_standardization:
            msg += " Feature standardization (StandardScaler) applied."
        flash(msg, "success")
    except Exception as e:
        flash(f"Pre-processing failed: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-preprocess'))


@app.route('/rename_classes', methods=['POST'])
def rename_classes():
    try:
        for key, val in request.form.items():
            if key.startswith('rename_class_'):
                code = int(key.replace('rename_class_', ''))
                name = str(val).strip()
                if name:
                    app_instance.encoder.set_display_name(code, name)

        # Refresh class distribution plot with new labels
        class_codes = list(range(len(app_instance.encoder.classes_)))
        class_labels = [app_instance.encoder.get_display_name(c) for c in class_codes]
        counts = [int(np.sum(app_instance.y_train == c) + np.sum(app_instance.y_val == c) + np.sum(app_instance.y_test == c)) for c in class_codes]
        cd_plot_path = os.path.join(app.config['PLOT_FOLDER'], 'class_distribution.png')
        plot_class_distribution_bar(counts, class_labels, cd_plot_path)

        flash("Target class display names updated successfully!", "success")
    except Exception as e:
        flash(f"Error updating class names: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-preprocess'))


@app.route('/train', methods=['POST'])
def train():
    classifier_choice = request.form.get('classifier_choice', 'parzen')
    use_standardization = 'use_standardization' in request.form

    # Auto-execute preprocessing if not done yet, or re-apply if standardization toggled
    if not app_instance.preprocessing_done:
        app_instance.execute_preprocessing(use_standardization=use_standardization)
    elif app_instance.use_standardization != use_standardization:
        app_instance.execute_preprocessing(use_standardization=use_standardization)

    try:
        if classifier_choice == 'parzen':
            # Parzen Window configuration
            h_input_str = request.form.get('h_values', '0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0')
            h_list = []
            for part in h_input_str.split(','):
                part = part.strip()
                if part:
                    try:
                        val = float(part)
                        if val > 0:
                            h_list.append(val)
                    except ValueError:
                        pass
            if not h_list:
                h_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

            results = app_instance.train_parzen(h_values=h_list, use_standardization=use_standardization)
            flash(
                f"Parzen Window Density Estimator Trained! "
                f"Default Model (Gaussian, h=1.0): Test Accuracy {results['default_results']['accuracy']}%, Error Rate {results['default_results']['error_rate']}%. "
                f"Optimal Model ({results['optimal_window'].capitalize()}, h*={results['optimal_h']}): Test Accuracy {results['optimal_results']['accuracy']}%, Error Rate {results['optimal_results']['error_rate']}%.",
                "success"
            )

        elif classifier_choice == 'hokashyap':
            if len(app_instance.encoder.classes_) != 2:
                flash(f"Ho-Kashyap Linear Classifier requires a binary dataset (exactly 2 classes). Current dataset has {len(app_instance.encoder.classes_)} classes. Please select Parzen Window, KNN, or Bayes.", "error")
                return redirect(url_for('index', _anchor='step-training'))

            default_eta = float(request.form.get('default_eta', 0.1))
            max_iter = int(request.form.get('max_iter', 500))
            tol = float(request.form.get('tol', 1e-5))
            b_init = float(request.form.get('b_init', 1.0))
            selected_etas_raw = request.form.getlist('selected_etas')
            selected_etas = [float(e) for e in selected_etas_raw] if selected_etas_raw else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            results = app_instance.train_hokashyap(
                default_eta=default_eta, max_iter=max_iter, tol=tol, b_init=b_init, selected_etas=selected_etas
            )
            flash(
                f"Ho-Kashyap Linear Classifier Executed! Status: {results['separability']}. "
                f"Ran {results['iterations_run']} iterations. Test Accuracy: {results['metrics']['accuracy']}%, "
                f"Optimal η*: {results['optimal_eta']:.2f}.",
                "success"
            )

        elif classifier_choice == 'knn':
            max_k = int(request.form.get('max_k', 40))
            results = app_instance.train_knn(max_k=max_k)
            flash(
                f"KNN evaluated for K=1 (Baseline Error: {results['k1_results']['error_rate']}%) and across K=1 to {app_instance.knn_max_k}. "
                f"Optimal elbow point K={results['selected_k']} evaluated with Error Rate: {results['optimal_results']['error_rate']}%.",
                "info"
            )

        else:  # Bayes
            results = app_instance.train_bayes()
            flash(
                f"Gaussian Naive Bayes Trained: Test Accuracy: {results['accuracy']}%, F1-Score: {results['f1_score']}%.",
                "success"
            )

    except Exception as e:
        flash(f"Training failed: {str(e)}", "error")

    anchor_map = {
        'parzen': 'step-results',
        'hokashyap': 'step-results-hk',
        'knn': 'step-results-knn',
        'bayes': 'step-results-bayes'
    }
    target_anchor = anchor_map.get(classifier_choice, 'step-results')
    return redirect(url_for('index', _anchor=target_anchor))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not app_instance.preprocessing_done:
            app_instance.execute_preprocessing()
        if app_instance.parzen_default_model is None and app_instance.active_classifier == 'parzen':
            app_instance.train_parzen()

        app_instance.predict_new_instance(request.form)
        pred = app_instance.last_prediction
        flash(
            f"Prediction completed via {pred['classifier']}: Predicted Class is '{pred['predicted_label']}' (Code {pred['predicted_code']}).",
            "success"
        )
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-prediction'))


def run_garbage_collector(keep_active_file=True):
    """
    Garbage Collector:
      1. Cleans up old/stale uploaded CSV files from Uploads/.
      2. Cleans up plot images from Statics/plots/.
      3. Forces Python gc.collect() to free unreferenced memory.
    """
    import gc
    stats = {
        'uploads_deleted': 0,
        'plots_deleted': 0,
        'bytes_freed': 0
    }
    upload_dir = app.config.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'Uploads'))
    active_file = app_instance.filename if keep_active_file else None

    if os.path.exists(upload_dir):
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if os.path.isfile(fpath):
                if active_file and fname == active_file:
                    continue
                try:
                    fsize = os.path.getsize(fpath)
                    os.remove(fpath)
                    stats['uploads_deleted'] += 1
                    stats['bytes_freed'] += fsize
                except Exception:
                    pass

    plot_dir = app.config.get('PLOT_FOLDER', os.path.join(BASE_DIR, 'Statics', 'plots'))
    if not keep_active_file and os.path.exists(plot_dir):
        for fname in os.listdir(plot_dir):
            fpath = os.path.join(plot_dir, fname)
            if os.path.isfile(fpath) and fname.endswith('.png'):
                try:
                    fsize = os.path.getsize(fpath)
                    os.remove(fpath)
                    stats['plots_deleted'] += 1
                    stats['bytes_freed'] += fsize
                except Exception:
                    pass

    # Collect unreferenced objects
    gc.collect()
    mb_freed = round(stats['bytes_freed'] / (1024 * 1024), 2)
    return stats, mb_freed


@app.route('/set_target', methods=['POST'])
def set_target():
    new_target = request.form.get('target_col')
    if not new_target:
        flash("Please select a valid target column.", "error")
        return redirect(url_for('index', _anchor='step-summary'))
    try:
        app_instance.execute_preprocessing(
            target_col=new_target,
            impute_strategy='mean',
            cat_handling='encode',
            use_standardization=app_instance.use_standardization
        )
        flash(f"Target feature updated to '{app_instance.target_col}'! Class distribution and encoding recomputed successfully.", "success")
    except Exception as e:
        flash(f"Failed to set target column: {str(e)}", "error")
    return redirect(url_for('index', _anchor='step-summary'))


@app.route('/cleanup', methods=['POST'])
def cleanup():
    stats, mb_freed = run_garbage_collector(keep_active_file=True)
    flash(
        f"Garbage Collection Completed! Removed {stats['uploads_deleted']} temporary file(s) and freed {mb_freed} MB disk space. Memory reclaimed.",
        "success"
    )
    return redirect(url_for('index'))


@app.route('/reset', methods=['POST'])
def reset_app():
    stats, mb_freed = run_garbage_collector(keep_active_file=False)
    app_instance.reset()
    flash(f"Studio state reset and garbage collection executed ({mb_freed} MB cleared).", "info")
    return redirect(url_for('index'))


@app.route('/performance-analysis')
def performance_analysis():
    if not app_instance.preprocessing_done:
        flash("Please upload and preprocess a dataset before visiting the Comparative Studio.", "warning")
        return redirect(url_for('index'))

    try:
        analysis_data = app_instance.run_performance_analysis()
        return render_template(
            'performance_analysis.html',
            app_state=app_instance,
            analysis=analysis_data
        )
    except Exception as e:
        flash(f"Could not compute comparative analysis: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/shutdown', methods=['POST'])
def shutdown():
    def stop():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=stop).start()
    return "Server is shutting down..."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
