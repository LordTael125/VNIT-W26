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
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'Templates'),
    static_folder=os.path.join(BASE_DIR, 'Statics')
)
app.secret_key = 'classifiers_knn_bayes_assignment_07_secret'
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
        return np.array([self.mapping[str(v)] for v in values])

    def transform(self, values):
        encoded = []
        for v in values:
            s_val = str(v)
            if s_val in self.mapping:
                encoded.append(self.mapping[s_val])
            else:
                encoded.append(0)
        return np.array(encoded)

    def inverse_transform(self, indices):
        return [self.inverse_mapping.get(int(idx), str(idx)) for idx in indices]


# =====================================================================
# CUSTOM K-NEAREST NEIGHBORS CLASSIFIER (FROM FIRST PRINCIPLES)
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
            # Deterministic tie-breaking: pick candidate with smallest average neighbor distance
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

        # Neighbor vote distribution
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
# CUSTOM BAYES CLASSIFIER (INTEGRATED FROM ASSIGNMENT 06)
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
    return cm


def custom_metrics_evaluation(y_true, y_pred, labels):
    cm = custom_confusion_matrix(y_true, y_pred, labels)
    k = len(labels)
    total_samples = len(y_true)

    correct = sum(cm[i][i] for i in range(k))
    incorrect = total_samples - correct
    accuracy = (correct / total_samples) if total_samples > 0 else 0.0
    error_rate = (incorrect / total_samples) if total_samples > 0 else 0.0

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for i in range(k):
        tp = cm[i][i]
        fp = sum(cm[row][i] for row in range(k)) - tp
        fn = sum(cm[i][col] for col in range(k)) - tp
        support = sum(cm[i])
        supports.append(support)

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    if total_samples > 0:
        weighted_prec = sum(p * s for p, s in zip(precisions, supports)) / total_samples
        weighted_rec = sum(r * s for r, s in zip(recalls, supports)) / total_samples
        weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_samples
    else:
        weighted_prec = weighted_rec = weighted_f1 = 0.0

    return {
        'accuracy': round(accuracy * 100, 2),
        'error_rate': round(error_rate * 100, 2),
        'error_rate_decimal': round(error_rate, 4),
        'precision': round(weighted_prec * 100, 2),
        'recall': round(weighted_rec * 100, 2),
        'f1_score': round(weighted_f1 * 100, 2),
        'confusion_matrix': cm,
        'incorrect_count': incorrect,
        'total_predictions': total_samples
    }


# =====================================================================
# CLASSIFIERS APPLICATION CONTROLLER
# =====================================================================

class ClassifiersApp:
    def __init__(self):
        self.reset()

    def reset(self):
        """Completely clears state to isolate datasets from prior uploads."""
        self.filename = None
        self.df_raw = None
        self.df_clean = None

        self.summary = None
        self.all_columns = []
        self.column_unique_map = {}

        # Target & Features
        self.target_col = None
        self.target_is_numeric = False
        self.raw_target_classes = []
        self.class_display_names = {}
        self.feature_cols = []
        self.numeric_features = []
        self.categorical_features = []
        self.target_distribution = None
        self.target_plot_url = None

        # Preprocessing Requirements State
        self.preprocessing_required = False
        self.preprocessing_reasons = []
        self.preprocessing_completed = False
        self.use_standardization = True  # User choice whether to use standardized features
        self.scaler = None
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.target_encoder = None
        self.feature_encoders = {}
        self.feature_medians = {}
        self.feature_modes = {}
        self.feature_min_max = {}

        # Prepared Datasets for Modeling
        self.X_train_raw = None
        self.X_test_raw = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.2

        # Model & Evaluation State
        self.selected_classifier = 'knn'  # 'knn' or 'bayes'
        self.max_k = 40
        self.k1_results = None
        self.k1_cm_plot_url = None
        self.elbow_results = None
        self.elbow_plot_url = None
        self.selected_elbow_k = None
        self.optimal_k_results = None
        self.optimal_cm_plot_url = None
        self.comparison_k1_vs_optimal = None

        # Bayes Classifier State
        self.bayes_results = None
        self.bayes_cm_plot_url = None

        # Active Model Reference for Predictions
        self.active_model = None
        self.active_model_name = None
        self.last_prediction = None

        self._clear_plots()

    def _clear_plots(self):
        plot_dir = app.config['PLOT_FOLDER']
        if os.path.exists(plot_dir):
            for f in os.listdir(plot_dir):
                if f.endswith('.png'):
                    try:
                        os.unlink(os.path.join(plot_dir, f))
                    except Exception:
                        pass

    def load_dataset(self, filepath, original_filename):
        self.reset()
        self.filename = original_filename

        if not os.path.exists(filepath):
            raise ValueError("File does not exist on server.")
        if os.path.getsize(filepath) == 0:
            raise ValueError("Uploaded CSV file is completely empty (0 bytes).")

        try:
            self.df_raw = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Could not parse CSV file: {str(e)}")

        if self.df_raw.empty:
            raise ValueError("The dataset contains zero rows.")
        if self.df_raw.shape[1] < 2:
            raise ValueError("The dataset must have at least 2 columns (features and a target).")

        self.all_columns = list(self.df_raw.columns)
        rows, cols = self.df_raw.shape
        missing_total = int(self.df_raw.isnull().sum().sum())
        duplicate_rows = int(self.df_raw.duplicated().sum())

        column_info = []
        column_unique_map = {}
        for col in self.df_raw.columns:
            missing_cnt = int(self.df_raw[col].isnull().sum())
            unique_cnt = int(self.df_raw[col].nunique(dropna=True))
            is_num = bool(pd.api.types.is_numeric_dtype(self.df_raw[col]))

            # Check unique values & numeric status
            non_null_vals = self.df_raw[col].dropna().unique()
            is_all_num = is_num or (len(non_null_vals) > 0 and all(is_numeric_value(v) for v in non_null_vals))
            norm_uniques = [normalize_class_label(v) for v in sorted(non_null_vals, key=lambda x: (float(x) if is_numeric_value(x) else str(x)))]

            column_unique_map[col] = {
                'is_numeric': bool(is_all_num),
                'unique_count': unique_cnt,
                'unique_values': norm_uniques[:20]
            }

            column_info.append({
                'name': col,
                'type': str(self.df_raw[col].dtype),
                'missing': missing_cnt,
                'non_null': rows - missing_cnt,
                'unique': unique_cnt,
                'is_numeric': is_num
            })

        self.column_unique_map = column_unique_map
        sample_records = self.df_raw.head(5).replace({np.nan: "NaN"}).to_dict(orient='records')

        self.summary = {
            'filename': self.filename,
            'rows': rows,
            'cols': cols,
            'missing_values': missing_total,
            'duplicate_rows': duplicate_rows,
            'column_info': column_info,
            'sample_records': sample_records
        }

        # Check if pre-processing is required
        reasons = []
        if missing_total > 0:
            reasons.append(f"Missing values detected ({missing_total} empty cells require imputation).")
        if duplicate_rows > 0:
            reasons.append(f"Duplicate rows detected ({duplicate_rows} duplicate samples require deduplication).")
        has_categorical = any(not c['is_numeric'] for c in column_info)
        if has_categorical:
            reasons.append("Categorical columns detected (require numerical encoding).")

        self.preprocessing_reasons = reasons
        self.preprocessing_required = len(reasons) > 0

    def configure_features(self, target_col, selected_features=None, class_display_names=None):
        if self.df_raw is None:
            raise ValueError("No dataset loaded.")
        if target_col not in self.df_raw.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        self.target_col = target_col

        if not selected_features:
            selected_features = [c for c in self.df_raw.columns if c != target_col]
        else:
            selected_features = [c for c in selected_features if c in self.df_raw.columns and c != target_col]

        if not selected_features:
            raise ValueError("You must select at least one feature column for classification.")

        self.feature_cols = selected_features

        df_target_valid = self.df_raw.dropna(subset=[target_col]).copy()
        if len(df_target_valid) < 10:
            raise ValueError("Too few records with valid target values to perform classification.")

        unique_targets_raw = df_target_valid[target_col].unique()
        if len(unique_targets_raw) < 2:
            raise ValueError(f"Target '{target_col}' has only 1 unique class ({unique_targets_raw[0]}). Classification requires at least 2 classes.")

        # Determine if target is numeric
        self.target_is_numeric = bool(
            pd.api.types.is_numeric_dtype(df_target_valid[target_col]) or
            all(is_numeric_value(v) for v in unique_targets_raw)
        )

        # Sorted unique normalized class representations
        if self.target_is_numeric:
            sorted_raw_targets = sorted(unique_targets_raw, key=lambda v: float(v))
        else:
            sorted_raw_targets = sorted(unique_targets_raw, key=lambda v: str(v))
        self.raw_target_classes = [normalize_class_label(v) for v in sorted_raw_targets]

        # Populate or update display names
        if class_display_names:
            for k, v in class_display_names.items():
                norm_k = normalize_class_label(k)
                if str(v).strip():
                    self.class_display_names[norm_k] = str(v).strip()
        else:
            for c in self.raw_target_classes:
                if c not in self.class_display_names:
                    self.class_display_names[c] = c

        self.numeric_features = [f for f in self.feature_cols if pd.api.types.is_numeric_dtype(df_target_valid[f])]
        self.categorical_features = [f for f in self.feature_cols if f not in self.numeric_features]

        self.df_clean = df_target_valid
        self._update_target_distribution(df_target_valid)

    def _update_target_distribution(self, df=None):
        if df is None:
            df = self.df_clean if self.df_clean is not None else self.df_raw
        if df is None or self.target_col is None:
            return

        df_target_valid = df.dropna(subset=[self.target_col]).copy()
        val_counts = df_target_valid[self.target_col].value_counts()
        total_valid = len(df_target_valid)
        dist_items = []
        for cls_val, cnt in val_counts.items():
            norm_cls = normalize_class_label(cls_val)
            disp_name = self.class_display_names.get(norm_cls, norm_cls)
            dist_items.append({
                'class': norm_cls,
                'display_name': disp_name,
                'count': int(cnt),
                'percentage': round((cnt / total_valid) * 100, 2)
            })

        if self.raw_target_classes:
            dist_items.sort(key=lambda d: self.raw_target_classes.index(d['class']) if d['class'] in self.raw_target_classes else 999)
        self.target_distribution = dist_items

        # Target distribution plot
        plt.figure(figsize=(6, 3.8))
        palette = sns.color_palette("mako", len(dist_items))
        bar_labels = [d['display_name'] for d in dist_items]
        bar_counts = [d['count'] for d in dist_items]
        bars = plt.bar(bar_labels, bar_counts, color=palette, width=0.45)
        plt.title(f'Target Distribution: {self.target_col}', fontsize=11, fontweight='bold', pad=10)
        plt.xlabel('Target Class', fontsize=10, fontweight='bold')
        plt.ylabel('Count', fontsize=10, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., h + max(1, 0.01 * total_valid),
                     f'{int(h)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()

        plot_name = f'target_dist_{int(time.time())}.png'
        plot_path = os.path.join(app.config['PLOT_FOLDER'], plot_name)
        plt.savefig(plot_path, dpi=120)
        plt.close()
        self.target_plot_url = f'plots/{plot_name}'

    def execute_preprocessing(self, remove_duplicates=True, imputation_strategy='median', use_standardization=True):
        """
        Executes missing value imputation using Scikit-Learn SimpleImputer,
        removes duplicate rows if requested, encodes categorical variables,
        and fits Scikit-Learn StandardScaler for feature standardization.
        """
        if self.df_clean is None:
            raise ValueError("Features must be configured before preprocessing.")

        df = self.df_clean.copy()
        dups_removed = 0
        if remove_duplicates:
            before_len = len(df)
            df = df.drop_duplicates()
            dups_removed = before_len - len(df)

        # 1. Target Encoding with Custom Display Names
        self.target_encoder = CustomLabelEncoder()
        normalized_targets = df[self.target_col].apply(normalize_class_label)
        y = self.target_encoder.fit_transform(normalized_targets)
        self.target_encoder.set_display_names(self.class_display_names)

        # 2. Imputation with Scikit-learn SimpleImputer
        X_df = pd.DataFrame(index=df.index)

        # Numeric features imputation
        if self.numeric_features:
            num_sub = df[self.numeric_features].copy()
            self.numeric_imputer = SimpleImputer(strategy=imputation_strategy)
            imputed_num = self.numeric_imputer.fit_transform(num_sub)
            for idx, col in enumerate(self.numeric_features):
                X_df[col] = imputed_num[:, idx]
                self.feature_medians[col] = float(np.median(imputed_num[:, idx]))
                self.feature_min_max[col] = {
                    'min': round(float(np.min(imputed_num[:, idx])), 2),
                    'max': round(float(np.max(imputed_num[:, idx])), 2),
                    'mean': round(float(np.mean(imputed_num[:, idx])), 2)
                }

        # Categorical features imputation & label encoding
        self.feature_encoders = {}
        if self.categorical_features:
            cat_sub = df[self.categorical_features].astype(str).copy()
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            imputed_cat = self.categorical_imputer.fit_transform(cat_sub)
            for idx, col in enumerate(self.categorical_features):
                col_vals = imputed_cat[:, idx]
                enc = CustomLabelEncoder()
                X_df[col] = enc.fit_transform(col_vals)
                self.feature_encoders[col] = enc
                self.feature_modes[col] = str(enc.classes_[0]) if enc.classes_ else "Unknown"

        # Ensure consistent column ordering
        X_df = X_df[self.feature_cols]

        # 3. Train-Test Split (stratified)
        self.test_size = 0.2
        X_train_raw, X_test_raw, y_train, y_test = custom_train_test_split(
            X_df.values, y, test_size=self.test_size, random_state=42, stratify=True
        )

        # 4. Feature Standardization using Scikit-Learn StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.use_standardization = bool(use_standardization)
        self.preprocessing_completed = True

        return {
            'duplicates_removed': dups_removed,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'features_count': len(self.feature_cols),
            'use_standardization': self.use_standardization
        }

    def train_knn(self, max_k=40, use_standardization=None):
        """
        Runs KNN workflow:
        1. Evaluates K=1 baseline.
        2. Computes error rate across K=1..max_k.
        3. Plots K vs Error Rate (Elbow plot).
        4. Detects elbow point K.
        5. Evaluates optimal elbow K model and builds side-by-side comparison.
        """
        if not self.preprocessing_completed:
            raise ValueError("Pre-processing must be completed before model training.")

        if use_standardization is not None:
            self.use_standardization = bool(use_standardization)

        self.selected_classifier = 'knn'
        max_k = min(int(max_k), len(self.y_train) - 1)
        self.max_k = max(5, max_k)

        # Select data representation based on standardization choice
        X_tr = self.X_train_scaled if self.use_standardization else self.X_train_raw
        X_te = self.X_test_scaled if self.use_standardization else self.X_test_raw
        classes = self.target_encoder.classes_
        display_classes = [self.target_encoder.get_display_name(c) for c in classes]
        labels = list(range(len(classes)))

        # -------------------------------------------------------------
        # STEP A: Train KNN with K = 1
        # -------------------------------------------------------------
        knn_1 = CustomKNNClassifier(n_neighbors=1)
        knn_1.fit(X_tr, self.y_train)
        y_pred_1 = knn_1.predict(X_te)
        metrics_1 = custom_metrics_evaluation(self.y_test, y_pred_1, labels)

        # Confusion Matrix Heatmap for K=1 with Human-Readable Display Labels
        plt.figure(figsize=(5.2, 4.2))
        sns.heatmap(
            metrics_1['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=display_classes,
            yticklabels=display_classes,
            cbar=False, annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title('Confusion Matrix (KNN: K=1)', fontsize=11, fontweight='bold', pad=10)
        plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
        plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
        plt.tight_layout()
        cm1_name = f'cm_k1_{int(time.time())}.png'
        cm1_path = os.path.join(app.config['PLOT_FOLDER'], cm1_name)
        plt.savefig(cm1_path, dpi=120)
        plt.close()
        self.k1_cm_plot_url = f'plots/{cm1_name}'

        self.k1_results = {
            'k': 1,
            'accuracy': metrics_1['accuracy'],
            'error_rate': metrics_1['error_rate'],
            'error_rate_decimal': metrics_1['error_rate_decimal'],
            'precision': metrics_1['precision'],
            'recall': metrics_1['recall'],
            'f1_score': metrics_1['f1_score'],
            'confusion_matrix': metrics_1['confusion_matrix'],
            'classes': display_classes,
            'raw_classes': [str(c) for c in classes],
            'standardized': self.use_standardization
        }

        # -------------------------------------------------------------
        # STEP B: Finding Optimal Value of K (K = 1 to max_k)
        # -------------------------------------------------------------
        k_values = list(range(1, self.max_k + 1))
        error_rates = []
        accuracies = []

        for k in k_values:
            clf_k = CustomKNNClassifier(n_neighbors=k)
            clf_k.fit(X_tr, self.y_train)
            preds_k = clf_k.predict(X_te)
            m_k = custom_metrics_evaluation(self.y_test, preds_k, labels)
            error_rates.append(m_k['error_rate_decimal'])
            accuracies.append(m_k['accuracy'])

        # Store results for user to visually inspect the elbow point
        self.selected_elbow_k = None
        self.optimal_k_results = None
        self.comparison_k1_vs_optimal = None

        self.elbow_results = {
            'k_values': k_values,
            'error_rates': [round(e * 100, 2) for e in error_rates],
            'error_rates_decimal': [round(e, 4) for e in error_rates],
            'accuracies': accuracies,
            'k1_error': round(error_rates[0] * 100, 2)
        }

        # Step C: Plot the K vs Error Rate curve without automated elbow guessing
        self._generate_elbow_plot(k_values, error_rates, selected_k=None)

        # Baseline K=1 is initial active model until user selects an elbow point K
        self.active_model = knn_1
        self.active_model_name = 'KNN (Baseline K=1)'

    def _generate_elbow_plot(self, k_values, error_rates, selected_k=None):
        """Plots 'K' vs. Error Rate curve. Highlights the user-selected elbow point if chosen."""
        plt.figure(figsize=(9, 4.8))
        plt.plot(k_values, error_rates, color='#4338ca', linestyle='-', linewidth=2.2,
                 marker='o', markersize=6, markerfacecolor='#3b82f6', markeredgecolor='#1e1b4b',
                 markeredgewidth=1.2, label='Error Rate ($E = 1 - Accuracy$)')

        # Baseline K=1 marker
        plt.scatter([1], [error_rates[0]], color='#f59e0b', s=110, zorder=4, label='Baseline K=1')

        # Highlight user-selected elbow point if available
        if selected_k is not None and selected_k in k_values:
            sel_idx = k_values.index(selected_k)
            sel_err = error_rates[sel_idx]
            plt.scatter([selected_k], [sel_err], color='#10b981', s=170, zorder=5, edgecolor='#064e3b', linewidth=2.5)
            plt.annotate(
                f'User-Selected Elbow (K={selected_k})\nError: {round(sel_err * 100, 1)}%',
                xy=(selected_k, sel_err),
                xytext=(selected_k + max(2, self.max_k * 0.08), sel_err + max(0.03, 0.08 * (max(error_rates) - min(error_rates) + 0.05))),
                arrowprops=dict(facecolor='#10b981', shrink=0.08, width=2, headwidth=7),
                fontsize=10, fontweight='bold', color='#065f46',
                bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#10b981", lw=1.5)
            )

        plt.title("'K' vs. Error Rate Curve (Inspect for Elbow Bend)", fontsize=13, fontweight='bold', pad=14)
        plt.xlabel("Values of K (Neighbors)", fontsize=11, fontweight='bold')
        plt.ylabel("Classification Error Rate", fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(np.arange(1, self.max_k + 1, max(1, self.max_k // 15)))
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()

        elbow_name = f'elbow_curve_{int(time.time())}.png'
        elbow_path = os.path.join(app.config['PLOT_FOLDER'], elbow_name)
        plt.savefig(elbow_path, dpi=130)
        plt.close()
        self.elbow_plot_url = f'plots/{elbow_name}'

    def evaluate_optimal_knn(self, chosen_k):
        """Evaluates KNN with the user-selected elbow K and updates comparison with K=1."""
        self.selected_elbow_k = int(chosen_k)
        X_tr = self.X_train_scaled if self.use_standardization else self.X_train_raw
        X_te = self.X_test_scaled if self.use_standardization else self.X_test_raw
        classes = self.target_encoder.classes_
        display_classes = [self.target_encoder.get_display_name(c) for c in classes]
        labels = list(range(len(classes)))

        knn_opt = CustomKNNClassifier(n_neighbors=self.selected_elbow_k)
        knn_opt.fit(X_tr, self.y_train)
        preds_opt = knn_opt.predict(X_te)
        metrics_opt = custom_metrics_evaluation(self.y_test, preds_opt, labels)

        # Confusion Matrix Heatmap for User-Selected Elbow K
        plt.figure(figsize=(5.2, 4.2))
        sns.heatmap(
            metrics_opt['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
            xticklabels=display_classes,
            yticklabels=display_classes,
            cbar=False, annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title(f'Confusion Matrix (Selected Elbow K={self.selected_elbow_k})', fontsize=11, fontweight='bold', pad=10)
        plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
        plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
        plt.tight_layout()
        cm_opt_name = f'cm_kopt_{int(time.time())}.png'
        cm_opt_path = os.path.join(app.config['PLOT_FOLDER'], cm_opt_name)
        plt.savefig(cm_opt_path, dpi=120)
        plt.close()
        self.optimal_cm_plot_url = f'plots/{cm_opt_name}'

        self.optimal_k_results = {
            'k': self.selected_elbow_k,
            'accuracy': metrics_opt['accuracy'],
            'error_rate': metrics_opt['error_rate'],
            'error_rate_decimal': metrics_opt['error_rate_decimal'],
            'precision': metrics_opt['precision'],
            'recall': metrics_opt['recall'],
            'f1_score': metrics_opt['f1_score'],
            'confusion_matrix': metrics_opt['confusion_matrix'],
            'classes': display_classes,
            'raw_classes': [str(c) for c in classes],
            'standardized': self.use_standardization
        }

        # Side-by-side comparison metrics
        k1_acc = self.k1_results['accuracy']
        kopt_acc = self.optimal_k_results['accuracy']
        k1_err = self.k1_results['error_rate']
        kopt_err = self.optimal_k_results['error_rate']
        k1_f1 = self.k1_results['f1_score']
        kopt_f1 = self.optimal_k_results['f1_score']

        self.comparison_k1_vs_optimal = {
            'delta_accuracy': round(kopt_acc - k1_acc, 2),
            'delta_error_rate': round(k1_err - kopt_err, 2),
            'delta_f1': round(kopt_f1 - k1_f1, 2),
            'k1_acc': k1_acc,
            'kopt_acc': kopt_acc,
            'k1_err': k1_err,
            'kopt_err': kopt_err,
            'k1_prec': self.k1_results['precision'],
            'kopt_prec': self.optimal_k_results['precision'],
            'k1_rec': self.k1_results['recall'],
            'kopt_rec': self.optimal_k_results['recall'],
            'k1_f1': k1_f1,
            'kopt_f1': kopt_f1,
            'optimal_k': self.selected_elbow_k
        }

        # Update plot to show the user's selected point
        if self.elbow_results:
            self._generate_elbow_plot(
                self.elbow_results['k_values'],
                self.elbow_results['error_rates_decimal'],
                selected_k=self.selected_elbow_k
            )

        # Update active model
        self.active_model = knn_opt
        self.active_model_name = f'KNN (Selected Elbow K={self.selected_elbow_k})'

    def train_bayes(self, use_standardization=None):
        """
        Trains the Bayes Classifier (PatternClassifiers) from first principles,
        evaluates metrics, records priors and Gaussian parameters.
        """
        if not self.preprocessing_completed:
            raise ValueError("Pre-processing must be completed before model training.")

        if use_standardization is not None:
            self.use_standardization = bool(use_standardization)

        self.selected_classifier = 'bayes'
        X_tr = self.X_train_scaled if self.use_standardization else self.X_train_raw
        X_te = self.X_test_scaled if self.use_standardization else self.X_test_raw
        classes = self.target_encoder.classes_
        display_classes = [self.target_encoder.get_display_name(c) for c in classes]
        labels = list(range(len(classes)))

        bayes_clf = PatternClassifiers()
        bayes_clf.fit(X_tr, self.y_train)
        y_pred = bayes_clf.predict(X_te)
        metrics = custom_metrics_evaluation(self.y_test, y_pred, labels)

        # Heatmap
        plt.figure(figsize=(5.2, 4.2))
        sns.heatmap(
            metrics['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
            xticklabels=display_classes,
            yticklabels=display_classes,
            cbar=False, annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title('Confusion Matrix (Bayes Classifier)', fontsize=11, fontweight='bold', pad=10)
        plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
        plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
        plt.tight_layout()
        bayes_cm_name = f'cm_bayes_{int(time.time())}.png'
        bayes_cm_path = os.path.join(app.config['PLOT_FOLDER'], bayes_cm_name)
        plt.savefig(bayes_cm_path, dpi=120)
        plt.close()
        self.bayes_cm_plot_url = f'plots/{bayes_cm_name}'

        # Format learned parameters
        learned = bayes_clf.displayParam()
        formatted_params = []
        for c_idx in sorted(learned['priors'].keys()):
            c_name = self.target_encoder.get_display_name(c_idx)
            raw_c = self.target_encoder.inverse_transform([c_idx])[0]
            feat_stats = []
            for i, f_col in enumerate(self.feature_cols):
                feat_key = f'f_{i}' if f'f_{i}' in learned['parameters'][c_idx] else f_col
                if feat_key in learned['parameters'][c_idx]:
                    feat_stats.append({
                        'feature': f_col,
                        'mean': learned['parameters'][c_idx][feat_key]['mean'],
                        'variance': learned['parameters'][c_idx][feat_key]['variance']
                    })
            formatted_params.append({
                'class_idx': c_idx,
                'class_name': c_name,
                'raw_name': raw_c,
                'prior': round(learned['priors'][c_idx], 4),
                'features': feat_stats
            })

        self.bayes_results = {
            'accuracy': metrics['accuracy'],
            'error_rate': metrics['error_rate'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classes': display_classes,
            'raw_classes': [str(c) for c in classes],
            'learned_params': formatted_params,
            'standardized': self.use_standardization
        }

        self.active_model = bayes_clf
        self.active_model_name = 'Bayes Classifier (Gaussian Naive Bayes)'

    def update_class_names(self, rename_dict):
        """Updates human-readable class names across target distributions, plots, confusion matrices, and active predictions."""
        for k, v in rename_dict.items():
            norm_k = normalize_class_label(k)
            if str(v).strip():
                self.class_display_names[norm_k] = str(v).strip()

        if self.target_encoder:
            self.target_encoder.set_display_names(self.class_display_names)

        # 1. Update distribution plot & records
        self._update_target_distribution()

        # 2. Update K=1 confusion matrix if available
        if self.k1_results and self.target_encoder:
            classes = self.target_encoder.classes_
            disp_classes = [self.target_encoder.get_display_name(c) for c in classes]
            self.k1_results['classes'] = disp_classes
            plt.figure(figsize=(5.2, 4.2))
            sns.heatmap(
                self.k1_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=disp_classes,
                yticklabels=disp_classes,
                cbar=False, annot_kws={"size": 12, "weight": "bold"}
            )
            plt.title('Confusion Matrix (KNN: K=1)', fontsize=11, fontweight='bold', pad=10)
            plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
            plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
            plt.tight_layout()
            cm1_name = f'cm_k1_{int(time.time())}.png'
            cm1_path = os.path.join(app.config['PLOT_FOLDER'], cm1_name)
            plt.savefig(cm1_path, dpi=120)
            plt.close()
            self.k1_cm_plot_url = f'plots/{cm1_name}'

        # 3. Update Optimal K confusion matrix if available
        if self.optimal_k_results and self.target_encoder:
            classes = self.target_encoder.classes_
            disp_classes = [self.target_encoder.get_display_name(c) for c in classes]
            self.optimal_k_results['classes'] = disp_classes
            plt.figure(figsize=(5.2, 4.2))
            sns.heatmap(
                self.optimal_k_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=disp_classes,
                yticklabels=disp_classes,
                cbar=False, annot_kws={"size": 12, "weight": "bold"}
            )
            plt.title(f'Confusion Matrix (Selected Elbow K={self.selected_elbow_k})', fontsize=11, fontweight='bold', pad=10)
            plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
            plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
            plt.tight_layout()
            cm_opt_name = f'cm_kopt_{int(time.time())}.png'
            cm_opt_path = os.path.join(app.config['PLOT_FOLDER'], cm_opt_name)
            plt.savefig(cm_opt_path, dpi=120)
            plt.close()
            self.optimal_cm_plot_url = f'plots/{cm_opt_name}'

        # 4. Update Bayes confusion matrix & learned_params if available
        if self.bayes_results and self.target_encoder:
            classes = self.target_encoder.classes_
            disp_classes = [self.target_encoder.get_display_name(c) for c in classes]
            self.bayes_results['classes'] = disp_classes
            plt.figure(figsize=(5.2, 4.2))
            sns.heatmap(
                self.bayes_results['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=disp_classes,
                yticklabels=disp_classes,
                cbar=False, annot_kws={"size": 12, "weight": "bold"}
            )
            plt.title('Confusion Matrix (Bayes Classifier)', fontsize=11, fontweight='bold', pad=10)
            plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
            plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
            plt.tight_layout()
            bayes_cm_name = f'cm_bayes_{int(time.time())}.png'
            bayes_cm_path = os.path.join(app.config['PLOT_FOLDER'], bayes_cm_name)
            plt.savefig(bayes_cm_path, dpi=120)
            plt.close()
            self.bayes_cm_plot_url = f'plots/{bayes_cm_name}'

            if 'learned_params' in self.bayes_results:
                for lp in self.bayes_results['learned_params']:
                    lp['class_name'] = self.target_encoder.get_display_name(lp['class_idx'])

        # 5. Update last prediction labels if present
        if self.last_prediction and self.target_encoder:
            pred_idx = self.last_prediction['predicted_class_code']
            raw_c = self.last_prediction.get('predicted_class_raw', str(pred_idx))
            new_label = self.target_encoder.get_display_name(raw_c)
            self.last_prediction['predicted_class_label'] = new_label
            for p in self.last_prediction.get('probabilities', []):
                p['class_label'] = self.target_encoder.get_display_name(p.get('class_raw', p['class_code']))
            if self.last_prediction.get('neighbors'):
                for nb in self.last_prediction['neighbors']:
                    nb['class_label'] = self.target_encoder.get_display_name(nb.get('class_raw', nb['class_code']))

    def predict_new_instance(self, input_dict):
        """
        Validates new user query inputs, encodes & standardizes them,
        predicts using active model, and provides a descriptive breakdown.
        """
        if self.active_model is None:
            raise ValueError("No model is currently trained. Please train a classifier first.")

        # Parse & impute inputs
        vector_values = []
        for col in self.feature_cols:
            raw_val = input_dict.get(col, "").strip()
            if col in self.numeric_features:
                if raw_val == "":
                    val = self.feature_medians.get(col, 0.0)
                else:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        raise ValueError(f"Feature '{col}' requires a numeric value, got '{raw_val}'.")
                vector_values.append(val)
            else:
                if raw_val == "":
                    raw_val = self.feature_modes.get(col, "Unknown")
                enc = self.feature_encoders.get(col)
                if enc:
                    num_val = float(enc.transform([raw_val])[0])
                else:
                    num_val = 0.0
                vector_values.append(num_val)

        x_raw = np.array(vector_values, dtype=np.float64).reshape(1, -1)

        # Scale if model expects standardized inputs
        if self.use_standardization and self.scaler:
            x_model = self.scaler.transform(x_raw)
        else:
            x_model = x_raw

        classes = self.target_encoder.classes_

        if isinstance(self.active_model, CustomKNNClassifier):
            explanation = self.active_model.explain_instance(x_model[0])
            pred_idx = explanation['predicted_class']
            pred_raw = self.target_encoder.inverse_transform([pred_idx])[0]
            pred_label = self.target_encoder.get_display_name(pred_raw)

            # Neighbor details with original target labels
            neighbor_list = []
            for nb in explanation['neighbors']:
                nb_raw = self.target_encoder.inverse_transform([nb['class']])[0]
                nb_lbl = self.target_encoder.get_display_name(nb_raw)
                neighbor_list.append({
                    'train_index': nb['train_index'],
                    'distance': nb['distance'],
                    'class_code': nb['class'],
                    'class_raw': str(nb_raw),
                    'class_label': str(nb_lbl)
                })

            # Vote breakdown
            proportions = explanation['probabilities']
            prob_list = []
            for c_idx, c_raw in enumerate(classes):
                p = round(proportions.get(c_idx, 0.0) * 100, 2)
                disp_lbl = self.target_encoder.get_display_name(c_raw)
                prob_list.append({
                    'class_code': c_idx,
                    'class_raw': str(c_raw),
                    'class_label': str(disp_lbl),
                    'probability': p,
                    'is_winner': (c_idx == pred_idx)
                })

            confidence = max(p['probability'] for p in prob_list) if prob_list else 100.0

            if pred_label != str(pred_raw):
                class_desc = f"'{pred_label}' (Raw: {pred_raw}, Class {pred_idx})"
            else:
                class_desc = f"'{pred_label}' (Class {pred_idx})"

            descriptive_text = (
                f"Prediction performed via K-Nearest Neighbors (K={self.active_model.n_neighbors}). "
                f"Among the {self.active_model.n_neighbors} nearest samples in feature space, "
                f"{int(proportions.get(pred_idx, 0.0) * self.active_model.n_neighbors)} neighbor(s) "
                f"voted for {class_desc}. "
                f"Feature standardization was {'applied (StandardScaler)' if self.use_standardization else 'not applied (Raw features)'}."
            )

            self.last_prediction = {
                'algorithm': f'K-Nearest Neighbors (K={self.active_model.n_neighbors})',
                'predicted_class_code': int(pred_idx),
                'predicted_class_raw': str(pred_raw),
                'predicted_class_label': str(pred_label),
                'confidence': confidence,
                'probabilities': prob_list,
                'neighbors': neighbor_list,
                'descriptive_text': descriptive_text,
                'input_values': input_dict
            }

        elif isinstance(self.active_model, PatternClassifiers):
            pred_idx = int(self.active_model.predict_single(x_model[0]))
            pred_raw = self.target_encoder.inverse_transform([pred_idx])[0]
            pred_label = self.target_encoder.get_display_name(pred_raw)
            probs_dict = self.active_model.predict_proba_single(x_model[0])

            prob_list = []
            for c_idx, c_raw in enumerate(classes):
                p = round(probs_dict.get(c_idx, 0.0) * 100, 2)
                disp_lbl = self.target_encoder.get_display_name(c_raw)
                prob_list.append({
                    'class_code': c_idx,
                    'class_raw': str(c_raw),
                    'class_label': str(disp_lbl),
                    'probability': p,
                    'is_winner': (c_idx == pred_idx)
                })

            confidence = max(p['probability'] for p in prob_list) if prob_list else 100.0

            if pred_label != str(pred_raw):
                class_desc = f"'{pred_label}' (Raw: {pred_raw}, Class {pred_idx})"
            else:
                class_desc = f"'{pred_label}' (Class {pred_idx})"

            descriptive_text = (
                f"Prediction performed via Bayes Classifier (Gaussian Naive Bayes). "
                f"The class with maximum posterior probability is {class_desc} "
                f"with {confidence}% estimated posterior probability. "
                f"Feature standardization was {'applied' if self.use_standardization else 'not applied'}."
            )

            self.last_prediction = {
                'algorithm': 'Bayes Classifier (Gaussian Naive Bayes)',
                'predicted_class_code': int(pred_idx),
                'predicted_class_raw': str(pred_raw),
                'predicted_class_label': str(pred_label),
                'confidence': confidence,
                'probabilities': prob_list,
                'neighbors': None,
                'descriptive_text': descriptive_text,
                'input_values': input_dict
            }


app_instance = ClassifiersApp()


# =====================================================================
# FLASK WEB ROUTES
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
            flash("Invalid format. Please upload a valid .csv file.", "error")
            return redirect(url_for('index', _anchor='step-upload'))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            app_instance.load_dataset(save_path, filename)
            flash(f"Dataset '{filename}' successfully loaded ({app_instance.summary['rows']} rows, {app_instance.summary['cols']} columns). Previous upload data flushed.", "success")
        except Exception as e:
            flash(f"Failed to load dataset: {str(e)}", "error")
            return redirect(url_for('index', _anchor='step-upload'))

    elif 'sample_dataset' in request.form and request.form.get('sample_dataset'):
        sample_name = request.form.get('sample_dataset')
        sample_path = os.path.join(app.config['DATASET_FOLDER'], sample_name)
        if os.path.exists(sample_path):
            try:
                app_instance.load_dataset(sample_path, sample_name)
                flash(f"Benchmark dataset '{sample_name}' loaded successfully ({app_instance.summary['rows']} rows, {app_instance.summary['cols']} columns).", "success")
            except Exception as e:
                flash(f"Error loading sample dataset: {str(e)}", "error")
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
            use_standardization=use_standardization
        )
        flash(
            f"Pre-processing completed: {res['train_samples']} training samples, "
            f"{res['test_samples']} test samples, {res['features_count']} features. "
            f"Standardization: {'Enabled (StandardScaler)' if res['use_standardization'] else 'Disabled'}.",
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
    classifier_choice = request.form.get('classifier_choice', 'knn')
    use_standardization = 'use_standardization' in request.form
    max_k = int(request.form.get('max_k', 40))

    try:
        if classifier_choice == 'knn':
            app_instance.train_knn(max_k=max_k, use_standardization=use_standardization)
            flash(
                f"KNN evaluated for K=1 (Baseline Error: {app_instance.k1_results['error_rate']}%) and across K=1 to {app_instance.max_k}. "
                f"Please inspect the 'K' vs. Error Rate plot below and select your chosen elbow point K to compare with K=1.",
                "info"
            )
        else:
            app_instance.train_bayes(use_standardization=use_standardization)
            flash(
                f"Bayes Classifier Trained: Accuracy: {app_instance.bayes_results['accuracy']}%, "
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
            f"Selected Elbow Point K={app_instance.selected_elbow_k} evaluated! "
            f"Accuracy: {app_instance.optimal_k_results['accuracy']}%, Error Rate: {app_instance.optimal_k_results['error_rate']}%. "
            f"Comparison with K=1 is now available below.",
            "success"
        )
    except Exception as e:
        flash(f"Error evaluating chosen K: {str(e)}", "error")

    return redirect(url_for('index', _anchor='step-training'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for col in app_instance.feature_cols:
            input_data[col] = request.form.get(f'feat_{col}', '')

        app_instance.predict_new_instance(input_data)
        flash(
            f"Prediction successful: Predicted Class {app_instance.last_prediction['predicted_class_code']} "
            f"('{app_instance.last_prediction['predicted_class_label']}') with {app_instance.last_prediction['confidence']}% confidence.",
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
    Dedicated comparative analysis page comparing:
    1. KNN (K=1, Raw Features)
    2. KNN (K=1, Standardized Features)
    3. KNN (Optimal Elbow K, Standardized Features)
    4. Bayes Classifier (Gaussian Naive Bayes)
    """
    datasets_to_evaluate = [
        {'name': 'heart.csv', 'target': 'target', 'title': 'Heart Disease Dataset'},
        {'name': 'Titanic-Dataset.csv', 'target': 'Survived', 'title': 'Titanic Passenger Survival'},
        {'name': 'breast_cancer.csv', 'target': 'target', 'title': 'Breast Cancer Wisconsin'},
        {'name': 'Naive-Bayes-Classification-Data.csv', 'target': 'diabetes', 'title': 'Blood Pressure & Glucose Classification'}
    ]

    analysis_results = []
    dataset_dir = app.config['DATASET_FOLDER']

    for item in datasets_to_evaluate:
        fpath = os.path.join(dataset_dir, item['name'])
        if not os.path.exists(fpath):
            continue

        try:
            df = pd.read_csv(fpath).dropna(subset=[item['target']])
            df = df.drop_duplicates()
            y_col = item['target']
            feat_cols = [c for c in df.columns if c != y_col]

            # Impute & encode
            num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in feat_cols if c not in num_cols]

            X_df = pd.DataFrame(index=df.index)
            if num_cols:
                imp = SimpleImputer(strategy='median')
                imputed = imp.fit_transform(df[num_cols])
                for idx, c in enumerate(num_cols):
                    X_df[c] = imputed[:, idx]
            if cat_cols:
                imp_c = SimpleImputer(strategy='most_frequent')
                imputed_c = imp_c.fit_transform(df[cat_cols].astype(str))
                for idx, c in enumerate(cat_cols):
                    enc = CustomLabelEncoder()
                    X_df[c] = enc.fit_transform(imputed_c[:, idx])

            y_enc = CustomLabelEncoder()
            y = y_enc.fit_transform(df[y_col].astype(str))
            classes = y_enc.classes_
            labels = list(range(len(classes)))

            X_raw = X_df[feat_cols].values
            X_tr_raw, X_te_raw, y_tr, y_te = custom_train_test_split(X_raw, y, test_size=0.25, random_state=42, stratify=True)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_raw)
            X_te_s = scaler.transform(X_te_raw)

            # 1. KNN (K=1, Raw)
            knn_k1_raw = CustomKNNClassifier(n_neighbors=1).fit(X_tr_raw, y_tr)
            m_k1_raw = custom_metrics_evaluation(y_te, knn_k1_raw.predict(X_te_raw), labels)

            # 2. KNN (K=1, Scaled)
            knn_k1_s = CustomKNNClassifier(n_neighbors=1).fit(X_tr_s, y_tr)
            m_k1_s = custom_metrics_evaluation(y_te, knn_k1_s.predict(X_te_s), labels)

            # 3. KNN (Empirical Best K, Scaled)
            errs = []
            max_search_k = min(35, len(y_tr) - 1)
            for k in range(1, max_search_k + 1):
                clf = CustomKNNClassifier(n_neighbors=k).fit(X_tr_s, y_tr)
                errs.append(custom_metrics_evaluation(y_te, clf.predict(X_te_s), labels)['error_rate_decimal'])
            best_k = int(np.argmin(errs) + 1)
            knn_opt = CustomKNNClassifier(n_neighbors=best_k).fit(X_tr_s, y_tr)
            m_opt = custom_metrics_evaluation(y_te, knn_opt.predict(X_te_s), labels)

            # 4. Bayes Classifier
            bayes = PatternClassifiers()
            bayes.fit(X_tr_s, y_tr)
            m_bayes = custom_metrics_evaluation(y_te, bayes.predict(X_te_s), labels)

            analysis_results.append({
                'title': item['title'],
                'filename': item['name'],
                'samples': len(df),
                'features': len(feat_cols),
                'classes': len(classes),
                'elbow_k': best_k,
                'k1_raw': m_k1_raw,
                'k1_scaled': m_k1_s,
                'knn_optimal': m_opt,
                'bayes': m_bayes,
                'accuracy_gain': round(m_opt['accuracy'] - m_k1_raw['accuracy'], 2),
                'scaling_gain': round(m_k1_s['accuracy'] - m_k1_raw['accuracy'], 2),
                'error_reduction': round(m_k1_raw['error_rate'] - m_opt['error_rate'], 2)
            })
        except Exception as e:
            print(f"Error benchmark {item['name']}: {e}")

    return render_template(
        'performance_analysis.html',
        analysis_results=analysis_results,
        app_state=app_instance
    )


@app.route('/shutdown', methods=['POST'])
def shutdown():
    def stop_server():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=stop_server).start()
    return "Server is shutting down safely..."


if __name__ == '__main__':
    print("Starting Classifiers Studio (Assignment 07: KNN & Bayes Classifier)...")
    app.run(host='0.0.0.0', port=5000, debug=False)
