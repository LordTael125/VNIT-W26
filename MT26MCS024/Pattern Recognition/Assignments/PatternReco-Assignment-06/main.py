import os
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'Templates'),
    static_folder=os.path.join(BASE_DIR, 'Statics')
)
app.secret_key = 'pattern_recognition_assignment_06_custom_secret'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'Uploads')
app.config['DATASET_FOLDER'] = os.path.join(BASE_DIR, 'Dataset')
app.config['PLOT_FOLDER'] = os.path.join(BASE_DIR, 'Statics', 'plots')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)


# =====================================================================
# CUSTOM MACHINE LEARNING IMPLEMENTATIONS (NO PRE-DEFINED ML LIBRARIES)
# =====================================================================

class CustomLabelEncoder:
    """Custom Label Encoder implemented from scratch."""
    def __init__(self):
        self.classes_ = []
        self.mapping = {}
        self.inverse_mapping = {}

    def fit(self, values):
        unique_vals = sorted(list(set(str(v) for v in values)))
        self.classes_ = unique_vals
        self.mapping = {val: idx for idx, val in enumerate(unique_vals)}
        self.inverse_mapping = {idx: val for idx, val in enumerate(unique_vals)}
        return self

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
                encoded.append(0)  # Graceful fallback for unseen category
        return np.array(encoded)

    def inverse_transform(self, indices):
        return [self.inverse_mapping.get(int(idx), str(idx)) for idx in indices]


class PatternClassifiers:
    """
    Custom Naive Bayes Classifier implemented from first principles.
    Adheres to the Pattern Recognition Lab specifications (Assignments 03 & 04).
    """
    def __init__(self, eps=1e-9):
        self.classes = []
        self.features = []
        self.priors = {}
        self.parameters = {}  # {class: {feature: {'mean': val, 'variance': val}}}
        self.eps = eps

    def displayParam(self):
        """Returns the internal model parameters (priors and conditional distribution parameters)."""
        return {
            'priors': self.priors,
            'parameters': self.parameters
        }

    def bayesianClassifier(self, x, mean, variance):
        """
        Calculates the class-conditional Gaussian probability density:
        P(x | mean, variance) = (1 / sqrt(2 * pi * variance)) * exp(-(x - mean)^2 / (2 * variance))
        """
        if variance <= 0:
            variance = self.eps
        coefficient = 1.0 / np.sqrt(2.0 * np.pi * variance)
        exponent = np.exp(-((x - mean) ** 2) / (2.0 * variance))
        likelihood = coefficient * exponent
        return max(likelihood, 1e-300)

    def fit(self, featureDF, classifications):
        """
        Calculates prior probabilities P(C_k) and Gaussian parameters (mean, variance)
        for every feature conditioned on class C_k.
        """
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
                # Sample variance (ddof=1)
                var_val = float(X_c[f].var(ddof=1))
                if np.isnan(var_val) or var_val < self.eps:
                    var_val = self.eps
                self.parameters[c][f] = {
                    'mean': round(mean_val, 4),
                    'variance': round(var_val, 6)
                }

    def predict_single(self, row):
        """
        Predicts class label for a single query instance using Bayes' decision rule:
        C_pred = argmax_c [ log P(c) + sum_i log P(x_i | c) ]
        """
        log_probabilities = {}
        for c in self.classes:
            log_prob = np.log(max(self.priors[c], self.eps))
            for f in self.features:
                mean = self.parameters[c][f]['mean']
                variance = self.parameters[c][f]['variance']
                val = float(row[f])
                likelihood = self.bayesianClassifier(val, mean, variance)
                log_prob += np.log(likelihood)
            log_probabilities[c] = log_prob

        return max(log_probabilities, key=log_probabilities.get)

    def predict(self, X):
        """Predicts class labels for multiple query instances."""
        return [self.predict_single(row) for _, row in X.iterrows()]

    def predict_proba_single(self, row):
        """
        Computes normalized posterior probabilities for all classes using the
        numerically stable log-sum-exp formulation.
        """
        log_probs = []
        for c in self.classes:
            lp = np.log(max(self.priors[c], self.eps))
            for f in self.features:
                mean = self.parameters[c][f]['mean']
                variance = self.parameters[c][f]['variance']
                val = float(row[f])
                likelihood = self.bayesianClassifier(val, mean, variance)
                lp += np.log(likelihood)
            log_probs.append(lp)

        log_probs = np.array(log_probs)
        max_lp = np.max(log_probs)
        probs = np.exp(log_probs - max_lp)
        normalized_probs = probs / np.sum(probs)
        return {c: float(normalized_probs[i]) for i, c in enumerate(self.classes)}


def custom_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Custom train-test partitioning function implemented from scratch.
    Supports stratified random sampling without any external libraries.
    """
    rng = np.random.RandomState(random_state)
    y_list = list(y)
    classes = sorted(list(set(y_list)))

    train_indices = []
    test_indices = []

    if stratify:
        for c in classes:
            c_indices = [i for i, val in enumerate(y_list) if val == c]
            rng.shuffle(c_indices)
            n_test = int(round(len(c_indices) * test_size))
            test_indices.extend(c_indices[:n_test])
            train_indices.extend(c_indices[n_test:])
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        indices = list(range(len(y_list)))
        rng.shuffle(indices)
        n_test = int(round(len(y_list) * test_size))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = [y_list[i] for i in train_indices]
    y_test = [y_list[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def custom_confusion_matrix(y_true, y_pred, labels):
    """Computes a k x k confusion matrix from scratch."""
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = [[0 for _ in range(k)] for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    return cm


def custom_metrics_evaluation(y_true, y_pred, labels):
    """
    Calculates Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-Score
    from first principles without external metric libraries.
    """
    cm = custom_confusion_matrix(y_true, y_pred, labels)
    k = len(labels)
    total_samples = len(y_true)

    # Accuracy
    correct_samples = sum(cm[i][i] for i in range(k))
    accuracy = (correct_samples / total_samples) if total_samples > 0 else 0.0

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

        # Precision = TP / (TP + FP)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # F1 = 2 * (P * R) / (P + R)
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
        'precision': round(weighted_prec * 100, 2),
        'recall': round(weighted_rec * 100, 2),
        'f1_score': round(weighted_f1 * 100, 2),
        'confusion_matrix': cm
    }


# =====================================================================
# APPLICATION CONTROLLER
# =====================================================================

class BayesClassifierApp:
    def __init__(self):
        self.reset()

    def reset(self):
        """Completely clears state to isolate datasets from prior uploads."""
        self.filename = None
        self.df_raw = None
        self.df_clean = None

        self.summary = None
        self.all_columns = []

        self.target_col = None
        self.feature_cols = []
        self.numeric_features = []
        self.categorical_features = []
        self.usable_records = 0
        self.target_distribution = None
        self.target_plot_url = None

        # Custom Model State
        self.selected_algorithm = 'Custom Gaussian Bayes'
        self.test_size = 0.2
        self.model = None
        self.target_encoder = None
        self.feature_encoders = {}
        self.feature_medians = {}
        self.feature_modes = {}
        self.feature_min_max = {}
        self.training_status = None
        self.training_error = None

        self.eval_results = None
        self.confusion_matrix_plot_url = None
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
        """Loads and validates a CSV dataset."""
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
            raise ValueError("The dataset contains no rows.")

        if self.df_raw.shape[1] < 2:
            raise ValueError("The dataset must have at least 2 columns (features and a target).")

        self.all_columns = list(self.df_raw.columns)
        rows, cols = self.df_raw.shape
        missing_total = int(self.df_raw.isnull().sum().sum())
        duplicate_rows = int(self.df_raw.duplicated().sum())

        column_info = []
        for col in self.df_raw.columns:
            dtype_str = str(self.df_raw[col].dtype)
            missing_cnt = int(self.df_raw[col].isnull().sum())
            unique_cnt = int(self.df_raw[col].nunique(dropna=True))
            column_info.append({
                'name': col,
                'type': dtype_str,
                'missing': missing_cnt,
                'non_null': rows - missing_cnt,
                'unique': unique_cnt,
                'is_numeric': bool(pd.api.types.is_numeric_dtype(self.df_raw[col]))
            })

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

    def prepare_data(self, target_col, selected_features=None):
        """Prepares dataset for training: target selection, feature split, imputation, distribution."""
        if self.df_raw is None:
            raise ValueError("No dataset loaded.")

        if target_col not in self.df_raw.columns:
            raise ValueError(f"Selected target column '{target_col}' not found.")

        self.target_col = target_col

        if not selected_features:
            selected_features = [c for c in self.df_raw.columns if c != target_col]
        else:
            selected_features = [c for c in selected_features if c in self.df_raw.columns and c != target_col]

        if not selected_features:
            raise ValueError("You must select at least one feature for classification.")

        self.feature_cols = selected_features

        df_target_clean = self.df_raw.dropna(subset=[target_col]).copy()
        if len(df_target_clean) < 10:
            raise ValueError("Too few records with valid target values to perform classification.")

        unique_targets = df_target_clean[target_col].unique()
        if len(unique_targets) < 2:
            raise ValueError(f"Target column '{target_col}' has only 1 unique class ({unique_targets[0]}). Classification requires at least 2 classes.")

        self.usable_records = len(df_target_clean)

        self.numeric_features = [f for f in self.feature_cols if pd.api.types.is_numeric_dtype(df_target_clean[f])]
        self.categorical_features = [f for f in self.feature_cols if f not in self.numeric_features]

        # Target distribution
        val_counts = df_target_clean[target_col].value_counts()
        total_valid = len(df_target_clean)
        dist_items = []
        for cls_val, cnt in val_counts.items():
            dist_items.append({
                'class': str(cls_val),
                'count': int(cnt),
                'percentage': round((cnt / total_valid) * 100, 2)
            })

        self.target_distribution = dist_items

        # Target distribution chart
        plt.figure(figsize=(6, 4))
        palette = sns.color_palette("mako", len(val_counts))
        bars = plt.bar([str(x) for x in val_counts.index], val_counts.values, color=palette, edgecolor='none', width=0.5)
        plt.title(f'Target Distribution: {target_col}', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel(target_col, fontsize=10, fontweight='bold')
        plt.ylabel('Count', fontsize=10, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(1, 0.01 * total_valid),
                     f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        dist_plot_filename = f'target_dist_{int(time.time())}.png'
        dist_plot_path = os.path.join(app.config['PLOT_FOLDER'], dist_plot_filename)
        plt.savefig(dist_plot_path, dpi=120)
        plt.close()
        self.target_plot_url = f'plots/{dist_plot_filename}'

        self.df_clean = df_target_clean

        self.model = None
        self.eval_results = None
        self.confusion_matrix_plot_url = None
        self.last_prediction = None
        self.training_status = None
        self.training_error = None

    def train_and_evaluate(self, test_size=0.2):
        """Trains the custom PatternClassifiers model and evaluates performance metrics."""
        if self.df_clean is None or not self.target_col or not self.feature_cols:
            raise ValueError("Pre-training preparation must be completed before training.")

        self.test_size = float(test_size)

        try:
            df = self.df_clean.copy()

            # Target encoding using custom encoder
            self.target_encoder = CustomLabelEncoder()
            y = self.target_encoder.fit_transform(df[self.target_col].astype(str))
            classes_ = self.target_encoder.classes_

            # Process features: imputation & custom encoding
            self.feature_encoders = {}
            self.feature_medians = {}
            self.feature_modes = {}
            self.feature_min_max = {}
            X = pd.DataFrame(index=df.index)

            for col in self.feature_cols:
                if col in self.numeric_features:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    self.feature_medians[col] = float(median_val)
                    col_series = df[col].fillna(median_val)
                    X[col] = col_series.astype(float)
                    self.feature_min_max[col] = {
                        'min': float(col_series.min()),
                        'max': float(col_series.max())
                    }
                else:
                    mode_val = df[col].mode()
                    default_mode = str(mode_val[0]) if not mode_val.empty else "Unknown"
                    self.feature_modes[col] = default_mode
                    col_series = df[col].fillna(default_mode).astype(str)
                    enc = CustomLabelEncoder()
                    X[col] = enc.fit_transform(col_series)
                    self.feature_encoders[col] = enc

            # Custom stratified split
            X_train, X_test, y_train, y_test = custom_train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=True
            )

            # Fit Custom Naive Bayes Classifier (PatternClassifiers)
            classifier = PatternClassifiers()
            classifier.fit(X_train, y_train)
            self.model = classifier

            # Predictions on held-out test data
            y_pred = classifier.predict(X_test)

            # Evaluate Metrics from first principles
            int_labels = list(range(len(classes_)))
            metrics = custom_metrics_evaluation(y_test, y_pred, int_labels)

            cm = metrics['confusion_matrix']
            cm_table = {
                'labels': [str(c) for c in classes_],
                'matrix': cm
            }

            # Generate Confusion Matrix Heatmap Plot
            plt.figure(figsize=(5.5, 4.5))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(c) for c in classes_],
                yticklabels=[str(c) for c in classes_],
                cbar=False,
                annot_kws={"size": 12, "weight": "bold"}
            )
            plt.title('Confusion Matrix (Custom Bayes)', fontsize=12, fontweight='bold', pad=10)
            plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
            plt.ylabel('Actual Class', fontsize=10, fontweight='bold')
            plt.tight_layout()

            cm_plot_filename = f'cm_{int(time.time())}.png'
            cm_plot_path = os.path.join(app.config['PLOT_FOLDER'], cm_plot_filename)
            plt.savefig(cm_plot_path, dpi=120)
            plt.close()
            self.confusion_matrix_plot_url = f'plots/{cm_plot_filename}'

            # Model internal parameters from displayParam()
            learned_params = classifier.displayParam()
            formatted_params = []
            for c_idx in sorted(learned_params['priors'].keys()):
                c_name = self.target_encoder.inverse_transform([c_idx])[0]
                prior_val = round(learned_params['priors'][c_idx], 4)
                feat_stats = []
                for f in self.feature_cols:
                    feat_stats.append({
                        'feature': f,
                        'mean': learned_params['parameters'][c_idx][f]['mean'],
                        'variance': learned_params['parameters'][c_idx][f]['variance']
                    })
                formatted_params.append({
                    'class_idx': c_idx,
                    'class_name': c_name,
                    'prior': prior_val,
                    'features': feat_stats
                })

            self.eval_results = {
                'algorithm': 'Custom Gaussian Bayes (PatternClassifiers)',
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test)),
                'total_samples': int(len(X)),
                'test_percentage': int(self.test_size * 100),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'cm_table': cm_table,
                'classes': [str(c) for c in classes_],
                'learned_params': formatted_params
            }

            self.training_status = f"Successfully trained custom PatternClassifiers model on {len(X_train)} training records and evaluated on {len(X_test)} testing records."
            self.training_error = None

        except Exception as e:
            self.training_status = None
            self.training_error = f"Error during model training: {str(e)}"
            raise ValueError(self.training_error)

    def predict_new_instance(self, input_dict):
        """Predicts outcome for a new user query instance using custom PatternClassifiers."""
        if self.model is None:
            raise ValueError("No trained classification model available. Please train the model first.")

        row_data = {}
        for col in self.feature_cols:
            raw_val = input_dict.get(col, "").strip()
            if col in self.numeric_features:
                if raw_val == "":
                    val = self.feature_medians.get(col, 0.0)
                else:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        raise ValueError(f"Feature '{col}' expects a numeric value, got '{raw_val}'.")
                row_data[col] = val
            else:
                if raw_val == "":
                    val_str = self.feature_modes.get(col, "Unknown")
                else:
                    val_str = raw_val

                enc = self.feature_encoders.get(col)
                if enc is not None:
                    row_data[col] = enc.transform([val_str])[0]
                else:
                    row_data[col] = 0

        # Construct single-row Series
        row_series = pd.Series(row_data)

        pred_idx = self.model.predict_single(row_series)
        proba_dict = self.model.predict_proba_single(row_series)

        pred_label_raw = self.target_encoder.inverse_transform([pred_idx])[0]

        probabilities = []
        classes_ = self.target_encoder.classes_
        for idx, cls_name in enumerate(classes_):
            p = float(proba_dict.get(idx, 0.0))
            probabilities.append({
                'class_code': str(idx),
                'class_label': str(cls_name),
                'probability': round(p * 100, 2),
                'is_winner': (idx == pred_idx)
            })

        max_confidence = max(item['probability'] for item in probabilities) if probabilities else 100.0

        self.last_prediction = {
            'input_values': input_dict,
            'predicted_class_code': int(pred_idx),
            'predicted_class_label': str(pred_label_raw),
            'confidence': max_confidence,
            'probabilities': probabilities,
            'algorithm': 'Custom PatternClassifiers'
        }

        return self.last_prediction


app_instance = BayesClassifierApp()


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
        sample_datasets=sample_datasets
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if not file.filename.lower().endswith('.csv'):
            flash("Invalid file format. Please upload a valid .csv file.", "error")
            return redirect(url_for('index'))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            app_instance.load_dataset(save_path, filename)
            flash(f"Dataset '{filename}' successfully loaded ({app_instance.summary['rows']} rows, {app_instance.summary['cols']} columns).", "success")
        except Exception as e:
            flash(f"Failed to load dataset: {str(e)}", "error")
            return redirect(url_for('index'))

    elif 'sample_dataset' in request.form and request.form.get('sample_dataset'):
        sample_name = request.form.get('sample_dataset')
        sample_path = os.path.join(app.config['DATASET_FOLDER'], sample_name)
        if os.path.exists(sample_path):
            try:
                app_instance.load_dataset(sample_path, sample_name)
                flash(f"Benchmark dataset '{sample_name}' loaded successfully.", "success")
            except Exception as e:
                flash(f"Error loading sample dataset: {str(e)}", "error")
        else:
            flash(f"Sample dataset '{sample_name}' not found.", "error")
    else:
        flash("Please choose a CSV file or select a sample dataset.", "error")

    return redirect(url_for('index'))


@app.route('/prepare', methods=['POST'])
def prepare():
    target_col = request.form.get('target_col')
    selected_features = request.form.getlist('features')

    if not target_col:
        flash("Please select a target variable for prediction.", "error")
        return redirect(url_for('index'))

    try:
        app_instance.prepare_data(target_col, selected_features)
        flash(f"Target '{target_col}' configured with {len(app_instance.feature_cols)} features ({len(app_instance.numeric_features)} numeric, {len(app_instance.categorical_features)} categorical).", "success")
    except Exception as e:
        flash(f"Preparation Error: {str(e)}", "error")

    return redirect(url_for('index'))


@app.route('/train', methods=['POST'])
def train():
    test_size = float(request.form.get('test_size', 0.2))

    try:
        app_instance.train_and_evaluate(test_size=test_size)
        flash(f"Custom PatternClassifiers model successfully trained! Accuracy: {app_instance.eval_results['accuracy']}%, F1-Score: {app_instance.eval_results['f1_score']}%.", "success")
    except Exception as e:
        flash(f"Training failed: {str(e)}", "error")

    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for col in app_instance.feature_cols:
            input_data[col] = request.form.get(f'feat_{col}', '')

        app_instance.predict_new_instance(input_data)
        flash(f"Prediction complete: Class {app_instance.last_prediction['predicted_class_code']} ({app_instance.last_prediction['predicted_class_label']}) with {app_instance.last_prediction['confidence']}% posterior probability.", "success")
    except Exception as e:
        flash(f"Prediction Error: {str(e)}", "error")

    return redirect(url_for('index'))


@app.route('/reset', methods=['POST'])
def reset_app():
    app_instance.reset()
    flash("Session reset. Previous dataset, models, and parameters cleared.", "info")
    return redirect(url_for('index'))


def evaluate_dataset_benchmark(filepath, target_col, feature_cols=None):
    """Evaluates standard datasets using our custom PatternClassifiers."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[target_col])
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            X[col] = df[col].fillna(df[col].median()).astype(float)
        else:
            enc = CustomLabelEncoder()
            X[col] = enc.fit_transform(df[col].fillna(str(df[col].mode()[0])).astype(str))

    y_enc = CustomLabelEncoder()
    y = y_enc.fit_transform(df[target_col].astype(str))
    classes = y_enc.classes_

    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    clf = PatternClassifiers()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    int_labels = list(range(len(classes)))
    metrics = custom_metrics_evaluation(y_test, y_pred, int_labels)

    return {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': len(feature_cols),
        'classes': [str(c) for c in classes],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'cm': metrics['confusion_matrix']
    }


@app.route('/performance-analysis')
def performance_analysis():
    """Renders academic comparative analysis using custom classifier."""
    d1_path = os.path.join(app.config['DATASET_FOLDER'], 'Naive-Bayes-Classification-Data.csv')
    d2_path = os.path.join(app.config['DATASET_FOLDER'], 'heart.csv')

    d1_results = None
    d2_results = None

    if os.path.exists(d1_path):
        d1_results = evaluate_dataset_benchmark(d1_path, target_col='diabetes')
    if os.path.exists(d2_path):
        d2_results = evaluate_dataset_benchmark(d2_path, target_col='target')

    return render_template(
        'performance_analysis.html',
        d1=d1_results,
        d2=d2_results
    )


def kill_server():
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)


@app.route('/shutdown', methods=['POST'])
def shutdown():
    upload_path = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_path):
        for filename in os.listdir(upload_path):
            file_path = os.path.join(upload_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    plot_path = app.config['PLOT_FOLDER']
    if os.path.exists(plot_path):
        for filename in os.listdir(plot_path):
            file_path = os.path.join(plot_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    threading.Thread(target=kill_server).start()
    return "Application shut down safely and cache cleared. You can close this window."


if __name__ == '__main__':
    app.run(debug=True, port=5000)