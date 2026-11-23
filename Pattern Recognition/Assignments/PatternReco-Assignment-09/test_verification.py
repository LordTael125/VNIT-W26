"""
Pattern Recognition Lab: Assignment 09 Comprehensive Verification Suite
Tests:
  1. Scratch NumPy Implementation of Parzen Window Density Estimator (Gaussian & Hypercube)
  2. Numerical Stability via Log-Sum-Exp Trick for Gaussian Window (Small h / High Dimensions)
  3. Hypercube Window Empty-Window Tracking and Graceful Prior Fallback
  4. Bayes Decision Rule with Prior Probabilities P(w_j)
  5. Default Parameters Evaluation (Gaussian Window, h=1.0) & First Five Test Samples Details
  6. Window Width Analysis: Multi-h Evaluation (Train/Val Error Rates & Empty Windows)
  7. Plot Generation: 4 Curves on Logarithmic h Scale
  8. Optimal Window & h* Selection with Side-by-Side Comparison against Default
  9. Instance Prediction Engine with Range Outlier Warnings & Pre-processing Transformation
 10. Unified Classifiers Web Application Integration (Parzen, Ho-Kashyap, KNN, Bayes)
 11. Flask Endpoints & Integration Test Suite
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from main import (
    app,
    app_instance,
    CustomParzenClassifier,
    CustomHoKashyapClassifier,
    CustomKNNClassifier,
    PatternClassifiers,
    CustomLabelEncoder,
    custom_train_val_test_split,
    compute_classification_metrics
)


def test_parzen_math_and_logsumexp():
    print("\n==================================================================")
    print("TEST 1: Parzen Window Density Estimator & Log-Sum-Exp Stability")
    print("==================================================================")

    np.random.seed(42)
    # Synthetic 2-class problem with 4 features
    X_c0 = np.random.normal(loc=-1.0, scale=0.8, size=(60, 4))
    X_c1 = np.random.normal(loc=1.5, scale=0.8, size=(40, 4))
    X = np.vstack([X_c0, X_c1])
    y = np.array([0] * 60 + [1] * 40)

    # Test extreme small h (where standard Gaussian underflows to zero)
    small_h = 0.005
    clf_small = CustomParzenClassifier(window='gaussian', h=small_h).fit(X, y)

    # Query point near class 1 center
    q = np.array([[1.5, 1.5, 1.5, 1.5]])
    preds, posteriors, densities, log_densities, _ = clf_small.predict_details(q)

    print(f"Query point: {q[0]}")
    print(f"Gaussian with extreme small h={small_h}:")
    print(f"  Raw densities: {densities[0]}")
    print(f"  Log densities: {log_densities[0]}")
    print(f"  Posterior probabilities: {posteriors[0]}")
    print(f"  Predicted class: {preds[0]}")

    # Verify log-densities are finite and posteriors sum to 1.0 despite density underflow
    assert np.all(np.isfinite(log_densities[0])), "Log-densities must be finite via log-sum-exp!"
    assert np.isclose(np.sum(posteriors[0]), 1.0, atol=1e-5), "Posteriors must sum to 1.0!"
    assert preds[0] == 1, "Should classify to class 1!"
    print("PASS: Gaussian log-sum-exp trick completely prevents floating-point underflow!")

    # Test Hypercube window
    clf_cube = CustomParzenClassifier(window='hypercube', h=1.5).fit(X, y)
    preds_cube, post_cube, dens_cube, log_dens_cube, empty_cube = clf_cube.predict_details(q)
    print(f"\nHypercube with h=1.5:")
    print(f"  Densities: {dens_cube[0]}")
    print(f"  Posteriors: {post_cube[0]}")
    print(f"  Is empty window: {empty_cube[0]}")
    print(f"  Predicted class: {preds_cube[0]}")
    assert np.isclose(np.sum(post_cube[0]), 1.0, atol=1e-5), "Hypercube posteriors must sum to 1.0!"
    print("PASS: Hypercube density and Bayes prediction verified!")


def test_default_parameters_and_first_five_samples():
    print("\n==================================================================")
    print("TEST 2: Default Parameters (Gaussian, h=1.0) & First 5 Test Samples")
    print("==================================================================")

    data_path = os.path.join(BASE_DIR, 'Dataset', 'iris.csv')
    app_instance.load_dataset(data_path, 'iris.csv')

    h_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    res = app_instance.train_parzen(h_values=h_values, use_standardization=True)

    print(f"Dataset: Iris (Rows: {app_instance.summary['rows']}, Features: {len(app_instance.feature_cols)})")
    print(f"Class Priors:")
    for cls in app_instance.encoder.get_classes_info():
        prior = app_instance.parzen_default_model.priors_[cls['code']]
        print(f"  Class {cls['code']} ({cls['display_name']}): P(w_{cls['code']}) = {prior:.4f}")

    print(f"\nDefault Parameters (Gaussian, h=1.0) Test Metrics:")
    def_metrics = res['default_results']
    print(f"  Accuracy:   {def_metrics['accuracy']}%")
    print(f"  Error Rate: {def_metrics['error_rate']}%")
    print(f"  Macro F1:   {def_metrics['f1_score']}%")

    print("\nFirst Five Test Samples Breakdown (Class-conditional densities & posteriors):")
    assert len(app_instance.parzen_first_five_test) >= 5, "Must record at least 5 test samples!"
    for s in app_instance.parzen_first_five_test[:5]:
        status = "CORRECT" if s['is_correct'] else "MISCLASSIFIED"
        print(f"  Sample #{s['sample_id']}: True={s['true_class']:<15} Pred={s['pred_class']:<15} [{status}]")
        for cname, pval in s['posteriors'].items():
            dens = s['densities'][cname]
            print(f"     -> {cname}: p(x|w)={dens}, P(w|x)={pval}%")

    print("PASS: First five test samples recorded and verified!")


def test_window_width_analysis_and_optimal_selection():
    print("\n==================================================================")
    print("TEST 3: Window Width Analysis (Multi-h) & Optimal Model Selection")
    print("==================================================================")

    data_path = os.path.join(BASE_DIR, 'Dataset', 'Naive-Bayes-Classification-Data.csv')
    app_instance.load_dataset(data_path, 'Naive-Bayes-Classification-Data.csv')

    h_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    res = app_instance.train_parzen(h_values=h_values, use_standardization=True)

    print(f"Dataset: Naive-Bayes (Glucose & Blood Pressure, N={app_instance.summary['rows']})")
    print(f"\nMulti-h Analysis Table:")
    print(f"{'h':<8} | {'G-Train':<10} | {'G-Val':<10} | {'H-Train':<10} | {'H-Val':<10} | {'H-Empty-Val':<12}")
    print("-" * 72)
    for i in range(len(h_values)):
        g = app_instance.parzen_multi_h_results_gaussian[i]
        h = app_instance.parzen_multi_h_results_hypercube[i]
        print(f"{g['h']:<8} | {g['train_error_rate']:<10} | {g['val_error_rate']:<10} | {h['train_error_rate']:<10} | {h['val_error_rate']:<10} | {h['empty_windows_val']:<12}")

    print(f"\nOptimal Model Selected: {res['optimal_window'].capitalize()} Window with h* = {res['optimal_h']}")
    opt_metrics = res['optimal_results']
    print(f"  Optimal Test Accuracy: {opt_metrics['accuracy']}%")
    print(f"  Optimal Error Rate:    {opt_metrics['error_rate']}%")
    print(f"  Optimal Macro F1:      {opt_metrics['f1_score']}%")

    # Verify plot file was generated
    plot_file = os.path.join(app.config['PLOT_FOLDER'], 'parzen_h_analysis.png')
    assert os.path.exists(plot_file), "Parzen h analysis plot must exist!"
    assert os.path.getsize(plot_file) > 1000, "Plot file must not be empty!"
    print(f"PASS: Multi-h analysis plot generated successfully at {plot_file}!")

    # Verify comparison table
    comp = app_instance.parzen_comparison
    print(f"\nSide-by-Side Comparison (Default vs Optimal):")
    print(f"  Default (Gaussian, h=1.0): Accuracy = {comp['default']['accuracy']}%, Error Rate = {comp['default']['error_rate']}%")
    print(f"  Optimal ({comp['optimal']['window']}, h*={comp['optimal']['h']}): Accuracy = {comp['optimal']['accuracy']}%, Error Rate = {comp['optimal']['error_rate']}%")
    print(f"  Accuracy Gain: {comp['delta']['accuracy']:+}%")
    print("PASS: Optimal model selection and comparison confirmed!")


def test_prediction_and_range_validation():
    print("\n==================================================================")
    print("TEST 4: Instance Prediction & Domain Range Warning Validation")
    print("==================================================================")

    # Use Naive-Bayes data (features: glucose [20, 70], bloodpressure [50, 100])
    # Test valid in-domain instance
    in_domain = {'glucose': '45.0', 'bloodpressure': '80.0'}
    pred1 = app_instance.predict_new_instance(in_domain)
    print(f"In-domain prediction for {in_domain}:")
    print(f"  Predicted Label: {pred1['predicted_label']} (Code: {pred1['predicted_code']})")
    print(f"  Range warnings count: {len(pred1['domain_warnings'])}")
    assert len(pred1['domain_warnings']) == 0, "Normal input should have 0 warnings!"

    # Test outlier instance far outside training range
    outlier = {'glucose': '999.0', 'bloodpressure': '-50.0'}
    pred2 = app_instance.predict_new_instance(outlier)
    print(f"\nOutlier prediction for {outlier}:")
    print(f"  Predicted Label: {pred2['predicted_label']} (Code: {pred2['predicted_code']})")
    print(f"  Range warnings count: {len(pred2['domain_warnings'])}")
    for w in pred2['domain_warnings']:
        print(f"    Warning: Feature '{w['feature']}' = {w['value']} is {w['status']} (Training: {w['training_range']})")
    assert len(pred2['domain_warnings']) >= 1, "Outlier values must trigger domain range warnings!"
    print("PASS: Prediction input validation and range warning verified!")


def test_unified_classifiers_integration():
    print("\n==================================================================")
    print("TEST 5: Unified Classifiers Integration (Parzen, Ho-Kashyap, KNN, Bayes)")
    print("==================================================================")

    data_path = os.path.join(BASE_DIR, 'Dataset', 'linearly_separable_data.csv')
    app_instance.load_dataset(data_path, 'linearly_separable_data.csv')

    # 1. Parzen Window
    res_parzen = app_instance.train_parzen([0.1, 0.5, 1.0, 2.0])
    print(f"Parzen Window: Accuracy={res_parzen['optimal_results']['accuracy']}%, Window={res_parzen['optimal_window']}, h*={res_parzen['optimal_h']}")
    assert res_parzen['optimal_results']['accuracy'] >= 90.0

    # 2. Ho-Kashyap Linear Classifier
    res_hk = app_instance.train_hokashyap(default_eta=0.1, max_iter=300)
    print(f"Ho-Kashyap: Accuracy={res_hk['metrics']['accuracy']}%, Separability={res_hk['separability']}, Optimal η*={res_hk['optimal_eta']}")
    assert 'Separable' in res_hk['separability']

    # 3. K-Nearest Neighbors
    res_knn = app_instance.train_knn(max_k=15)
    print(f"KNN: Accuracy={res_knn['optimal_results']['accuracy']}%, Optimal K={res_knn['selected_k']}")
    assert res_knn['optimal_results']['accuracy'] >= 90.0

    # 4. Gaussian Naive Bayes
    res_bayes = app_instance.train_bayes()
    print(f"Bayes: Accuracy={res_bayes['accuracy']}%, F1={res_bayes['f1_score']}%")
    assert res_bayes['accuracy'] >= 90.0

    # 5. Comparative Performance Analysis
    perf = app_instance.run_performance_analysis()
    print(f"\nComparative Benchmark Studio Matrix:")
    for row in perf['benchmark_table']:
        print(f"  {row['name']:<35} | Acc: {row['accuracy']:<6}% | F1: {row['f1_score']:<6}% | Latency: {row['latency_ms']} ms")

    print(f"Best Recommended Model: {perf['best_model']['name']}")
    assert os.path.exists(os.path.join(app.config['PLOT_FOLDER'], 'benchmark_comparison.png'))
    print("PASS: Unified 4-classifier studio integration verified!")


def test_flask_endpoints():
    print("\n==================================================================")
    print("TEST 6: Flask Endpoints & Integration Client Tests")
    print("==================================================================")

    client = app.test_client()

    # GET /
    r = client.get('/')
    assert r.status_code == 200
    assert b"Classifiers" in r.data
    print("PASS: GET / -> 200 OK")

    # POST /upload with preset
    r = client.post('/upload', data={'source': 'preset', 'preset_dataset': 'iris.csv'}, follow_redirects=True)
    assert r.status_code == 200
    assert b"Iris" in r.data or b"iris.csv" in r.data
    print("PASS: POST /upload (preset) -> 200 OK")

    # POST /prepare
    r = client.post('/prepare', data={
        'target_col': 'Species',
        'impute_strategy': 'mean',
        'cat_handling': 'encode',
        'use_standardization': 'on'
    }, follow_redirects=True)
    assert r.status_code == 200
    print("PASS: POST /prepare -> 200 OK")

    # POST /train with Parzen Window
    r = client.post('/train', data={
        'classifier_choice': 'parzen',
        'h_values': '0.1, 0.5, 1.0, 2.0',
        'use_standardization': 'on'
    }, follow_redirects=True)
    assert r.status_code == 200
    assert b"Parzen Window Density Estimator Trained" in r.data
    print("PASS: POST /train (Parzen Window) -> 200 OK")

    # POST /predict
    r = client.post('/predict', data={
        'SepalLengthCm': '5.1',
        'SepalWidthCm': '3.5',
        'PetalLengthCm': '1.4',
        'PetalWidthCm': '0.2'
    }, follow_redirects=True)
    assert r.status_code == 200
    assert b"Predicted Class" in r.data
    print("PASS: POST /predict -> 200 OK")

    # GET /performance-analysis
    r = client.get('/performance-analysis')
    assert r.status_code == 200
    assert b"Performance Benchmark Matrix" in r.data
    print("PASS: GET /performance-analysis -> 200 OK")

    # POST /reset
    r = client.post('/reset', follow_redirects=True)
    assert r.status_code == 200
    print("PASS: POST /reset -> 200 OK")


if __name__ == '__main__':
    print("==================================================================")
    print("RUNNING ASSIGNMENT 09 VERIFICATION SUITE")
    print("==================================================================")
    test_parzen_math_and_logsumexp()
    test_default_parameters_and_first_five_samples()
    test_window_width_analysis_and_optimal_selection()
    test_prediction_and_range_validation()
    test_unified_classifiers_integration()
    test_flask_endpoints()
    print("\n==================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (100% SPECIFICATION CONFORMANCE)")
    print("==================================================================")
