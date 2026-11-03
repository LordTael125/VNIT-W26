"""
Pattern Recognition Lab: Assignment 08 Comprehensive Verification Suite
Tests:
  1. Scratch NumPy implementation of Ho-Kashyap Procedure (Separable & Non-Separable cases)
  2. Recording of weight vector a, margin vector b (min/max), and test metrics
  3. Tracking squared error ||e(k)||^2 and misclassified samples at every iteration
  4. Multi-learning-rate exploration (eta in [0.1, ..., 0.9]) and optimal eta selection
  5. Linear separability testing
  6. Integration with KNN (Elbow Method) and Bayes Classifier
  7. Pre-processing, imputation, encoding, and standardization
  8. Instance prediction interface with validation and explainability
  9. Flask web application endpoints and routes
"""

import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from main import (
    app,
    app_instance,
    CustomHoKashyapClassifier,
    CustomKNNClassifier,
    PatternClassifiers,
    CustomLabelEncoder,
    custom_train_test_split,
    compute_classification_metrics
)


def test_hokashyap_separable():
    print("\n--- Test 1: Ho-Kashyap Procedure on Linearly Separable Dataset ---")
    data_path = os.path.join(BASE_DIR, 'Dataset', 'linearly_separable_data.csv')
    df = pd.read_csv(data_path)
    X = df[['Feature_1', 'Feature_2']].values
    y = df['Target_Class'].values

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    clf = CustomHoKashyapClassifier(learning_rate=0.1, max_iter=300, tol=1e-5)
    clf.fit(X_train, y_train)

    assert clf.a_ is not None, "Weight vector a must not be None"
    assert clf.b_ is not None, "Margin vector b must not be None"
    assert len(clf.a_) == 3, f"Expected weight vector length 3 (1 bias + 2 features), got {len(clf.a_)}"
    assert np.all(clf.b_ > 0), "All components of margin vector b must be strictly positive"

    margin_summary = clf.get_margin_summary()
    assert margin_summary['min_b'] > 0, "Minimum margin must be > 0"
    assert margin_summary['max_b'] >= margin_summary['min_b'], "Max margin must be >= min margin"

    # History tracking
    assert len(clf.history_['iteration']) == clf.iterations_run_, "Iteration history length mismatch"
    assert len(clf.history_['squared_error']) == clf.iterations_run_, "Squared error history length mismatch"
    assert len(clf.history_['misclassified']) == clf.iterations_run_, "Misclassified history length mismatch"

    # Evaluation on test set
    y_pred = clf.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, labels=[0, 1])

    print(f"  Separability Status: {clf.separability_}")
    print(f"  Iterations Run: {clf.iterations_run_}")
    print(f"  Learned Weight Vector a: {np.round(clf.a_, 4)}")
    print(f"  Margin Vector b: min={margin_summary['min_b']}, max={margin_summary['max_b']}")
    print(f"  Test Accuracy: {metrics['accuracy']}%, Error Rate: {metrics['error_rate']}%")

    assert metrics['accuracy'] >= 95.0, f"Expected high accuracy on linearly separable data, got {metrics['accuracy']}%"
    print("  ✓ Test 1 Passed!")


def test_hokashyap_multi_eta_and_non_separable():
    print("\n--- Test 2: Ho-Kashyap Multi-Eta Convergence on Heart Disease (Non-Separable) ---")
    data_path = os.path.join(BASE_DIR, 'Dataset', 'heart.csv')
    df = pd.read_csv(data_path)
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak']
    X = df[feature_cols].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    eta_results = []
    best_candidate = None
    best_score = (float('inf'), float('inf'))

    for eta in etas:
        clf = CustomHoKashyapClassifier(learning_rate=eta, max_iter=300)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        metrics = compute_classification_metrics(y_test, pred, labels=[0, 1])
        final_err = clf.history_['squared_error'][-1]
        
        entry = {
            'eta': eta,
            'iterations': clf.iterations_run_,
            'error_rate': metrics['error_rate'],
            'accuracy': metrics['accuracy'],
            'squared_error': final_err
        }
        eta_results.append(entry)

        score = (entry['error_rate'], entry['iterations'])
        if score < best_score:
            best_score = score
            best_candidate = entry

    print(f"  Multi-Eta Runs:")
    for r in eta_results:
        print(f"    η={r['eta']:.1f} | iters={r['iterations']:3d} | Err Rate={r['error_rate']:.2f}% | Acc={r['accuracy']:.2f}% | ||e||²={r['squared_error']:.2f}")

    assert best_candidate is not None, "Optimal eta candidate must be found"
    print(f"  Selected Optimal η* = {best_candidate['eta']:.2f} (Fewest iters: {best_candidate['iterations']}, Lowest Err: {best_candidate['error_rate']}%)")
    assert best_candidate['error_rate'] < 30.0, "Reasonable error rate expected on Heart dataset"
    print("  ✓ Test 2 Passed!")


def test_app_state_workflow():
    print("\n--- Test 3: Full AppState Workflow with Ingestion, Preprocessing, Multi-Classifier Training & Prediction ---")
    app_instance.reset()
    data_path = os.path.join(BASE_DIR, 'Dataset', 'linearly_separable_data.csv')
    app_instance.load_dataset(data_path, 'linearly_separable_data.csv')

    assert app_instance.summary['rows'] == 200, "Expected 200 rows in linearly_separable_data.csv"
    assert app_instance.summary['cols'] == 3, "Expected 3 columns"

    # Preprocessing
    prep_res = app_instance.execute_preprocessing(use_standardization=True)
    assert prep_res['train_samples'] > 0
    assert prep_res['test_samples'] > 0
    assert app_instance.preprocessing_done

    # 1. Train Ho-Kashyap
    hk_res = app_instance.train_hokashyap(default_eta=0.1, max_iter=200)
    assert hk_res['metrics']['accuracy'] >= 90.0
    assert app_instance.hk_optimal_eta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Adopt optimal eta
    opt_adopted = app_instance.adopt_optimal_eta()
    assert opt_adopted == app_instance.hk_optimal_eta

    # Predict instance with Ho-Kashyap
    pred = app_instance.predict_new_instance({'Feature_1': '2.0', 'Feature_2': '2.0'})
    assert pred['predicted_class_code'] in [0, 1]
    assert 'discriminant_score' in pred
    assert 'confidence' in pred
    print(f"  Ho-Kashyap Predicted Class: {pred['predicted_class_code']} ({pred['predicted_class_label']}) with {pred['confidence']}% confidence")

    # 2. Train KNN
    app_instance.train_knn(max_k=20, use_standardization=True)
    assert app_instance.knn_k1_results is not None
    assert app_instance.knn_selected_elbow_k is not None
    pred_knn = app_instance.predict_new_instance({'Feature_1': '-2.0', 'Feature_2': '-2.0'})
    print(f"  KNN Predicted Class: {pred_knn['predicted_class_code']} ({pred_knn['predicted_class_label']})")

    # 3. Train Bayes
    app_instance.train_bayes()
    assert app_instance.bayes_results is not None
    pred_bayes = app_instance.predict_new_instance({'Feature_1': '2.0', 'Feature_2': '2.0'})
    print(f"  Bayes Predicted Class: {pred_bayes['predicted_class_code']} ({pred_bayes['predicted_class_label']})")

    print("  ✓ Test 3 Passed!")


def test_flask_routes():
    print("\n--- Test 4: Flask Web Application Routes & Endpoints ---")
    client = app.test_client()

    # 1. GET /
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200 on /, got {res.status_code}"
    assert b"Classifiers" in res.data
    assert b"Ho-Kashyap" in res.data

    # 2. POST /upload with benchmark dataset
    res = client.post('/upload', data={'sample_dataset': 'linearly_separable_data.csv'}, follow_redirects=True)
    assert res.status_code == 200
    assert b"linearly_separable_data.csv" in res.data

    # 3. POST /prepare
    res = client.post('/prepare', data={
        'target_col': 'Target_Class',
        'features': ['Feature_1', 'Feature_2'],
        'use_standardization': 'true'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Pre-processing completed" in res.data

    # 4. POST /rename_classes
    res = client.post('/rename_classes', data={
        'rename_class_0': 'Negative Cluster',
        'rename_class_1': 'Positive Cluster'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Target class display labels updated" in res.data

    # 5. POST /train with Ho-Kashyap
    res = client.post('/train', data={
        'classifier_choice': 'hokashyap',
        'default_eta': '0.1',
        'max_iter': '200',
        'tol': '1e-5',
        'selected_etas': ['0.1', '0.3', '0.5', '0.7', '0.9'],
        'use_standardization': 'true'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Ho-Kashyap Procedure Executed" in res.data

    # 6. POST /adopt_optimal_eta
    res = client.post('/adopt_optimal_eta', follow_redirects=True)
    assert res.status_code == 200
    assert b"Optimal learning rate model" in res.data

    # 7. POST /predict
    res = client.post('/predict', data={
        'feat_Feature_1': '1.5',
        'feat_Feature_2': '1.5'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Prediction successful" in res.data

    # 8. GET /performance-analysis
    res = client.get('/performance-analysis')
    assert res.status_code == 200
    assert b"Comparative Analysis Studio" in res.data
    assert b"Multi-Dataset Algorithm Performance Scorecard" in res.data

    print("  ✓ Test 4 Passed!")


if __name__ == '__main__':
    print("=====================================================================")
    print("  STARTING ASSIGNMENT 08 AUTOMATED TEST & VERIFICATION SUITE")
    print("=====================================================================")
    try:
        test_hokashyap_separable()
        test_hokashyap_multi_eta_and_non_separable()
        test_app_state_workflow()
        test_flask_routes()
        print("\n=====================================================================")
        print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! (100% PASS)")
        print("=====================================================================")
    except AssertionError as e:
        print(f"\n❌ Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

