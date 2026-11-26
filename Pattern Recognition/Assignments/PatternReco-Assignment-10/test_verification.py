#!/usr/bin/env python3
"""
Automated Comprehensive Test Verification Suite for Assignment 10
Tests from-scratch 2D DFT matrix operations, frequency centering,
Parseval's theorem, filtering, invariant features, classification,
and Flask application routes.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

# Add current folder to path
sys.path.insert(0, os.path.dirname(__file__))
from main import app, state, Custom2DDFT, FrequencyFilters, InvariantSpectralFeatures, PatternLibrary, compute_spectral_metrics


class Test2DDFTEngine(unittest.TestCase):
    """Verifies pure from-scratch NumPy 2D DFT and 2D IDFT matrix algebra."""

    def setUp(self):
        np.random.seed(42)
        self.M, self.N = 64, 64
        self.img = np.random.rand(self.M, self.N)

    def test_uncentered_dft_accuracy(self):
        """Verify from-scratch uncentered DFT matches np.fft.fft2 down to floating-point error."""
        F_scratch = Custom2DDFT.fft2(self.img, center=False)
        F_numpy = np.fft.fft2(self.img)
        diff = np.max(np.abs(F_scratch - F_numpy))
        self.assertLess(diff, 1e-11, f"Uncentered 2D DFT numerical error {diff} exceeded threshold.")

    def test_centered_dft_accuracy(self):
        """Verify (-1)^(x+y) spatial multiplication matches np.fft.fftshift(np.fft.fft2(img))."""
        F_scratch = Custom2DDFT.fft2(self.img, center=True)
        F_numpy_shift = np.fft.fftshift(np.fft.fft2(self.img))
        diff = np.max(np.abs(F_scratch - F_numpy_shift))
        self.assertLess(diff, 1e-10, f"Centered 2D DFT numerical error {diff} exceeded threshold.")

    def test_centering_dc_component_location(self):
        """Verify zero-frequency DC component shifts from (0, 0) to (M/2, N/2)."""
        const_img = np.ones((self.M, self.N)) * 3.5
        F_uncentered = Custom2DDFT.fft2(const_img, center=False)
        F_centered = Custom2DDFT.fft2(const_img, center=True)

        # Uncentered DC must be at (0, 0)
        self.assertAlmostEqual(F_uncentered[0, 0].real, 3.5 * self.M * self.N, places=5)
        self.assertAlmostEqual(np.abs(F_uncentered[self.M // 2, self.N // 2]), 0.0, places=5)

        # Centered DC must be at (M/2, N/2)
        self.assertAlmostEqual(np.abs(F_centered[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(F_centered[self.M // 2, self.N // 2].real, 3.5 * self.M * self.N, places=5)

    def test_idft_exact_reconstruction(self):
        """Verify 2D IDFT accurately inverts 2D DFT (both centered and uncentered)."""
        # Uncentered
        F_uncentered = Custom2DDFT.fft2(self.img, center=False)
        rec_uncentered = Custom2DDFT.ifft2(F_uncentered, center=False)
        err_uncentered = np.max(np.abs(self.img - rec_uncentered))
        self.assertLess(err_uncentered, 1e-12, f"Uncentered IDFT reconstruction error: {err_uncentered}")

        # Centered
        F_centered = Custom2DDFT.fft2(self.img, center=True)
        rec_centered = Custom2DDFT.ifft2(F_centered, center=True)
        err_centered = np.max(np.abs(self.img - rec_centered))
        self.assertLess(err_centered, 1e-12, f"Centered IDFT reconstruction error: {err_centered}")

    def test_parseval_theorem(self):
        """Verify Parseval's theorem: sum |f(x, y)|^2 == (1/MN) * sum |F(u, v)|^2."""
        F = Custom2DDFT.fft2(self.img, center=True)
        spatial_energy = np.sum(np.abs(self.img) ** 2)
        freq_energy = (1.0 / (self.M * self.N)) * np.sum(np.abs(F) ** 2)
        diff = abs(spatial_energy - freq_energy)
        self.assertLess(diff, 1e-11, f"Parseval theorem deviation {diff} exceeded threshold.")


class TestFrequencyFilters(unittest.TestCase):
    """Verifies Ideal, Gaussian, and Butterworth filter transfer functions and metrics."""

    def setUp(self):
        self.M, self.N = 64, 64
        self.D0 = 15.0

    def test_filter_shapes_and_extrema(self):
        """Verify transfer function boundary conditions at DC center (M/2, N/2) and far frequencies."""
        u0, v0 = self.M // 2, self.N // 2

        # Ideal Low-Pass
        H_id = FrequencyFilters.ideal_lowpass(self.M, self.N, self.D0)
        self.assertEqual(H_id[u0, v0], 1.0)
        self.assertEqual(H_id[0, 0], 0.0)  # Corner is distance sqrt(32^2 + 32^2) = 45.25 > 15

        # Gaussian Low-Pass
        H_ga = FrequencyFilters.gaussian_lowpass(self.M, self.N, self.D0)
        self.assertAlmostEqual(H_ga[u0, v0], 1.0, places=6)
        self.assertLess(H_ga[0, 0], 0.02)

        # Butterworth Low-Pass
        H_bu = FrequencyFilters.butterworth_lowpass(self.M, self.N, self.D0, order=2)
        self.assertAlmostEqual(H_bu[u0, v0], 1.0, places=6)
        self.assertLess(H_bu[0, 0], 0.05)

        # High-Pass complements
        H_ihp = FrequencyFilters.ideal_highpass(self.M, self.N, self.D0)
        np.testing.assert_allclose(H_id + H_ihp, 1.0)

    def test_multi_d0_monotonicity(self):
        """Verify that spectral energy retained increases monotonically with cutoff radius D0."""
        img = np.ones((self.M, self.N))
        img[::2, :] = 0.0  # Periodic horizontal stripes
        F = Custom2DDFT.fft2(img, center=True)

        cutoffs = [5, 10, 15, 20, 30, 50]
        prev_energy = 0.0
        for d in cutoffs:
            H = FrequencyFilters.ideal_lowpass(self.M, self.N, d)
            G = H * F
            rec = Custom2DDFT.ifft2(G, center=True)
            m = compute_spectral_metrics(img, F, G, rec, d)
            self.assertGreaterEqual(m['energy_retained_pct'], prev_energy - 1e-5)
            prev_energy = m['energy_retained_pct']


class TestInvariantSpectralFeatures(unittest.TestCase):
    """Verifies translation, scale, illumination, and rotation invariance properties."""

    def setUp(self):
        # Create base test texture
        x = np.arange(64)
        X, Y = np.meshgrid(x, x)
        self.base_img = 0.5 + 0.5 * np.cos(2 * np.pi * X / 16.0)

    def test_translation_invariance(self):
        """Verify spatial circular translation leaves normalized (R + A)-D feature vector identical."""
        shifted_img = np.roll(self.base_img, shift=13, axis=1)

        f_orig = InvariantSpectralFeatures.extract(self.base_img, num_rings=16, num_wedges=16)
        f_shift = InvariantSpectralFeatures.extract(shifted_img, num_rings=16, num_wedges=16)

        dist = np.linalg.norm(f_orig - f_shift)
        self.assertLess(dist, 1e-12, f"Feature vector was not invariant to spatial translation (dist={dist}).")

    def test_illumination_scaling_invariance(self):
        """Verify uniform illumination scaling leaves normalized feature vector identical."""
        scaled_img = self.base_img * 4.25

        f_orig = InvariantSpectralFeatures.extract(self.base_img, num_rings=16, num_wedges=16)
        f_scaled = InvariantSpectralFeatures.extract(scaled_img, num_rings=16, num_wedges=16)

        dist = np.linalg.norm(f_orig - f_scaled)
        self.assertLess(dist, 1e-12, f"Feature vector was not invariant to illumination scaling (dist={dist}).")

    def test_radial_ring_rotation_invariance(self):
        """Verify concentric circular patterns retain radial ring energy under rotation."""
        from scipy.ndimage import rotate
        x = np.arange(64) - 31.5
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        concentric = 0.5 + 0.5 * np.cos(2 * np.pi * R / 10.0)

        f_orig = InvariantSpectralFeatures.extract(concentric, num_rings=16, num_wedges=16)

        # Exact 90-degree rotation (discrete grid exactness)
        rot90 = np.rot90(concentric)
        f_rot90 = InvariantSpectralFeatures.extract(rot90, num_rings=16, num_wedges=16)
        self.assertAlmostEqual(float(np.linalg.norm(f_orig[:16] - f_rot90[:16])), 0.0, delta=1e-12)

        # Arbitrary rotation (accounting for discrete pixel interpolation blur)
        concentric_rot = rotate(concentric, 40, reshape=False, mode='wrap')
        f_rot = InvariantSpectralFeatures.extract(concentric_rot, num_rings=16, num_wedges=16)

        # Radial rings are the first 16 dimensions
        rings_orig = f_orig[:16]
        rings_rot = f_rot[:16]
        diff = np.linalg.norm(rings_orig - rings_rot)
        self.assertLess(diff, 0.10, f"Radial rings deviated under rotation (diff={diff}).")


class TestPatternClassification(unittest.TestCase):
    """Verifies pattern recognition classification accuracy across reference classes."""

    @classmethod
    def setUpClass(cls):
        dataset_dir = os.path.join(os.path.dirname(__file__), 'Dataset')
        PatternLibrary.load_library(dataset_dir)

    def test_query_instances_classification(self):
        """Verify that unseen query instances correctly classify to their intended classes."""
        dataset_dir = os.path.join(os.path.dirname(__file__), 'Dataset')
        queries = [
            ('test_vertical_shifted.png', 'Vertical Stripes'),
            ('test_horizontal_scaled.png', 'Horizontal Stripes'),
            ('test_diagonal_noisy.png', 'Diagonal Stripes'),
            ('test_concentric_rotated.png', 'Concentric Rings'),
            ('test_checkerboard_shifted.png', 'Checkerboard'),
            ('test_radial_star_rotated.png', 'Radial Star'),
            ('test_dot_lattice_shifted.png', 'Dot Lattice'),
            ('test_natural_texture_scaled.png', 'Natural Texture')
        ]

        for fname, expected_label in queries:
            path = os.path.join(dataset_dir, fname)
            if not os.path.exists(path):
                continue
            img = np.array(Image.open(path).convert('L').resize((64, 64))) / 255.0
            res = PatternLibrary.classify_query(img)
            self.assertEqual(res['predicted_class'], expected_label,
                             f"Query {fname} predicted as {res['predicted_class']}, expected {expected_label}")
            self.assertGreaterEqual(res['top_confidence'], 90.0)


class TestFlaskEndpoints(unittest.TestCase):
    """Tests all Flask routes and workflows end-to-end."""

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        """Verify GET / renders homepage successfully."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'2D DFT Spectral Analysis', resp.data)

    def test_upload_sample_and_preprocess(self):
        """Verify POST /upload and POST /preprocess execute without error."""
        # Upload
        resp_up = self.client.post('/upload', data={'selected_sample': 'checkerboard_1.png'}, follow_redirects=True)
        self.assertEqual(resp_up.status_code, 200)
        self.assertIsNotNone(state.raw_image_path)

        # Preprocess
        resp_prep = self.client.post('/preprocess', data={
            'target_size': '64',
            'convert_gray': 'true',
            'normalize': 'true'
        }, follow_redirects=True)
        self.assertEqual(resp_prep.status_code, 200)
        self.assertIsNotNone(state.preprocessed_array)

    def test_analyze_dft_and_filters(self):
        """Verify POST /analyze_dft generates metrics and 4-panel visualizer."""
        # Ensure state has preprocessed image
        img = np.array(Image.open(os.path.join(app.config['DATASET_FOLDER'], 'checkerboard_1.png')).convert('L').resize((64, 64))) / 255.0
        state.preprocessed_array = img

        resp_dft = self.client.post('/analyze_dft', data={
            'centering_toggle': 'on',
            'filter_type': 'ideal',
            'filter_mode': 'lowpass',
            'cutoff_d0': '15'
        }, follow_redirects=True)
        self.assertEqual(resp_dft.status_code, 200)
        self.assertIn('dimensions', state.dft_results)
        self.assertTrue(os.path.exists(os.path.join(app.config['PLOTS_FOLDER'], state.dft_results['plot_filename'])))

    def test_filter_sweep_and_optimization(self):
        """Verify POST /filter_sweep executes multi-cutoff analysis."""
        img = np.array(Image.open(os.path.join(app.config['DATASET_FOLDER'], 'checkerboard_1.png')).convert('L').resize((64, 64))) / 255.0
        state.preprocessed_array = img

        resp_sweep = self.client.post('/filter_sweep', data={
            'cutoffs': '5, 10, 15, 20, 25, 30, 40, 50',
            'sweep_mode': 'lowpass'
        }, follow_redirects=True)
        self.assertEqual(resp_sweep.status_code, 200)
        self.assertIn('optimal', state.filter_analysis)
        self.assertGreaterEqual(state.filter_analysis['optimal']['energy_retained_pct'], 95.0)

    def test_predict_query_endpoint(self):
        """Verify POST /predict_query runs classification and feature plotting."""
        resp_pred = self.client.post('/predict_query', data={
            'selected_query': 'test_vertical_shifted.png'
        }, follow_redirects=True)
        self.assertEqual(resp_pred.status_code, 200)
        self.assertEqual(state.query_results['predicted_class'], 'Vertical Stripes')

    def test_cleanup_garbage_collector(self):
        """Verify POST /cleanup executes garbage collection and returns json."""
        resp_clean = self.client.post('/cleanup')
        self.assertEqual(resp_clean.status_code, 200)
        data = resp_clean.get_json()
        self.assertEqual(data['status'], 'success')


if __name__ == '__main__':
    unittest.main()
