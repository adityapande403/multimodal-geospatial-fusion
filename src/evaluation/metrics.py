"""
Evaluation Metrics Module
Implements comprehensive metrics: RMSE, SSIM, correlation, entropy, and uncertainty quantification
Target: RMSE < 0.5m, SSIM > 0.85
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

logger = logging.getLogger(__name__)


class GeospatialMetrics:
    """Compute comprehensive geospatial evaluation metrics."""
    
    @staticmethod
    def rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute Root Mean Square Error.
        Target: < 0.5m for spatial alignment accuracy
        
        Args:
            predicted: Predicted array
            ground_truth: Ground truth array
        
        Returns:
            RMSE value
        """
        mse = np.mean((predicted - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)
    
    @staticmethod
    def mae(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute Mean Absolute Error.
        
        Args:
            predicted: Predicted array
            ground_truth: Ground truth array
        
        Returns:
            MAE value
        """
        mae = np.mean(np.abs(predicted - ground_truth))
        return float(mae)
    
    @staticmethod
    def ssim_score(
        predicted: np.ndarray, 
        ground_truth: np.ndarray,
        data_range: Optional[float] = None
    ) -> float:
        """
        Compute Structural Similarity Index.
        Target: > 0.85 for alignment quality
        
        Args:
            predicted: Predicted array
            ground_truth: Ground truth array
            data_range: Data range for SSIM computation
        
        Returns:
            SSIM value
        """
        if data_range is None:
            data_range = max(predicted.max() - predicted.min(), 
                           ground_truth.max() - ground_truth.min())
        
        if predicted.ndim == 3:
            # Multi-band: compute SSIM for each band and average
            ssim_values = []
            for i in range(predicted.shape[0]):
                s = ssim(predicted[i], ground_truth[i], data_range=data_range)
                ssim_values.append(s)
            return float(np.mean(ssim_values))
        else:
            return float(ssim(predicted, ground_truth, data_range=data_range))
    
    @staticmethod
    def psnr_score(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            predicted: Predicted array
            ground_truth: Ground truth array
        
        Returns:
            PSNR value in dB
        """
        data_range = max(predicted.max() - predicted.min(), 
                        ground_truth.max() - ground_truth.min())
        return float(psnr(ground_truth, predicted, data_range=data_range))
    
    @staticmethod
    def correlation_coefficient(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            predicted: Predicted array
            ground_truth: Ground truth array
        
        Returns:
            Correlation coefficient
        """
        pred_flat = predicted.ravel()
        truth_flat = ground_truth.ravel()
        correlation = np.corrcoef(pred_flat, truth_flat)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def spectral_angle_mapper(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute Spectral Angle Mapper (SAM).
        
        Args:
            predicted: Predicted spectral vector
            ground_truth: Ground truth spectral vector
        
        Returns:
            Mean spectral angle in radians
        """
        # Normalize vectors
        pred_norm = predicted / (np.linalg.norm(predicted) + 1e-8)
        truth_norm = ground_truth / (np.linalg.norm(ground_truth) + 1e-8)
        
        # Compute angle
        dot_product = np.clip(np.dot(pred_norm, truth_norm), -1, 1)
        angle = np.arccos(dot_product)
        return float(angle)
    
    @staticmethod
    def compute_all_metrics(
        predicted: np.ndarray, 
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predicted: Predicted array
            ground_truth: Ground truth array
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'rmse': GeospatialMetrics.rmse(predicted, ground_truth),
            'mae': GeospatialMetrics.mae(predicted, ground_truth),
            'ssim': GeospatialMetrics.ssim_score(predicted, ground_truth),
            'psnr': GeospatialMetrics.psnr_score(predicted, ground_truth),
            'correlation': GeospatialMetrics.correlation_coefficient(predicted, ground_truth),
        }
        
        return metrics


class ImageQualityMetrics:
    """Compute image quality and entropy metrics."""
    
    @staticmethod
    def shannon_entropy(image: np.ndarray) -> float:
        """
        Compute Shannon entropy.
        
        Args:
            image: Input image
        
        Returns:
            Entropy value
        """
        # Normalize to 0-255
        if image.max() > 1:
            img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            img_normalized = (image * 255).astype(np.uint8)
        
        # Compute histogram
        hist, _ = np.histogram(img_normalized, bins=256)
        hist = hist / hist.sum()
        
        # Compute entropy
        ent = entropy(hist)
        return float(ent)
    
    @staticmethod
    def laplacian_variance(image: np.ndarray) -> float:
        """
        Compute Laplacian variance (sharpness metric).
        
        Args:
            image: Input image
        
        Returns:
            Laplacian variance
        """
        laplacian = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
        
        lap = np.convolve(image.ravel(), laplacian.ravel(), mode='valid')
        lap_var = np.var(lap)
        return float(lap_var)
    
    @staticmethod
    def contrast(image: np.ndarray) -> float:
        """
        Compute image contrast.
        
        Args:
            image: Input image
        
        Returns:
            Contrast value
        """
        return float(np.std(image))


class UncertaintyQuantification:
    """Quantify prediction uncertainty."""
    
    def __init__(self, num_samples: int = 100):
        """
        Initialize uncertainty estimator.
        
        Args:
            num_samples: Number of Monte Carlo samples
        """
        self.num_samples = num_samples
    
    def monte_carlo_uncertainty(
        self, 
        predictions: np.ndarray,
        dropout_enabled: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            predictions: Array of predictions from multiple forward passes
            dropout_enabled: Whether dropout was enabled during prediction
        
        Returns:
            Tuple of (mean prediction, uncertainty estimate)
        """
        if predictions.ndim < 2:
            raise ValueError("Predictions should be from multiple samples")
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        logger.info(f"MC Uncertainty | Mean std: {np.mean(std_pred):.4f}")
        return mean_pred, std_pred
    
    def bayesian_uncertainty(
        self, 
        predicted_mean: np.ndarray, 
        predicted_var: np.ndarray
    ) -> Dict:
        """
        Compute Bayesian uncertainty metrics.
        
        Args:
            predicted_mean: Predicted mean values
            predicted_var: Predicted variance values
        
        Returns:
            Dictionary of uncertainty metrics
        """
        uncertainty_metrics = {
            'aleatoric': float(np.mean(predicted_var)),  # Irreducible uncertainty
            'epistemic': float(np.var(predicted_mean)),  # Reducible uncertainty
            'total_uncertainty': float(np.mean(predicted_var) + np.var(predicted_mean))
        }
        
        logger.info(f"Bayesian uncertainty | Total: {uncertainty_metrics['total_uncertainty']:.4f}")
        return uncertainty_metrics
    
    def calibration_error(
        self, 
        predictions: np.ndarray, 
        uncertainties: np.ndarray,
        ground_truth: np.ndarray,
        num_bins: int = 10
    ) -> Dict:
        """
        Compute calibration error.
        
        Args:
            predictions: Model predictions
            uncertainties: Estimated uncertainties
            ground_truth: Ground truth values
            num_bins: Number of calibration bins
        
        Returns:
            Calibration metrics
        """
        errors = np.abs(predictions - ground_truth)
        
        # Sort by uncertainty
        sorted_idx = np.argsort(uncertainties)
        sorted_errors = errors[sorted_idx]
        
        # Compute calibration curve
        bin_size = len(errors) // num_bins
        expected = []
        actual = []
        
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(errors)
            expected.append(np.percentile(uncertainties[sorted_idx[start:end]], 50))
            actual.append(np.mean(sorted_errors[start:end]))
        
        calibration_error = np.mean(np.abs(np.array(expected) - np.array(actual)))
        
        metrics = {
            'calibration_error': float(calibration_error),
            'expected_uncertainties': expected,
            'actual_errors': actual
        }
        
        logger.info(f"Calibration error: {calibration_error:.4f}")
        return metrics
