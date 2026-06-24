"""
Change Detection Module
Implements NDVI-based, pixel-by-pixel, and land cover transition analysis
for temporal change detection in geospatial data
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import ndimage
from skimage import filters

logger = logging.getLogger(__name__)


class NDVIChangeDetector:
    """Detect changes using NDVI (Normalized Difference Vegetation Index)."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize NDVI change detector.
        
        Args:
            threshold: Change threshold for NDVI
        """
        self.threshold = threshold
    
    def compute_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Compute NDVI from red and NIR bands.
        
        Args:
            red: Red band data
            nir: Near-infrared band data
        
        Returns:
            NDVI array
        """
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)
        logger.info(f"Computed NDVI: range [{ndvi.min():.3f}, {ndvi.max():.3f}]")
        return ndvi
    
    def detect_change(self, ndvi_t1: np.ndarray, ndvi_t2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect vegetation change between two time points.
        
        Args:
            ndvi_t1: NDVI at time 1
            ndvi_t2: NDVI at time 2
        
        Returns:
            Tuple of (change map, metrics)
        """
        ndvi_diff = ndvi_t2 - ndvi_t1
        
        # Binary change map
        change_map = np.abs(ndvi_diff) > self.threshold
        
        # Compute metrics
        metrics = {
            'ndvi_diff_mean': float(np.mean(ndvi_diff)),
            'ndvi_diff_std': float(np.std(ndvi_diff)),
            'change_pixels': int(np.sum(change_map)),
            'change_percentage': float(100 * np.sum(change_map) / change_map.size),
            'vegetation_loss': int(np.sum(ndvi_diff < -self.threshold)),
            'vegetation_gain': int(np.sum(ndvi_diff > self.threshold))
        }
        
        logger.info(f"NDVI Change Detection | Loss: {metrics['vegetation_loss']} | Gain: {metrics['vegetation_gain']}")
        
        return change_map.astype(np.uint8), metrics


class PixelChangeDetector:
    """Detect changes using pixel-by-pixel comparison."""
    
    def __init__(self, method: str = 'difference', threshold: float = 0.05):
        """
        Initialize pixel change detector.
        
        Args:
            method: Comparison method ('difference', 'ratioed', 'chi_square')
            threshold: Change threshold
        """
        self.method = method
        self.threshold = threshold
    
    def difference_method(self, img_t1: np.ndarray, img_t2: np.ndarray) -> np.ndarray:
        """
        Simple difference method.
        
        Args:
            img_t1: Image at time 1
            img_t2: Image at time 2
        
        Returns:
            Change map
        """
        if img_t1.ndim == 3:
            # Multi-band: compute difference for each band and combine
            diff = np.abs(img_t2 - img_t1)
            diff_combined = np.mean(diff, axis=0)
        else:
            diff_combined = np.abs(img_t2 - img_t1)
        
        change_map = diff_combined > self.threshold
        return change_map.astype(np.uint8)
    
    def ratioed_method(self, img_t1: np.ndarray, img_t2: np.ndarray) -> np.ndarray:
        """
        Ratioed (log difference) method.
        
        Args:
            img_t1: Image at time 1
            img_t2: Image at time 2
        
        Returns:
            Change map
        """
        if img_t1.ndim == 3:
            ratio = np.log((img_t2 + 1e-8) / (img_t1 + 1e-8))
            ratio_combined = np.mean(np.abs(ratio), axis=0)
        else:
            ratio_combined = np.abs(np.log((img_t2 + 1e-8) / (img_t1 + 1e-8)))
        
        change_map = ratio_combined > self.threshold
        return change_map.astype(np.uint8)
    
    def detect_change(self, img_t1: np.ndarray, img_t2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect changes using selected method.
        
        Args:
            img_t1: Image at time 1
            img_t2: Image at time 2
        
        Returns:
            Tuple of (change map, metrics)
        """
        if self.method == 'difference':
            change_map = self.difference_method(img_t1, img_t2)
        elif self.method == 'ratioed':
            change_map = self.ratioed_method(img_t1, img_t2)
        else:
            change_map = self.difference_method(img_t1, img_t2)
        
        # Compute metrics
        metrics = {
            'method': self.method,
            'change_pixels': int(np.sum(change_map)),
            'change_percentage': float(100 * np.sum(change_map) / change_map.size),
            'total_pixels': int(change_map.size)
        }
        
        logger.info(f"Pixel-based change detection | Method: {self.method} | Changes: {metrics['change_percentage']:.2f}%")
        
        return change_map, metrics


class LandCoverTransition:
    """Analyze land cover class transitions."""
    
    def __init__(self, classes: list = None):
        """
        Initialize land cover analyzer.
        
        Args:
            classes: List of land cover class names
        """
        self.classes = classes or ['water', 'vegetation', 'urban', 'agriculture', 'bare_soil']
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
    
    def classify_pixels(self, img: np.ndarray) -> np.ndarray:
        """
        Classify pixels to land cover classes.
        Uses clustering-based classification.
        
        Args:
            img: Multi-band image
        
        Returns:
            Classification map
        """
        from sklearn.cluster import KMeans
        
        if img.ndim == 3:
            # Reshape to (pixels, bands)
            h, w = img.shape[1], img.shape[2]
            img_reshaped = img.transpose(1, 2, 0).reshape(-1, img.shape[0])
        else:
            h, w = img.shape
            img_reshaped = img.reshape(-1, 1)
        
        # Clustering
        kmeans = KMeans(n_clusters=self.num_classes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(img_reshaped)
        classification = labels.reshape(h, w)
        
        logger.info(f"Classified image to {self.num_classes} land cover classes")
        return classification
    
    def compute_transition_matrix(
        self, 
        class_t1: np.ndarray, 
        class_t2: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute transition matrix between two time points.
        
        Args:
            class_t1: Classification at time 1
            class_t2: Classification at time 2
        
        Returns:
            Tuple of (transition matrix, transition statistics)
        """
        # Ensure same shape
        if class_t1.shape != class_t2.shape:
            raise ValueError("Classification maps must have same shape")
        
        # Compute transition matrix
        transition_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        
        for i in range(self.num_classes):
            mask_t1 = class_t1 == i
            for j in range(self.num_classes):
                mask_t2 = class_t2 == j
                transition_matrix[i, j] = np.sum(mask_t1 & mask_t2)
        
        # Normalize by row
        transition_matrix_norm = transition_matrix.astype(float) / (transition_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        # Compute statistics
        stats = {
            'transition_matrix': transition_matrix,
            'transition_matrix_normalized': transition_matrix_norm
        }
        
        # Major transitions
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and transition_matrix[i, j] > 0:
                    pct = 100 * transition_matrix[i, j] / transition_matrix.sum()
                    if pct > 1:
                        logger.info(f"Transition {self.classes[i]} -> {self.classes[j]}: {pct:.2f}%")
        
        return transition_matrix, stats
    
    def get_change_summary(self, transition_matrix: np.ndarray) -> Dict:
        """
        Get summary of changes.
        
        Args:
            transition_matrix: Transition matrix
        
        Returns:
            Dictionary with change summary
        """
        diagonal = np.diag(transition_matrix)
        total = transition_matrix.sum()
        
        summary = {
            'total_pixels': int(total),
            'unchanged_pixels': int(diagonal.sum()),
            'changed_pixels': int(total - diagonal.sum()),
            'change_percentage': 100 * (total - diagonal.sum()) / (total + 1e-8),
            'unchanged_by_class': {self.classes[i]: int(diagonal[i]) for i in range(self.num_classes)}
        }
        
        logger.info(f"Change summary | Changed: {summary['change_percentage']:.2f}%")
        return summary


class TemporalChangeAnalysis:
    """Analyze temporal trends in change detection."""
    
    def __init__(self, window_size: int = 30, confidence_level: float = 0.95):
        """
        Initialize temporal analyzer.
        
        Args:
            window_size: Window size in days
            confidence_level: Statistical confidence level
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
    
    def analyze_temporal_trend(
        self, 
        change_maps: Dict[str, np.ndarray],
        dates: Dict[str, str]
    ) -> Dict:
        """
        Analyze temporal trends.
        
        Args:
            change_maps: Dictionary of date -> change map
            dates: Dictionary mapping to actual dates
        
        Returns:
            Dictionary with trend analysis
        """
        change_percentages = []
        for date, change_map in change_maps.items():
            pct = 100 * np.sum(change_map) / change_map.size
            change_percentages.append(pct)
        
        trend_analysis = {
            'mean_change': float(np.mean(change_percentages)),
            'std_change': float(np.std(change_percentages)),
            'min_change': float(np.min(change_percentages)),
            'max_change': float(np.max(change_percentages)),
            'trend': 'increasing' if change_percentages[-1] > change_percentages[0] else 'decreasing'
        }
        
        logger.info(f"Temporal trend: {trend_analysis['trend']} | Mean: {trend_analysis['mean_change']:.2f}%")
        return trend_analysis
