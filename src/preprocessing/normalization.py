"""
Data Normalization and Standardization Module
Implements multiple normalization strategies for geospatial data preprocessing
"""

import logging
from typing import Tuple, Dict, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize and standardize geospatial data."""
    
    def __init__(self, method: str = 'minmax', percentile_range: Tuple[float, float] = (2, 98)):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('minmax', 'standardization', 'percentile', 'robust')
            percentile_range: Percentile range for percentile-based normalization
        """
        self.method = method
        self.percentile_range = percentile_range
        self.scalers = {}
        self.params = {}
    
    def minmax_normalization(
        self, 
        data: np.ndarray, 
        feature_range: Tuple[float, float] = (0, 1),
        clip: bool = True
    ) -> np.ndarray:
        """
        Min-Max normalization to [0, 1] range.
        
        Args:
            data: Input array (can be multi-dimensional)
            feature_range: Target range
            clip: Whether to clip outliers
        
        Returns:
            Normalized array
        """
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        
        if data_max == data_min:
            logger.warning("Data min equals max, returning zeros")
            return np.zeros_like(data)
        
        normalized = (data - data_min) / (data_max - data_min)
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        if clip:
            normalized = np.clip(normalized, feature_range[0], feature_range[1])
        
        self.params['method'] = 'minmax'
        self.params['data_min'] = float(data_min)
        self.params['data_max'] = float(data_max)
        self.params['feature_range'] = feature_range
        
        logger.info(f"Min-Max normalization: [{data_min:.4f}, {data_max:.4f}] -> {feature_range}")
        return normalized
    
    def standardization(self, data: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Standardization (z-score normalization).
        
        Args:
            data: Input array
            epsilon: Small constant to avoid division by zero
        
        Returns:
            Standardized array
        """
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std < epsilon:
            logger.warning("Standard deviation too small")
            return data
        
        standardized = (data - mean) / (std + epsilon)
        
        self.params['method'] = 'standardization'
        self.params['mean'] = float(mean)
        self.params['std'] = float(std)
        
        logger.info(f"Standardization: mean={mean:.4f}, std={std:.4f}")
        return standardized
    
    def percentile_normalization(self, data: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Percentile-based normalization (robust to outliers).
        
        Args:
            data: Input array
            clip: Whether to clip to percentile range
        
        Returns:
            Normalized array
        """
        p_low, p_high = self.percentile_range
        vmin = np.nanpercentile(data, p_low)
        vmax = np.nanpercentile(data, p_high)
        
        normalized = (data - vmin) / (vmax - vmin + 1e-8)
        
        if clip:
            normalized = np.clip(normalized, 0, 1)
        
        self.params['method'] = 'percentile'
        self.params['percentile_low'] = float(vmin)
        self.params['percentile_high'] = float(vmax)
        self.params['percentile_range'] = self.percentile_range
        
        logger.info(f"Percentile normalization: p[{p_low},{p_high}] = [{vmin:.4f}, {vmax:.4f}]")
        return normalized
    
    def robust_normalization(self, data: np.ndarray) -> np.ndarray:
        """
        Robust normalization using median and IQR.
        
        Args:
            data: Input array
        
        Returns:
            Normalized array
        """
        median = np.nanmedian(data)
        q75 = np.nanpercentile(data, 75)
        q25 = np.nanpercentile(data, 25)
        iqr = q75 - q25
        
        if iqr < 1e-8:
            logger.warning("IQR too small")
            return data
        
        normalized = (data - median) / iqr
        
        self.params['method'] = 'robust'
        self.params['median'] = float(median)
        self.params['iqr'] = float(iqr)
        self.params['q25'] = float(q25)
        self.params['q75'] = float(q75)
        
        logger.info(f"Robust normalization: median={median:.4f}, IQR={iqr:.4f}")
        return normalized
    
    def normalize(self, data: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Apply selected normalization method.
        
        Args:
            data: Input array
            clip: Whether to clip values
        
        Returns:
            Normalized array
        """
        # Handle NaN values
        nan_mask = np.isnan(data)
        data_clean = data.copy()
        
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} NaN values, will be handled separately")
            # Replace NaN temporarily for normalization
            data_clean[nan_mask] = np.nanmedian(data)
        
        if self.method == 'minmax':
            normalized = self.minmax_normalization(data_clean, clip=clip)
        elif self.method == 'standardization':
            normalized = self.standardization(data_clean)
        elif self.method == 'percentile':
            normalized = self.percentile_normalization(data_clean, clip=clip)
        elif self.method == 'robust':
            normalized = self.robust_normalization(data_clean)
        else:
            logger.warning(f"Unknown normalization method: {self.method}, using minmax")
            normalized = self.minmax_normalization(data_clean, clip=clip)
        
        # Restore NaN values
        normalized[nan_mask] = np.nan
        
        return normalized
    
    def normalize_multimodal(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize multimodal data independently.
        
        Args:
            data_dict: Dictionary of modality name -> data array
        
        Returns:
            Dictionary of normalized arrays
        """
        normalized_data = {}
        
        for modality, data in data_dict.items():
            normalized_data[modality] = self.normalize(data)
            logger.info(f"Normalized {modality}: shape {data.shape}")
        
        return normalized_data
    
    def get_normalization_params(self) -> Dict:
        """
        Get normalization parameters for inverse transformation.
        
        Returns:
            Dictionary of normalization parameters
        """
        return self.params.copy()


class BandNormalizer:
    """Normalize individual bands of multi-band images."""
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize band normalizer.
        
        Args:
            method: Normalization method
        """
        self.method = method
        self.band_params = {}
    
    def normalize_bands(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize each band independently.
        
        Args:
            data: Multi-band array (bands, height, width)
        
        Returns:
            Tuple of (normalized data, normalization parameters)
        """
        if data.ndim != 3:
            raise ValueError("Expected 3D array (bands, height, width)")
        
        normalized = np.zeros_like(data, dtype=np.float32)
        params = {}
        
        normalizer = DataNormalizer(method=self.method)
        
        for i in range(data.shape[0]):
            normalized[i] = normalizer.normalize(data[i])
            params[f'band_{i}'] = normalizer.get_normalization_params()
            logger.info(f"Normalized band {i}")
        
        self.band_params = params
        return normalized, params
    
    def compute_spectral_indices(self, data: np.ndarray, bands_config: Dict) -> Dict[str, np.ndarray]:
        """
        Compute spectral indices from normalized data.
        
        Args:
            data: Normalized multi-band array
            bands_config: Configuration for band indices
        
        Returns:
            Dictionary of spectral indices
        """
        indices = {}
        
        # Assuming standard band order: Red, Green, Blue, NIR
        if data.shape[0] >= 4:
            red = data[0]
            nir = data[3]
            
            # NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir - red) / (nir + red + 1e-8)
            indices['ndvi'] = ndvi
            logger.info("Computed NDVI")
            
            # Additional indices if data available
            if data.shape[0] >= 5:
                swir = data[4]
                # NDBI (Normalized Difference Built-up Index)
                ndbi = (swir - nir) / (swir + nir + 1e-8)
                indices['ndbi'] = ndbi
                logger.info("Computed NDBI")
        
        return indices
