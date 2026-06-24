"""
TIFF File Handler and Geospatial Data Loader
Handles multi-temporal TIFF files with georeferencing and metadata extraction
"""

import logging
from typing import Tuple, Dict, Optional, List
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import rioxarray
from pathlib import Path

logger = logging.getLogger(__name__)


class TIFFLoader:
    """Load and process GeoTIFF files with full geospatial metadata preservation."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize TIFF loader.
        
        Args:
            config: Configuration dictionary with TIFF specifications
        """
        self.config = config or {}
        self.resolution = self.config.get('resolution', 0.5)
        self.crs = self.config.get('crs', 'EPSG:4326')
    
    def load_tiff(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a GeoTIFF file with metadata.
        
        Args:
            file_path: Path to the GeoTIFF file
        
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {file_path}")
        
        try:
            with rasterio.open(file_path) as src:
                data = src.read()  # Shape: (bands, height, width)
                
                metadata = {
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'crs': src.crs,
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'resolution': (src.res[0], src.res[1]),
                    'count': src.count,
                    'nodata': src.nodata,
                    'tags': src.tags(),
                    'file_path': str(file_path)
                }
                
                logger.info(f"Loaded TIFF: {file_path.name} | Shape: {data.shape} | CRS: {src.crs}")
                
                return data, metadata
        
        except Exception as e:
            logger.error(f"Error loading TIFF file {file_path}: {str(e)}")
            raise
    
    def load_multiple_tiffs(self, file_paths: List[str], modalities: List[str]) -> Dict:
        """
        Load multiple TIFF files for different modalities.
        
        Args:
            file_paths: List of TIFF file paths
            modalities: List of modality names (satellite, lidar, gpr, gnss, aerial)
        
        Returns:
            Dictionary with modality names as keys and (data, metadata) as values
        """
        multimodal_data = {}
        
        for file_path, modality in zip(file_paths, modalities):
            try:
                data, metadata = self.load_tiff(file_path)
                multimodal_data[modality] = {
                    'data': data,
                    'metadata': metadata
                }
            except Exception as e:
                logger.warning(f"Failed to load {modality} from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(multimodal_data)} modalities: {list(multimodal_data.keys())}")
        return multimodal_data
    
    def save_tiff(self, data: np.ndarray, output_path: str, metadata: Dict) -> None:
        """
        Save data as GeoTIFF with geospatial metadata.
        
        Args:
            data: Array to save (bands, height, width)
            output_path: Path for output file
            metadata: Metadata dictionary with CRS, transform, etc.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=metadata.get('crs'),
                transform=metadata.get('transform'),
                nodata=metadata.get('nodata')
            ) as dst:
                dst.write(data)
            
            logger.info(f"Saved GeoTIFF: {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving TIFF file {output_path}: {str(e)}")
            raise
    
    def get_band_info(self, data: np.ndarray, metadata: Dict) -> Dict:
        """
        Extract band information and statistics.
        
        Args:
            data: TIFF data array
            metadata: Metadata dictionary
        
        Returns:
            Dictionary with band statistics
        """
        band_info = {
            'num_bands': data.shape[0],
            'height': data.shape[1],
            'width': data.shape[2],
            'dtype': str(data.dtype),
            'bands': []
        }
        
        for i in range(data.shape[0]):
            band_data = data[i]
            band_info['bands'].append({
                'index': i,
                'min': float(np.nanmin(band_data)),
                'max': float(np.nanmax(band_data)),
                'mean': float(np.nanmean(band_data)),
                'std': float(np.nanstd(band_data)),
                'nodata_count': int(np.isnan(band_data).sum())
            })
        
        return band_info
    
    def validate_georeference(self, metadata: Dict) -> bool:
        """
        Validate georeference information.
        
        Args:
            metadata: Metadata dictionary
        
        Returns:
            True if georeference is valid
        """
        required_keys = ['crs', 'transform', 'bounds']
        has_all_keys = all(key in metadata for key in required_keys)
        
        if not has_all_keys:
            logger.warning("Missing required georeference metadata")
            return False
        
        if metadata['crs'] is None:
            logger.warning("CRS is None")
            return False
        
        return True


class MultimodalProcessor:
    """Process multimodal geospatial data from multiple sources."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize multimodal processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.loader = TIFFLoader(config)
    
    def align_arrays(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Align arrays to common shape.
        
        Args:
            arrays: Dictionary of modality -> array
        
        Returns:
            Dictionary of aligned arrays
        """
        from skimage.transform import resize
        
        # Find minimum dimensions
        min_height = min(arr.shape[1] for arr in arrays.values())
        min_width = min(arr.shape[2] for arr in arrays.values())
        
        aligned = {}
        for modality, array in arrays.items():
            if array.shape[1] != min_height or array.shape[2] != min_width:
                resized = resize(
                    array,
                    (array.shape[0], min_height, min_width),
                    order=1,
                    preserve_range=True
                )
                aligned[modality] = resized
            else:
                aligned[modality] = array
        
        logger.info(f"Aligned arrays to shape: ({len(arrays)}, {min_height}, {min_width})")
        return aligned
    
    def concatenate_modalities(self, multimodal_data: Dict, modality_order: List[str] = None) -> np.ndarray:
        """
        Concatenate multimodal data along channel dimension.
        
        Args:
            multimodal_data: Dictionary with modality data
            modality_order: Order to concatenate modalities
        
        Returns:
            Concatenated array (total_bands, height, width)
        """
        if modality_order is None:
            modality_order = list(multimodal_data.keys())
        
        # Align arrays first
        arrays = {mod: multimodal_data[mod]['data'] for mod in modality_order}
        aligned = self.align_arrays(arrays)
        
        # Concatenate along band dimension
        concatenated = np.concatenate([aligned[mod] for mod in modality_order], axis=0)
        
        logger.info(f"Concatenated multimodal data: {concatenated.shape}")
        return concatenated
