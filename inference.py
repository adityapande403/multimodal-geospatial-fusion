"""
Inference script for multimodal fusion model
Usage: python inference.py --model models/fusion_model.pth --input data/test.tif --output results/
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np

from src.fusion import MultimodalFusionCNN
from src.preprocessing import TIFFLoader, DataNormalizer
from src.utils import load_config, setup_logging
from src.evaluation import GeospatialMetrics

logger = logging.getLogger(__name__)


def main(args):
    """Main inference function."""
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = MultimodalFusionCNN(
        num_modalities=config['model']['params']['num_modalities'],
        feature_dim=config['model']['params']['feature_dim'],
        num_layers=config['model']['params']['num_layers'],
        attention_enabled=config['model']['attention']['enabled']
    ).to(device)
    
    # Load checkpoint
    if Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info(f"Model loaded from {args.model}")
    else:
        logger.warning(f"Model checkpoint not found at {args.model}, using random initialization")
    
    model.eval()
    
    # Load input data
    loader = TIFFLoader(config['data']['tiff'])
    
    if Path(args.input).exists():
        data, metadata = loader.load_tiff(args.input)
        logger.info(f"Input data loaded: {data.shape}")
        
        # Normalize
        normalizer = DataNormalizer(method=config['data']['normalization']['method'])
        data_norm = normalizer.normalize(data)
        
        # Convert to tensor
        data_tensor = torch.from_numpy(data_norm).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            predictions, features = model({'satellite': data_tensor})
        
        predictions_np = predictions.squeeze().cpu().numpy()
        logger.info(f"Predictions shape: {predictions_np.shape}")
        
        # Save results
        output_path = Path(args.output) / 'predictions.npy'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, predictions_np)
        logger.info(f"Predictions saved to {output_path}")
        
        # Print statistics
        print(f"\nInference Results:")
        print(f"  Min: {predictions_np.min():.6f}")
        print(f"  Max: {predictions_np.max():.6f}")
        print(f"  Mean: {predictions_np.mean():.6f}")
        print(f"  Std: {predictions_np.std():.6f}")
    else:
        logger.error(f"Input file not found: {args.input}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference with multimodal fusion model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/fusion_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input TIFF file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
