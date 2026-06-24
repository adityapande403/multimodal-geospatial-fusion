"""
Main training script for multimodal fusion model
Usage: python train.py --config configs/config.yaml --epochs 200 --batch-size 32
"""

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.fusion import MultimodalFusionCNN
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def main(args):
    """Main training function."""
    
    # Setup logging
    setup_logging(log_level='INFO', log_file=f"logs/training_{Path.cwd().name}.log")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    logger.info(f"Training started with {args.epochs} epochs, batch size {args.batch_size}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    # Training loop (placeholder - implement with actual data loader)
    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs}")
        scheduler.step()
    
    # Save model
    output_path = Path(args.output) / "fusion_model_final.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multimodal geospatial fusion model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for saved models'
    )
    
    args = parser.parse_args()
    main(args)
