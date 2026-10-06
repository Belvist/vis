"""Main training script for EarAI"""
import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path

from earai.training.student import create_student_model
from earai.training.teachers import create_teachers
from earai.training.losses import create_losses
from earai.training.trainer import create_trainer, create_dataloaders
from earai.decoder.heads import create_decoder


def load_config(config_path: str) -> dict:
    """Load training config from YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='EarAI Training')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['device'] = args.device
    
    # Create models
    print("Creating student model...")
    student = create_student_model(config).to(args.device)
    
    print("Creating decoder...")
    decoder = create_decoder(config).to(args.device)
    
    print("Loading teachers...")
    teachers, ocr = create_teachers(args.device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(config, student, decoder, teachers, ocr_teacher=ocr)
    
    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Train
        train_losses = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch} train: {train_losses}")
        
        # Validate
        if epoch % 5 == 0:
            val_losses = trainer.validate(val_loader)
            print(f"Epoch {epoch} val: {val_losses}")
        
        # Checkpoint
        if epoch % 10 == 0:
            trainer.save_checkpoint(f'epoch_{epoch}')
    
    # Final checkpoint
    trainer.save_checkpoint('final')
    print("Training complete!")


if __name__ == '__main__':
    main()