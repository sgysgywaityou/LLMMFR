"""
Training script for LLMMFR
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm import tqdm
import logging
from datetime import datetime

from config import Config
from models.llmmfr import LLMMFR
from data.dataset import FakeNewsDataset
from utils.metrics import compute_metrics


def setup_logging(config):
    """Setup logging configuration"""
    log_file = os.path.join(config.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_loss_mul = 0
    total_loss_domain = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['images'].to(device)
        ded_input_ids = batch['ded_input_ids'].to(device)
        ded_attention_mask = batch['ded_attention_mask'].to(device)
        red_input_ids = batch['red_input_ids'].to(device)
        red_attention_mask = batch['red_attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        output = model(
            input_ids, attention_mask, images,
            ded_input_ids, ded_attention_mask,
            red_input_ids, red_attention_mask,
            labels=labels
        )

        # Get loss
        loss = output['total_loss']
        loss_mul = output['loss_mul']
        loss_domain = output['loss_domain']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Update statistics
        total_loss += loss.item()
        total_loss_mul += loss_mul.item()
        total_loss_domain += loss_domain.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'loss_mul': f'{loss_mul.item():.4f}',
            'loss_domain': f'{loss_domain.item():.4f}'
        })

        # Log to wandb
        if batch_idx % 50 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/loss_mul': loss_mul.item(),
                'train/loss_domain': loss_domain.item(),
                'train/lr': optimizer.param_groups[0]['lr']
            })

    # Update Gumbel temperature
    model.update_temperature(epoch, config.epochs)

    avg_loss = total_loss / len(dataloader)
    avg_loss_mul = total_loss_mul / len(dataloader)
    avg_loss_domain = total_loss_domain / len(dataloader)

    return avg_loss, avg_loss_mul, avg_loss_domain


def evaluate(model, dataloader, device, logger):
    """Evaluate model on validation/test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            ded_input_ids = batch['ded_input_ids'].to(device)
            ded_attention_mask = batch['ded_attention_mask'].to(device)
            red_input_ids = batch['red_input_ids'].to(device)
            red_attention_mask = batch['red_attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(
                input_ids, attention_mask, images,
                ded_input_ids, ded_attention_mask,
                red_input_ids, red_attention_mask,
                labels=None
            )

            preds = torch.argmax(output['logits_final'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds)

    return metrics


def main(config):
    """Main training function"""

    # Setup
    logger = setup_logging(config)
    logger.info(f"Config: {config}")

    # Initialize wandb
    wandb.init(project="LLMMFR", config=vars(config))

    # Create datasets and dataloaders
    logger.info("Loading datasets...")
    train_dataset = FakeNewsDataset(config, split='train')
    val_dataset = FakeNewsDataset(config, split='val')
    test_dataset = FakeNewsDataset(config, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    logger.info("Creating model...")
    model = LLMMFR(config).to(config.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.warmup_steps, T_mult=2)

    # Training loop
    best_val_accuracy = 0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{config.epochs}")
        logger.info(f"{'=' * 50}")

        # Train
        train_loss, train_loss_mul, train_loss_domain = train_epoch(
            model, train_loader, optimizer, scheduler, None, config.device, logger, epoch
        )
        logger.info(
            f"Train - Loss: {train_loss:.4f}, Loss_mul: {train_loss_mul:.4f}, Loss_domain: {train_loss_domain:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, config.device, logger)
        logger.info(f"Val - Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                    f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss_epoch': train_loss,
            'train/loss_mul_epoch': train_loss_mul,
            'train/loss_domain_epoch': train_loss_domain,
            'val/accuracy': val_metrics['accuracy'],
            'val/precision': val_metrics['precision'],
            'val/recall': val_metrics['recall'],
            'val/f1': val_metrics['f1']
        })

        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'config': config
            }, os.path.join(config.checkpoint_dir, 'best_model.pt'))
            logger.info(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")

    logger.info(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")

    # Test best model
    logger.info("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, config.device, logger)
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")

    wandb.log({
        'test/accuracy': test_metrics['accuracy'],
        'test/precision': test_metrics['precision'],
        'test/recall': test_metrics['recall'],
        'test/f1': test_metrics['f1']
    })

    wandb.finish()

    return model, test_metrics


if __name__ == "__main__":
    config = Config()
    model, metrics = main(config)