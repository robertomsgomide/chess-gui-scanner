import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from labels import PIECE_LABELS, labels_to_fen
import logging
import numpy as np
import json
import pickle
import hashlib
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "num_classes": len(PIECE_LABELS),
        "dropout_rate1": 0.5,
        "dropout_rate2": 0.3,
        "use_batch_norm": True
    },
    "training": {
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "max_epochs": 100,
        "patience": 10,
        "min_epochs": 5
    },
    "data": {
        "img_size": 60,
        "validation_split": 0.2,
        "use_augmentation": True
    }
}

# Save and load configuration
def save_config(config: Dict, filename: str = "chess_classifier_config.json") -> None:
    """Save configuration to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename: str = "chess_classifier_config.json") -> Dict:
    """Load configuration from a JSON file or return default config."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

NUM_CLASSES = len(PIECE_LABELS) # 13

class ChessPieceDataset(Dataset):
    """Dataset for chess pieces."""
    def __init__(self, data_tuples: List[Tuple[torch.Tensor, int]], transform=None):
        """
        Args:
            data_tuples: List of (tensor, label_idx) pairs
            transform: Optional transform to apply to the images
        """
        self.data = data_tuples
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Skip transform if we're already dealing with a tensor and transform is provided
        # This prevents the "pic should be PIL Image" error
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        return image, label

class SimpleCNN(nn.Module):
    """
    An enhanced CNN for classification of chess squares.
    Features:
    - Deeper architecture with residual connections
    - Batch normalization and dropout for regularization
    - Adaptive pooling for flexible input sizes
    """
    def __init__(self, config: Dict = None):
        if config is None:
            config = DEFAULT_CONFIG["model"]
            
        super(SimpleCNN, self).__init__()
        num_classes = config.get("num_classes", NUM_CLASSES)
        dropout_rate1 = config.get("dropout_rate1", 0.5)
        dropout_rate2 = config.get("dropout_rate2", 0.3)
        use_batch_norm = config.get("use_batch_norm", True)
        
        # First block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # Second block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        # Third block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        # Adaptive pooling and fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128*4*4, 256)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        # Final layers
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class CNNClassifier:
    """
    A PyTorch-based CNN classifier for chess-square images.
    - Includes data augmentation and proper dataset handling
    - Configurable architecture and training parameters
    - Tracks and saves training metrics
    """
    def __init__(self, config_file: str = "chess_classifier_config.json"):
        self.config = load_config(config_file)
        save_config(self.config, config_file)  # Save the config in case it was loaded from defaults
        
        self.model = SimpleCNN(self.config["model"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Base transform for preprocessing
        img_size = self.config["data"]["img_size"]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Augmentation transform for training
        self.augment_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(5),  # Slight rotation
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Small translations and scaling
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Lighting variations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.label_to_idx = {lbl:i for i,lbl in enumerate(PIECE_LABELS)}
        self.idx_to_label = {i:lbl for lbl,i in self.label_to_idx.items()}
        
        # Load existing training data if available
        self.training_data_file = "training_data.pkl"
        self.training_data, self.validation_data = self._load_training_data()
        
        # Pending storage system
        self.pending_file = "pending_training_data.pkl"
        self.pending_threshold = 8  # Number of samples before triggering training
        self.pending_data = self._load_pending_data()
        
        if os.path.exists("chess_classifier.pt"):
            self.model.load_state_dict(torch.load("chess_classifier.pt", map_location=self.device))
            logger.info("Loaded existing model weights from chess_classifier.pt")
        else:
            logger.info("No existing model found, starting from scratch.")

    def _preprocess(self, pil_img):
        """Convert a PIL image to a tensor on the correct device."""
        tensor_img = self.transform(pil_img)
        return tensor_img.to(self.device)

    def _load_pending_data(self):
        """Load pending training data from disk."""
        if os.path.exists(self.pending_file):
            try:
                with open(self.pending_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded {len(data)} pending training positions")
                return data
            except Exception as e:
                logger.error(f"Error loading pending data: {e}")
                return {}
        return {}

    def _save_pending_data(self):
        """Save pending training data to disk."""
        try:
            with open(self.pending_file, 'wb') as f:
                pickle.dump(self.pending_data, f)
        except Exception as e:
            logger.error(f"Error saving pending data: {e}")

    def _load_training_data(self):
        """Load training and validation data from disk."""
        if os.path.exists(self.training_data_file):
            try:
                with open(self.training_data_file, 'rb') as f:
                    data = pickle.load(f)
                training_data = data.get('training_data', [])
                validation_data = data.get('validation_data', [])
                logger.info(f"Loaded {len(training_data)} training samples and {len(validation_data)} validation samples")
                return training_data, validation_data
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                return [], []
        return [], []

    def _save_training_data(self):
        """Save training and validation data to disk."""
        try:
            data = {
                'training_data': self.training_data,
                'validation_data': self.validation_data
            }
            with open(self.training_data_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(self.training_data)} training samples and {len(self.validation_data)} validation samples")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def _generate_position_hash(self, labels_2d, side_to_move='w', castling_rights='KQkq', ep_field='-'):
        """Generate a unique hash for a chess position based on its FEN."""
        try:
            # Create FEN from position
            fen = labels_to_fen(labels_2d, side_to_move, castling_rights, ep_field)
            # Use just the position part (before the first space) for uniqueness
            position_part = fen.split(' ')[0]
            return hashlib.md5(position_part.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating position hash: {e}")
            # Fallback: create hash from string representation of labels
            position_str = str(labels_2d)
            return hashlib.md5(position_str.encode()).hexdigest()

    def _add_to_pending(self, squares_2d, labels_2d, position_hash):
        """Add correction data to pending storage."""
        corrections = []
        for r in range(8):
            for c in range(8):
                pil_img = squares_2d[r][c]
                lbl = labels_2d[r][c]
                if lbl not in self.label_to_idx:
                    continue
                
                # Only store corrections (where prediction != actual label)
                predicted_lbl = self.predict_label(pil_img)
                if predicted_lbl != lbl:
                    corrections.append({
                        'image': pil_img,
                        'label': lbl,
                        'label_idx': self.label_to_idx[lbl],
                        'predicted': predicted_lbl,
                        'position': (r, c)
                    })
        
        if corrections:
            self.pending_data[position_hash] = {
                'corrections': corrections,
                'position_labels': [row[:] for row in labels_2d],  # Deep copy
                'timestamp': np.datetime64('now').item()
            }
            self._save_pending_data()
            logger.info(f"Added {len(corrections)} corrections from position {position_hash[:8]}... to pending data")
            return len(corrections)
        return 0

    def _process_pending_data(self, force=False):
        """Process accumulated pending data when threshold is reached."""
        if not force and len(self.pending_data) < self.pending_threshold:
            return
        
        logger.info(f"Processing {len(self.pending_data)} pending positions for training")
        
        # Extract all corrections from pending data
        total_corrections = 0
        for position_hash, position_data in self.pending_data.items():
            corrections = position_data['corrections']
            for correction in corrections:
                pil_img = correction['image']
                y_idx = correction['label_idx']
                
                # Split data into training and validation
                if random.random() < (1 - self.config["data"]["validation_split"]):
                    self.training_data.append((pil_img, y_idx))
                else:
                    self.validation_data.append((pil_img, y_idx))
                total_corrections += 1
        
        logger.info(f"Added {total_corrections} corrections to training data")
        
        # Clear pending data after processing
        self.pending_data.clear()
        self._save_pending_data()
        
        # Now proceed with normal training
        return total_corrections

    def predict_label(self, pil_img):
        """Predict the label for a single chess piece image."""
        try:
            if not os.path.exists("chess_classifier.pt"):
                return "empty"

            self.model.eval()
            with torch.no_grad():
                x = self._preprocess(pil_img).unsqueeze(0)
                logits = self.model(x)
                pred_idx = logits.argmax(dim=1).item()
            return self.idx_to_label.get(pred_idx, "empty")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "empty"
    
    def predict_batch(self, pil_images):
        """Predict labels for a batch of chess piece images."""
        if not os.path.exists("chess_classifier.pt"):
            return ["empty"] * len(pil_images)
            
        self.model.eval()
        results = []
        
        # Process in batches of 16
        batch_size = 16
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            with torch.no_grad():
                # Apply transforms to all images in batch, then stack
                transformed_imgs = [self.transform(img) for img in batch]
                batch_tensor = torch.stack(transformed_imgs).to(self.device)
                logits = self.model(batch_tensor)
                pred_indices = logits.argmax(dim=1).tolist()
                
                for idx in pred_indices:
                    results.append(self.idx_to_label.get(idx, "empty"))
                
        return results
    
    def train_on_data(self, squares_2d, labels_2d, side_to_move='w', castling_rights='KQkq', ep_field='-'):
        """
        Train the model on new data from the chess board using pending storage system.
        
        Args:
            squares_2d: 2D array of PIL images
            labels_2d: 2D array of piece labels  
            side_to_move: Side to move ('w' or 'b')
            castling_rights: Castling rights string
            ep_field: En passant field
        """
        # Generate unique hash for this position
        position_hash = self._generate_position_hash(labels_2d, side_to_move, castling_rights, ep_field)
        
        # Check if we've already processed this exact position
        if position_hash in self.pending_data:
            logger.info(f"Position {position_hash[:8]}... already in pending data, skipping")
            return
        
        # Add corrections to pending storage
        corrections_added = self._add_to_pending(squares_2d, labels_2d, position_hash)
        
        if corrections_added == 0:
            logger.info("No corrections needed for this position")
            return
        
        # Check for immediate training if many corrections (rich training example)
        if corrections_added >= 15:
            logger.info(f"Position has {corrections_added} corrections, training immediately (bypassing pending storage)")
            
            # Extract corrections directly and train immediately (bypass pending system)
            position_data = self.pending_data[position_hash]
            corrections = position_data['corrections']
            
            # Add corrections directly to training data
            for correction in corrections:
                pil_img = correction['image']
                y_idx = correction['label_idx']
                
                # Split data into training and validation
                if random.random() < (1 - self.config["data"]["validation_split"]):
                    self.training_data.append((pil_img, y_idx))
                else:
                    self.validation_data.append((pil_img, y_idx))
            
            # Remove from pending since we're processing immediately
            del self.pending_data[position_hash]
            self._save_pending_data()
            
            logger.info(f"Added {corrections_added} corrections directly to training data")
            
        # Check if we've reached the threshold for training
        elif len(self.pending_data) >= self.pending_threshold:
            logger.info(f"Reached pending threshold ({len(self.pending_data)} positions), processing for training")
            total_corrections = self._process_pending_data()
            
            if total_corrections == 0:
                logger.info("No corrections to process")
                return
        else:
            logger.info(f"Added to pending storage. {len(self.pending_data)}/{self.pending_threshold} positions accumulated")
            return

        # If we reach here, we have enough data to train
        if len(self.training_data) < 8:
            logger.info(f"Still not enough data to train ({len(self.training_data)} samples)")
            return

        # If no validation data, move some training data to validation
        if len(self.validation_data) == 0 and len(self.training_data) > 8:
            num_val_samples = min(8, int(len(self.training_data) * self.config["data"]["validation_split"]))
            val_indices = random.sample(range(len(self.training_data)), num_val_samples)
            self.validation_data = [self.training_data[i] for i in sorted(val_indices, reverse=True)]
            for i in sorted(val_indices, reverse=True):
                del self.training_data[i]

        # Execute the training
        self._execute_training()

    def get_pending_status(self):
        """Get information about pending training data."""
        total_corrections = 0
        position_info = []
        
        for position_hash, position_data in self.pending_data.items():
            corrections = position_data['corrections']
            total_corrections += len(corrections)
            position_info.append({
                'hash': position_hash[:8] + '...',
                'corrections': len(corrections),
                'timestamp': position_data.get('timestamp', 'unknown')
            })
        
        return {
            'total_positions': len(self.pending_data),
            'total_corrections': total_corrections,
            'threshold': self.pending_threshold,
            'positions': position_info
        }

    def force_process_pending(self):
        """Manually trigger processing of pending data regardless of threshold."""
        if len(self.pending_data) == 0:
            logger.info("No pending data to process")
            return False
        
        logger.info(f"Force processing {len(self.pending_data)} pending positions")
        total_corrections = self._process_pending_data()
        
        if total_corrections > 0 and len(self.training_data) >= 8:
            # Continue with normal training logic
            if len(self.validation_data) == 0 and len(self.training_data) > 8:
                num_val_samples = min(8, int(len(self.training_data) * self.config["data"]["validation_split"]))
                val_indices = random.sample(range(len(self.training_data)), num_val_samples)
                self.validation_data = [self.training_data[i] for i in sorted(val_indices, reverse=True)]
                for i in sorted(val_indices, reverse=True):
                    del self.training_data[i]
            
            # Execute the training portion from train_on_data
            self._execute_training()
            return True
        
        return False

    def _execute_training(self):
        """Execute the actual training process."""
        # Create datasets and dataloaders
        use_augmentation = self.config["data"]["use_augmentation"]
        train_transform = self.augment_transform if use_augmentation else self.transform
        train_dataset = ChessPieceDataset(self.training_data, transform=train_transform)
        train_loader = DataLoader(train_dataset, 
                               batch_size=self.config["training"]["batch_size"],
                               shuffle=True)
        
        val_dataset = ChessPieceDataset(self.validation_data, transform=self.transform)
        val_loader = DataLoader(val_dataset, 
                             batch_size=self.config["training"]["batch_size"],
                             shuffle=False)
        
        # Training setup
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config["training"]["learning_rate"], 
            weight_decay=self.config["training"]["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        # Training parameters
        max_epochs = min(self.config["training"]["max_epochs"], 
                         self.config["training"]["min_epochs"] + len(self.training_data) // 8)
        patience = self.config["training"]["patience"]
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Store metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            metrics["train_loss"].append(avg_train_loss)
            metrics["lr"].append(optimizer.param_groups[0]["lr"])
            
            # Validation phase
            if len(val_loader) > 0:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        logits = self.model(x_batch)
                        loss = criterion(logits, y_batch)
                        val_loss += loss.item()
                        
                        _, predicted = logits.max(1)
                        total += y_batch.size(0)
                        correct += predicted.eq(y_batch).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100. * correct / total if total > 0 else 0
                
                metrics["val_loss"].append(avg_val_loss)
                metrics["val_acc"].append(val_acc)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model (keep only the essential checkpoint)
                    torch.save(self.model.state_dict(), "chess_classifier_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience and epoch >= self.config["training"]["min_epochs"]:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

                logger.info(f"Epoch {epoch+1}/{max_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                # If no validation data, just log training metrics
                logger.info(f"Epoch {epoch+1}/{max_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save metrics
        with open("chess_classifier_metrics.json", "w") as f:
            json.dump(metrics, f)
            
        # Load best model if it exists
        if os.path.exists("chess_classifier_best.pt"):
            self.model.load_state_dict(torch.load("chess_classifier_best.pt", map_location=self.device))
            logger.info("Loaded best model weights from chess_classifier_best.pt")
        
        torch.save(self.model.state_dict(), "chess_classifier.pt")
        logger.info("Model weights saved to chess_classifier.pt")
        
        # Save training data for future sessions
        self._save_training_data()