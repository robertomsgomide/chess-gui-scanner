import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from labels import PIECE_LABELS
import logging
import numpy as np
import json
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
        self.training_data = []
        self.validation_data = []
        
        if os.path.exists("chess_classifier.pt"):
            self.model.load_state_dict(torch.load("chess_classifier.pt", map_location=self.device))
            logger.info("Loaded existing model weights from chess_classifier.pt")
        else:
            logger.info("No existing model found, starting from scratch.")

    def _preprocess(self, pil_img):
        """Convert a PIL image to a tensor on the correct device."""
        tensor_img = self.transform(pil_img)
        return tensor_img.to(self.device)

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
                tensors = [self._preprocess(img).unsqueeze(0) for img in batch]
                batch_tensor = torch.cat(tensors, dim=0)
                logits = self.model(batch_tensor)
                pred_indices = logits.argmax(dim=1).tolist()
                
                for idx in pred_indices:
                    results.append(self.idx_to_label.get(idx, "empty"))
                
        return results
    
    def train_on_data(self, squares_2d, labels_2d):
        """Train the model on new data from the chess board."""
        new_samples_added = 0
        for r in range(8):
            for c in range(8):
                pil_img = squares_2d[r][c]
                lbl = labels_2d[r][c]
                if lbl not in self.label_to_idx:
                    continue
                # Only add samples where prediction != label
                if self.predict_label(pil_img) == lbl:
                    continue  # Skip already correct predictions
                
                # Store the PIL image directly, not the tensor
                y_idx = self.label_to_idx[lbl]
                # Split data into training and validation
                if random.random() < (1 - self.config["data"]["validation_split"]):
                    self.training_data.append((pil_img, y_idx))
                else:
                    self.validation_data.append((pil_img, y_idx))
                new_samples_added += 1

        if len(self.training_data) < 8:
            logger.info(f"Not enough data to train ({new_samples_added} new samples)")
            return

        # If no validation data, move some training data to validation
        if len(self.validation_data) == 0 and len(self.training_data) > 8:
            num_val_samples = min(8, int(len(self.training_data) * self.config["data"]["validation_split"]))
            val_indices = random.sample(range(len(self.training_data)), num_val_samples)
            self.validation_data = [self.training_data[i] for i in sorted(val_indices, reverse=True)]
            for i in sorted(val_indices, reverse=True):
                del self.training_data[i]

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
                    # Save best model
                    torch.save(self.model.state_dict(), "chess_classifier_best.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'config': self.config
                    }, "chess_classifier_checkpoint.pt")
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