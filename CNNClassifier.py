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
import sqlite3
import hashlib
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
from collections import Counter

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

class ChessDataDB:
    """SQLite database handler for chess training data."""
    
    def __init__(self, db_path: str = "chess_training_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create samples table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_data BLOB NOT NULL,
                    label_idx INTEGER NOT NULL,
                    label_name TEXT NOT NULL,
                    split TEXT NOT NULL CHECK (split IN ('train', 'val')),
                    position_hash TEXT,
                    timestamp REAL DEFAULT (julianday('now')),
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL
                )
            ''')
            
            # Create pending positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_positions (
                    position_hash TEXT PRIMARY KEY,
                    position_labels TEXT NOT NULL,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Create pending corrections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_hash TEXT NOT NULL,
                    image_data BLOB NOT NULL,
                    label_idx INTEGER NOT NULL,
                    label_name TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    row_pos INTEGER NOT NULL,
                    col_pos INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    FOREIGN KEY (position_hash) REFERENCES pending_positions(position_hash)
                )
            ''')
            
            # Create indices for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label_idx)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_corrections_hash ON pending_corrections(position_hash)')
            
            conn.commit()
    
    def add_sample(self, pil_image: Image.Image, label_idx: int, label_name: str, 
                   split: str, position_hash: Optional[str] = None) -> int:
        """Add a single sample to the database."""
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO samples (image_data, label_idx, label_name, split, position_hash, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (img_data, label_idx, label_name, split, position_hash, 
                  pil_image.width, pil_image.height))
            return cursor.lastrowid
    
    def get_sample_count(self, split: Optional[str] = None) -> int:
        """Get the count of samples, optionally filtered by split."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if split:
                cursor.execute('SELECT COUNT(*) FROM samples WHERE split = ?', (split,))
            else:
                cursor.execute('SELECT COUNT(*) FROM samples')
            return cursor.fetchone()[0]
    
    def get_class_distribution(self, split: str) -> Dict[int, int]:
        """Get the distribution of classes in a split."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT label_idx, COUNT(*) as count 
                FROM samples 
                WHERE split = ? 
                GROUP BY label_idx
            ''', (split,))
            return dict(cursor.fetchall())
    
    def get_samples_batch(self, split: str, batch_size: int, offset: int) -> List[Tuple[bytes, int]]:
        """Get a batch of samples from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_data, label_idx 
                FROM samples 
                WHERE split = ? 
                ORDER BY id 
                LIMIT ? OFFSET ?
            ''', (split, batch_size, offset))
            return cursor.fetchall()
    
    def clear_samples(self, split: Optional[str] = None):
        """Clear samples from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if split:
                cursor.execute('DELETE FROM samples WHERE split = ?', (split,))
            else:
                cursor.execute('DELETE FROM samples')
            conn.commit()
    
    def add_pending_position(self, position_hash: str, position_labels: List[List[str]]):
        """Add a pending position to the database."""
        labels_json = json.dumps(position_labels)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO pending_positions (position_hash, position_labels)
                VALUES (?, ?)
            ''', (position_hash, labels_json))
            conn.commit()
    
    def add_pending_correction(self, position_hash: str, pil_image: Image.Image, 
                             label_idx: int, label_name: str, predicted_label: str,
                             row: int, col: int):
        """Add a pending correction to the database."""
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pending_corrections 
                (position_hash, image_data, label_idx, label_name, predicted_label, 
                 row_pos, col_pos, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (position_hash, img_data, label_idx, label_name, predicted_label,
                  row, col, pil_image.width, pil_image.height))
            conn.commit()
    
    def get_pending_positions(self) -> List[Tuple[str, List[List[str]]]]:
        """Get all pending positions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT position_hash, position_labels FROM pending_positions')
            results = []
            for row in cursor.fetchall():
                position_hash, labels_json = row
                labels = json.loads(labels_json)
                results.append((position_hash, labels))
            return results
    
    def get_pending_corrections(self, position_hash: str) -> List[Dict]:
        """Get all corrections for a specific position."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_data, label_idx, label_name, predicted_label, 
                       row_pos, col_pos, width, height
                FROM pending_corrections
                WHERE position_hash = ?
            ''', (position_hash,))
            
            corrections = []
            for row in cursor.fetchall():
                img_data, label_idx, label_name, predicted_label, row_pos, col_pos, w, h = row
                # Convert bytes back to PIL image
                pil_image = Image.open(io.BytesIO(img_data))
                corrections.append({
                    'image': pil_image,
                    'label': label_name,
                    'label_idx': label_idx,
                    'predicted': predicted_label,
                    'position': (row_pos, col_pos)
                })
            return corrections
    
    def clear_pending_data(self):
        """Clear all pending data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM pending_corrections')
            cursor.execute('DELETE FROM pending_positions')
            conn.commit()
    
    def position_exists_in_pending(self, position_hash: str) -> bool:
        """Check if a position exists in pending data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM pending_positions WHERE position_hash = ?', (position_hash,))
            return cursor.fetchone()[0] > 0
    
    def get_pending_count(self) -> Tuple[int, int]:
        """Get count of pending positions and total corrections."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM pending_positions')
            position_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM pending_corrections')
            correction_count = cursor.fetchone()[0]
            return position_count, correction_count

class ChessPieceDataset(Dataset):
    """Dataset for chess pieces with lazy loading from SQLite."""
    
    def __init__(self, db: ChessDataDB, split: str, transform=None):
        """
        Args:
            db: ChessDataDB instance
            split: 'train' or 'val'
            transform: Optional transform to apply to the images
        """
        self.db = db
        self.split = split
        self.transform = transform
        self.length = db.get_sample_count(split)
        
        # Cache for recently loaded samples (LRU-style)
        self.cache_size = 100
        self.cache = {}
        self.cache_order = []
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            image, label = self.cache[idx]
        else:
            # Load from database
            samples = self.db.get_samples_batch(self.split, 1, idx)
            if not samples:
                raise IndexError(f"Index {idx} out of range")
            
            img_data, label = samples[0]
            # Convert bytes back to PIL image
            image = Image.open(io.BytesIO(img_data))
            
            # Add to cache
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                oldest_idx = self.cache_order.pop(0)
                del self.cache[oldest_idx]
            
            self.cache[idx] = (image, label)
            self.cache_order.append(idx)
        
        # Apply transform if provided
        if self.transform:
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
    - Uses SQLite for scalable data persistence with lazy loading
    - Implements class balancing for handling dataset imbalance
    - Includes data augmentation and proper dataset handling
    - Configurable architecture and training parameters
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
        
        # Initialize database backend
        self.db = ChessDataDB()
        
        # Pending storage system
        self.pending_threshold = 8  # Number of samples before triggering training
        
        if os.path.exists("chess_classifier.pt"):
            self.model.load_state_dict(torch.load("chess_classifier.pt", map_location=self.device))
            logger.info("Loaded existing model weights from chess_classifier.pt")
        else:
            logger.info("No existing model found, starting from scratch.")
            
        # Log current data statistics
        train_count = self.db.get_sample_count('train')
        val_count = self.db.get_sample_count('val')
        logger.info(f"Database contains {train_count} training and {val_count} validation samples")
    


    def _preprocess(self, pil_img):
        """Convert a PIL image to a tensor on the correct device."""
        tensor_img = self.transform(pil_img)
        return tensor_img.to(self.device)

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
        corrections_count = 0
        
        # First add the position to pending positions table
        self.db.add_pending_position(position_hash, labels_2d)
        
        for r in range(8):
            for c in range(8):
                pil_img = squares_2d[r][c]
                lbl = labels_2d[r][c]
                if lbl not in self.label_to_idx:
                    continue
                
                # Only store corrections (where prediction != actual label)
                predicted_lbl = self.predict_label(pil_img)
                if predicted_lbl != lbl:
                    self.db.add_pending_correction(
                        position_hash, pil_img, self.label_to_idx[lbl],
                        lbl, predicted_lbl, r, c
                    )
                    corrections_count += 1
        
        if corrections_count > 0:
            logger.info(f"Added {corrections_count} corrections from position {position_hash[:8]}... to pending data")
        
        return corrections_count

    def _process_pending_data(self, force=False):
        """Process accumulated pending data when threshold is reached."""
        position_count, _ = self.db.get_pending_count()
        
        if not force and position_count < self.pending_threshold:
            return 0
        
        logger.info(f"Processing {position_count} pending positions for training")
        
        # Get all pending positions
        pending_positions = self.db.get_pending_positions()
        total_corrections = 0
        
        for position_hash, position_labels in pending_positions:
            corrections = self.db.get_pending_corrections(position_hash)
            
            for correction in corrections:
                pil_img = correction['image']
                label_idx = correction['label_idx']
                label_name = correction['label']
                
                # Split data into training and validation
                split = 'val' if random.random() < self.config["data"]["validation_split"] else 'train'
                self.db.add_sample(pil_img, label_idx, label_name, split, position_hash)
                total_corrections += 1
        
        logger.info(f"Added {total_corrections} corrections to training data")
        
        # Clear pending data after processing
        self.db.clear_pending_data()
        
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
        if self.db.position_exists_in_pending(position_hash):
            logger.info(f"Position {position_hash[:8]}... already in pending data, skipping")
            return
        
        # Add corrections to pending storage
        corrections_added = self._add_to_pending(squares_2d, labels_2d, position_hash)
        
        if corrections_added == 0:
            logger.info("No corrections needed for this position")
            return
        
        # Check for immediate training if many corrections (rich training example)
        if corrections_added >= 15:
            logger.info(f"Position has {corrections_added} corrections, training immediately")
            
            # Get corrections for immediate processing
            corrections = self.db.get_pending_corrections(position_hash)
            
            # Add corrections directly to training data
            for correction in corrections:
                pil_img = correction['image']
                label_idx = correction['label_idx']
                label_name = correction['label']
                
                # Split data into training and validation
                split = 'val' if random.random() < self.config["data"]["validation_split"] else 'train'
                self.db.add_sample(pil_img, label_idx, label_name, split, position_hash)
            
            # Remove from pending since we're processing immediately
            self.db.clear_pending_data()  # This clears all pending data
            
            logger.info(f"Added {corrections_added} corrections directly to training data")
            
        # Check if we've reached the threshold for training
        else:
            position_count, _ = self.db.get_pending_count()
            if position_count >= self.pending_threshold:
                logger.info(f"Reached pending threshold ({position_count} positions), processing for training")
                total_corrections = self._process_pending_data()
                
                if total_corrections == 0:
                    logger.info("No corrections to process")
                    return
            else:
                logger.info(f"Added to pending storage. {position_count}/{self.pending_threshold} positions accumulated")
                return

        # If we reach here, check if we have enough data to train
        train_count = self.db.get_sample_count('train')
        if train_count < 8:
            logger.info(f"Still not enough data to train ({train_count} samples)")
            return

        # Execute the training
        self._execute_training()

    def get_pending_status(self):
        """Get information about pending training data."""
        position_count, correction_count = self.db.get_pending_count()
        
        # Get details about each position
        pending_positions = self.db.get_pending_positions()
        position_info = []
        
        for position_hash, _ in pending_positions:
            corrections = self.db.get_pending_corrections(position_hash)
            position_info.append({
                'hash': position_hash[:8] + '...',
                'corrections': len(corrections),
            })
        
        return {
            'total_positions': position_count,
            'total_corrections': correction_count,
            'threshold': self.pending_threshold,
            'positions': position_info
        }

    def force_process_pending(self):
        """Manually trigger processing of pending data regardless of threshold."""
        position_count, _ = self.db.get_pending_count()
        
        if position_count == 0:
            logger.info("No pending data to process")
            return False
        
        logger.info(f"Force processing {position_count} pending positions")
        total_corrections = self._process_pending_data(force=True)
        
        if total_corrections > 0 and self.db.get_sample_count('train') >= 8:
            # Execute the training
            self._execute_training()
            return True
        
        return False

    def _execute_training(self):
        """Execute the actual training process with class balancing."""
        # Get data counts
        train_count = self.db.get_sample_count('train')
        val_count = self.db.get_sample_count('val')
        
        if train_count == 0:
            logger.warning("No training data available")
            return
            
        logger.info(f"Starting training with {train_count} training and {val_count} validation samples")
        
        # Calculate class weights for balanced training
        class_distribution = self.db.get_class_distribution('train')
        class_weights = self._calculate_class_weights(class_distribution)
        
        # Create datasets and dataloaders
        use_augmentation = self.config["data"]["use_augmentation"]
        train_transform = self.augment_transform if use_augmentation else self.transform
        
        train_dataset = ChessPieceDataset(self.db, 'train', transform=train_transform)
        train_loader = DataLoader(train_dataset, 
                               batch_size=self.config["training"]["batch_size"],
                               shuffle=True,
                               num_workers=0)  # SQLite doesn't support multi-threading well
        
        val_dataset = ChessPieceDataset(self.db, 'val', transform=self.transform)
        val_loader = DataLoader(val_dataset, 
                             batch_size=self.config["training"]["batch_size"],
                             shuffle=False,
                             num_workers=0)
        
        # Training setup
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config["training"]["learning_rate"], 
            weight_decay=self.config["training"]["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Use weighted cross entropy loss for class balancing
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # Log class distribution and weights
        logger.info("Class distribution in training set:")
        for class_idx, count in sorted(class_distribution.items()):
            class_name = self.idx_to_label.get(class_idx, 'unknown')
            weight = class_weights[class_idx]
            logger.info(f"  {class_name}: {count} samples, weight: {weight:.3f}")

        # Training parameters
        max_epochs = min(self.config["training"]["max_epochs"], 
                         self.config["training"]["min_epochs"] + train_count // 64)
        patience = self.config["training"]["patience"]
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Store metrics
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "class_weights": class_weights
        }

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0
            metrics["train_loss"].append(avg_train_loss)
            metrics["lr"].append(optimizer.param_groups[0]["lr"])
            
            # Validation phase
            if len(val_loader) > 0:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                # Per-class accuracy tracking
                class_correct = Counter()
                class_total = Counter()
                
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
                        
                        # Track per-class accuracy
                        for i in range(y_batch.size(0)):
                            label = y_batch[i].item()
                            class_total[label] += 1
                            if predicted[i] == label:
                                class_correct[label] += 1
                
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
                else:
                    patience_counter += 1
                    if patience_counter >= patience and epoch >= self.config["training"]["min_epochs"]:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

                logger.info(f"Epoch {epoch+1}/{max_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Log per-class validation accuracy periodically
                if (epoch + 1) % 10 == 0:
                    logger.info("Per-class validation accuracy:")
                    for class_idx in range(NUM_CLASSES):
                        if class_idx in class_total and class_total[class_idx] > 0:
                            class_name = self.idx_to_label.get(class_idx, 'unknown')
                            acc = 100. * class_correct[class_idx] / class_total[class_idx]
                            logger.info(f"  {class_name}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
            else:
                # If no validation data, just log training metrics
                logger.info(f"Epoch {epoch+1}/{max_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save metrics
        with open("chess_classifier_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Load best model if it exists
        if os.path.exists("chess_classifier_best.pt"):
            self.model.load_state_dict(torch.load("chess_classifier_best.pt", map_location=self.device))
            logger.info("Loaded best model weights from chess_classifier_best.pt")
        
        torch.save(self.model.state_dict(), "chess_classifier.pt")
        logger.info("Model weights saved to chess_classifier.pt")
    
    def _calculate_class_weights(self, class_distribution: Dict[int, int]) -> List[float]:
        """
        Calculate class weights using inverse frequency weighting.
        
        Args:
            class_distribution: Dictionary mapping class index to count
            
        Returns:
            List of weights for each class
        """
        # Initialize weights for all classes
        weights = [1.0] * NUM_CLASSES
        
        if not class_distribution:
            return weights
            
        # Calculate total samples
        total_samples = sum(class_distribution.values())
        
        # Calculate inverse frequency weights
        for class_idx in range(NUM_CLASSES):
            count = class_distribution.get(class_idx, 0)
            if count > 0:
                # Inverse frequency with smoothing
                weight = total_samples / (NUM_CLASSES * count)
                # Cap weights to prevent extreme values
                weights[class_idx] = min(max(weight, 0.1), 10.0)
            else:
                # Give higher weight to classes with no samples yet
                weights[class_idx] = 2.0
                
        # Normalize weights so their mean is 1.0
        mean_weight = sum(weights) / len(weights)
        if mean_weight > 0:
            weights = [w / mean_weight for w in weights]
            
        return weights