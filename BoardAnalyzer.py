import numpy as np
import os
import pickle
from typing import List

class BoardAnalyzer:
    """
    Analyzes chess positions to determine:
    1. Board orientation (POV) - is it viewed from white or black side
    2. Side to move (white's or black's turn)
    
    Uses various heuristics and machine learning to make these determinations.
    """
    def __init__(self):
        # Configuration
        self.orientation_data = []  # [(board_matrix, is_flipped_orientation), ...]
        self.side_to_move_data = []  # [(board_matrix, side_to_move), ...]
        
        # Sample weights for various heuristics
        self.heuristics_weights = {
            "piece_distribution": 0.6,
            "board_state": 0.4
        }
    
    def save_training_data(self, labels_2d: List[List[str]], is_flipped: bool, side_to_move: str):
        """
        Save orientation and side-to-move training data when user confirms it.
        
        Args:
            labels_2d: 2D grid of piece labels
            is_flipped: True if board is viewed from black's perspective
            side_to_move: 'w' or 'b' indicating which side to move
        """
        # Convert board to a flat feature vector for training
        board_features = self._board_to_features(labels_2d)
        
        # Store data for training
        self.orientation_data.append((board_features, is_flipped))
        self.side_to_move_data.append((board_features, side_to_move))
        
        # Save to disk after each training example
        self.save_to_disk()
    
    def predict_orientation(self, labels_2d: List[List[str]]) -> bool:
        """
        Predict if board is viewed from black's perspective (flipped)
        
        Args:
            labels_2d: 2D grid of piece labels
            
        Returns:
            bool: True if board is likely viewed from black's perspective
        """
        # If we have no training data, use heuristics only
        if len(self.orientation_data) < 5:
            return self._heuristic_orientation(labels_2d)
        
        # Use a combination of ML and heuristics
        ml_prediction = self._ml_predict_orientation(labels_2d)
        heuristic_prediction = self._heuristic_orientation(labels_2d)
        
        # Weight the predictions based on training data size
        ml_weight = min(0.8, len(self.orientation_data) / 50)  # Max 0.8 weight for ML
        return ml_prediction if np.random.random() < ml_weight else heuristic_prediction
    
    def predict_side_to_move(self, labels_2d: List[List[str]]) -> str:
        """
        Predict which side is to move
        
        Args:
            labels_2d: 2D grid of piece labels
            
        Returns:
            str: 'w' for white or 'b' for black
        """
        # If we have no training data, use heuristics only
        if len(self.side_to_move_data) < 5:
            return self._heuristic_side_to_move(labels_2d)
        
        # Use a combination of ML and heuristics
        ml_prediction = self._ml_predict_side_to_move(labels_2d)
        heuristic_prediction = self._heuristic_side_to_move(labels_2d)
        
        # Weight the predictions based on training data size
        ml_weight = min(0.8, len(self.side_to_move_data) / 50)  # Max 0.8 weight for ML
        return ml_prediction if np.random.random() < ml_weight else heuristic_prediction
    
    def save_to_disk(self):
        """Save the analyzer state to a file"""
        analyzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_analyzer.pkl")
        try:
            with open(analyzer_path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Error saving analyzer: {e}")
    
    @classmethod
    def load_from_disk(cls):
        """Load analyzer state from a file"""
        analyzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_analyzer.pkl")
        if os.path.exists(analyzer_path):
            try:
                with open(analyzer_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading analyzer: {e}")
        return cls()  # Return a new instance if loading fails
    
    def _board_to_features(self, labels_2d: List[List[str]]) -> List[float]:
        """Convert board labels to a feature vector for ML."""
        # Piece encoding: 0=empty, 1-6=white pieces, 7-12=black pieces
        piece_map = {
            'empty': 0, 
            'wp': 1, 'wn': 2, 'wb': 3, 'wr': 4, 'wq': 5, 'wk': 6, 
            'bp': 7, 'bn': 8, 'bb': 9, 'br': 10, 'bq': 11, 'bk': 12
        }
        
        # Flatten the board into a feature vector
        features = []
        for row in labels_2d:
            for cell in row:
                # One-hot encode each piece
                piece_features = [0] * 13
                piece_features[piece_map.get(cell, 0)] = 1
                features.extend(piece_features)
                
        return features
    
    def _ml_predict_orientation(self, labels_2d: List[List[str]]) -> bool:
        """Use machine learning to predict board orientation."""
        # Convert board to features
        features = self._board_to_features(labels_2d)
        
        # Basic k-nearest neighbors approach
        k = min(5, len(self.orientation_data))
        similarities = []
        
        for board_features, is_flipped in self.orientation_data:
            # Calculate cosine similarity
            similarity = self._cosine_similarity(features, board_features)
            similarities.append((similarity, is_flipped))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        # Take top k
        top_k = similarities[:k]
        
        # Count votes
        flipped_count = sum(1 for _, is_flipped in top_k if is_flipped)
        
        # Return majority vote
        return flipped_count > k/2
    
    def _ml_predict_side_to_move(self, labels_2d: List[List[str]]) -> str:
        """Use machine learning to predict side to move."""
        # Convert board to features
        features = self._board_to_features(labels_2d)
        
        # Basic k-nearest neighbors approach
        k = min(5, len(self.side_to_move_data))
        similarities = []
        
        for board_features, side in self.side_to_move_data:
            # Calculate cosine similarity
            similarity = self._cosine_similarity(features, board_features)
            similarities.append((similarity, side))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        # Take top k
        top_k = similarities[:k]
        
        # Count votes
        white_count = sum(1 for _, side in top_k if side == 'w')
        
        # Return majority vote
        return 'w' if white_count > k/2 else 'b'
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a*b for a, b in zip(v1, v2))
        norm_v1 = sum(a*a for a in v1) ** 0.5
        norm_v2 = sum(b*b for b in v2) ** 0.5
        return dot_product / (norm_v1 * norm_v2) if norm_v1 * norm_v2 > 0 else 0
    
    def _heuristic_orientation(self, labels_2d: List[List[str]]) -> bool:
        """Use heuristics to determine board orientation."""
        # Initial position has most pieces in the first/last two ranks
        w_pieces_top = sum(1 for r in range(2) for c in range(8) 
                           if labels_2d[r][c].startswith('w'))
        w_pieces_bottom = sum(1 for r in range(6, 8) for c in range(8) 
                             if labels_2d[r][c].startswith('w'))
        
        b_pieces_top = sum(1 for r in range(2) for c in range(8) 
                           if labels_2d[r][c].startswith('b'))
        b_pieces_bottom = sum(1 for r in range(6, 8) for c in range(8) 
                             if labels_2d[r][c].startswith('b'))
        
        # If white pieces are mostly at the top, it's likely flipped
        white_top_score = w_pieces_top > w_pieces_bottom
        black_bottom_score = b_pieces_bottom > b_pieces_top
        
        # Default to non-flipped if it's ambiguous
        if white_top_score and black_bottom_score:
            return True  # Flipped
        elif not white_top_score and not black_bottom_score:
            return False  # Not flipped
        
        # More specific check for kings and queens position
        white_king_row = None
        black_king_row = None
        
        for r in range(8):
            for c in range(8):
                if labels_2d[r][c] == 'wk':
                    white_king_row = r
                elif labels_2d[r][c] == 'bk':
                    black_king_row = r
        
        # If we found both kings, check their relative positions
        if white_king_row is not None and black_king_row is not None:
            return white_king_row < black_king_row  # Flipped if white king is higher
        
        # Default to not flipped
        return False
    
    def _heuristic_side_to_move(self, labels_2d: List[List[str]]) -> str:
        """Use heuristics to determine side to move."""
        # Count pieces
        white_pieces = sum(1 for r in range(8) for c in range(8) 
                           if labels_2d[r][c].startswith('w'))
        black_pieces = sum(1 for r in range(8) for c in range(8) 
                           if labels_2d[r][c].startswith('b'))
        
        # More pieces usually means it's your turn (you haven't captured yet)
        if white_pieces > black_pieces + 1:
            return 'b'  # Black's turn (white has more pieces)
        elif black_pieces > white_pieces + 1:
            return 'w'  # White's turn (black has more pieces)
        
        # Try checking pawn structure
        white_pawn_ranks = [r for r in range(8) for c in range(8) 
                           if labels_2d[r][c] == 'wp']
        black_pawn_ranks = [r for r in range(8) for c in range(8) 
                           if labels_2d[r][c] == 'bp']
        
        # If we have a pawn that's advanced further than the opponent's most advanced pawn
        if white_pawn_ranks and black_pawn_ranks:
            min_white_rank = min(white_pawn_ranks)
            max_black_rank = max(black_pawn_ranks)
            
            if min_white_rank < 2:  # White pawn very advanced
                return 'b'  # Likely black to move after white advanced
            if max_black_rank > 5:  # Black pawn very advanced
                return 'w'  # Likely white to move after black advanced
        
        # Default to white's turn
        return 'w' 