import numpy as np
import sqlite3
import json
from typing import List, Tuple, Dict, Any, Optional

#########################################
# Board Analyzer
#########################################
class BoardAnalyzer:
    """
    Analyzes chess positions to determine:
    1. Board orientation (POV) - is it viewed from white or black side
    2. Side to move (white's or black's turn)
    
    Uses various heuristics and machine learning to make these determinations.
    Now uses SQLite for secure, queryable storage instead of pickle files.
    """
    def __init__(self, db_path: str = "chess_training_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema for board analyzer data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create orientation training table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orientation_training (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    board_features TEXT NOT NULL,  -- JSON array of features
                    is_flipped BOOLEAN NOT NULL,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Create index for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orientation_timestamp ON orientation_training(timestamp)')
            
            conn.commit()
    
    def save_training_data(self, labels_2d: List[List[str]], is_flipped: bool):
        """
        Save orientation training data when user confirms it.
        
        Args:
            labels_2d: 2D grid of piece labels
            is_flipped: True if board is viewed from black's perspective
        """
        # Convert board to a flat feature vector for training
        board_features = self._board_to_features(labels_2d)
        
        # Store data in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO orientation_training (board_features, is_flipped)
                VALUES (?, ?)
            ''', (json.dumps(board_features), is_flipped))
            conn.commit()
    
    def get_training_data_count(self) -> int:
        """Get the count of orientation training examples."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM orientation_training')
            return cursor.fetchone()[0]
    
    def _load_orientation_data(self, limit: Optional[int] = None) -> List[Tuple[List[float], bool]]:
        """
        Load orientation training data from database.
        
        Args:
            limit: Maximum number of recent examples to load (None for all)
            
        Returns:
            List of (board_features, is_flipped) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if limit:
                cursor.execute('''
                    SELECT board_features, is_flipped 
                    FROM orientation_training 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            else:
                cursor.execute('''
                    SELECT board_features, is_flipped 
                    FROM orientation_training 
                    ORDER BY timestamp DESC
                ''')
            
            results = []
            for row in cursor.fetchall():
                features = json.loads(row[0])
                is_flipped = bool(row[1])
                results.append((features, is_flipped))
            
            return results
    
    def predict_orientation(self, labels_2d: List[List[str]]) -> bool:
        """
        Predict if board is viewed from black's perspective (flipped)
        
        Args:
            labels_2d: 2D grid of piece labels
            
        Returns:
            bool: True if board is likely viewed from black's perspective
        """
        # Load training data (limit to recent 100 examples for performance)
        orientation_data = self._load_orientation_data(limit=100)
        
        return self._predict_with_ml_and_heuristics(
            labels_2d, 
            orientation_data, 
            self._heuristic_orientation
        )
    
    def predict_side_to_move(self, labels_2d: List[List[str]]) -> str:
        """
        Determine which side is to move based on:
        1. Current board orientation (point of view)
        2. Which king – if any – is in check.

        The logic is:
            • If exactly one king is in check, that colour is to move (because the side
              that is in check must respond).
            • If no king is in check, fall back to a sensible default that matches the
              player's point of view: when the board is shown from White's side we
              assume White to move, otherwise Black to move.

        Args:
            labels_2d: 8×8 grid of piece labels in the *current display orientation*.

        Returns:
            'w' if White is to move, 'b' if Black is to move.
        """
        return self._heuristic_side_to_move(labels_2d)
    
    def clear_orientation_data(self):
        """Clear all orientation training data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM orientation_training')
            conn.commit()
    
    def remove_recent_orientation_data(self, count: int = 1) -> int:
        """
        Remove the most recent orientation training examples.
        
        Args:
            count: Number of recent examples to remove
            
        Returns:
            Number of examples actually removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get IDs of most recent entries
            cursor.execute('''
                SELECT id FROM orientation_training 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (count,))
            
            ids_to_remove = [row[0] for row in cursor.fetchall()]
            
            if ids_to_remove:
                placeholders = ','.join('?' * len(ids_to_remove))
                cursor.execute(f'DELETE FROM orientation_training WHERE id IN ({placeholders})', ids_to_remove)
                conn.commit()
                return cursor.rowcount
            
            return 0
    
    def get_orientation_stats(self) -> Dict[str, Any]:
        """Get statistics about orientation training data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute('SELECT COUNT(*) FROM orientation_training')
            total = cursor.fetchone()[0]
            
            # Flipped vs not flipped
            cursor.execute('SELECT is_flipped, COUNT(*) FROM orientation_training GROUP BY is_flipped')
            orientation_counts = dict(cursor.fetchall())
            
            # Recent entries
            cursor.execute('''
                SELECT timestamp, is_flipped 
                FROM orientation_training 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''')
            recent = cursor.fetchall()
            
            return {
                'total': total,
                'flipped_count': orientation_counts.get(1, 0),
                'normal_count': orientation_counts.get(0, 0),
                'recent_entries': recent
            }
    

    
    def _predict_with_ml_and_heuristics(self, labels_2d: List[List[str]], training_data: List, heuristic_func) -> Any:
        """
        Generic prediction method that combines ML and heuristics
        
        Args:
            labels_2d: 2D grid of piece labels
            training_data: List of training examples
            heuristic_func: Function to call for heuristic prediction
            
        Returns:
            Prediction result (type depends on heuristic_func)
        """
        # If we have insufficient training data, use heuristics only
        if len(training_data) < 5:
            return heuristic_func(labels_2d)
        
        # Use a combination of ML and heuristics
        ml_prediction = self._ml_predict(labels_2d, training_data)
        heuristic_prediction = heuristic_func(labels_2d)
        
        # Weight the predictions based on training data size
        ml_weight = min(0.8, len(training_data) / 50)  # Max 0.8 weight for ML
        return ml_prediction if np.random.random() < ml_weight else heuristic_prediction
    
    def _ml_predict(self, labels_2d: List[List[str]], training_data: List) -> Any:
        """
        Generic ML prediction using k-nearest neighbors
        
        Args:
            labels_2d: 2D grid of piece labels
            training_data: List of (features, label) tuples
            
        Returns:
            Prediction based on majority vote of k nearest neighbors
        """
        # Convert board to features
        features = self._board_to_features(labels_2d)
        
        # Basic k-nearest neighbors approach
        k = min(5, len(training_data))
        similarities = []
        
        for board_features, label in training_data:
            # Calculate cosine similarity
            similarity = self._cosine_similarity(features, board_features)
            similarities.append((similarity, label))
        
        # Sort by similarity (highest first) and take top k
        similarities.sort(reverse=True)
        top_k = similarities[:k]
        
        # Count votes for boolean prediction (orientation)
        true_count = sum(1 for _, label in top_k if label)
        return true_count > k/2
    
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
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a*b for a, b in zip(v1, v2))
        norm_v1 = sum(a*a for a in v1) ** 0.5
        norm_v2 = sum(b*b for b in v2) ** 0.5
        return dot_product / (norm_v1 * norm_v2) if norm_v1 * norm_v2 > 0 else 0
    
    def _analyze_board(self, labels_2d: List[List[str]]) -> Dict[str, Any]:
        """
        Analyze the board once and extract all needed information
        
        Returns:
            Dictionary containing board analysis results
        """
        analysis = {
            'white_pieces_top': 0,
            'white_pieces_bottom': 0,
            'black_pieces_top': 0,
            'black_pieces_bottom': 0,
            'white_pieces_total': 0,
            'black_pieces_total': 0,
            'white_king_row': None,
            'black_king_row': None,
            'white_pawn_ranks': [],
            'black_pawn_ranks': []
        }
        
        for r in range(8):
            for c in range(8):
                piece = labels_2d[r][c]
                
                if piece.startswith('w'):
                    analysis['white_pieces_total'] += 1
                    if r < 2:
                        analysis['white_pieces_top'] += 1
                    elif r >= 6:
                        analysis['white_pieces_bottom'] += 1
                    
                    if piece == 'wk':
                        analysis['white_king_row'] = r
                    elif piece == 'wp':
                        analysis['white_pawn_ranks'].append(r)
                
                elif piece.startswith('b'):
                    analysis['black_pieces_total'] += 1
                    if r < 2:
                        analysis['black_pieces_top'] += 1
                    elif r >= 6:
                        analysis['black_pieces_bottom'] += 1
                    
                    if piece == 'bk':
                        analysis['black_king_row'] = r
                    elif piece == 'bp':
                        analysis['black_pawn_ranks'].append(r)
        
        return analysis
    
    def _heuristic_orientation(self, labels_2d: List[List[str]]) -> bool:
        """Use heuristics to determine board orientation."""
        analysis = self._analyze_board(labels_2d)
        
        # If white pieces are mostly at the top, it's likely flipped
        white_top_score = analysis['white_pieces_top'] > analysis['white_pieces_bottom']
        black_bottom_score = analysis['black_pieces_bottom'] > analysis['black_pieces_top']
        
        # Check piece distribution first
        if white_top_score and black_bottom_score:
            return True  # Flipped
        elif not white_top_score and not black_bottom_score:
            return False  # Not flipped
        
        # Use king positions as tiebreaker - check if they're in expected relative positions
        if analysis['white_king_row'] is not None and analysis['black_king_row'] is not None:
            # In normal orientation: white king should be on row 7, black on row 0
            # In flipped orientation: white king should be on row 0, black on row 7
            # If white king is closer to bottom (higher row) than black king, it's normal orientation
            king_row_diff = analysis['white_king_row'] - analysis['black_king_row']
            return king_row_diff < 0  # Flipped if white king is on lower row number than black king
        
        # Default to not flipped
        return False
    
    def _heuristic_side_to_move(self, labels_2d: List[List[str]]) -> str:
        """Determine side to move using only orientation and check status."""
        # We'll need the helper utilities here.
        import chess  # local import to avoid mandatory dependency if analyser used headless
        from labels import labels_to_fen

        # 1) Work out board orientation (True → flipped, i.e. Black at the bottom)
        is_flipped = self.predict_orientation(labels_2d)

        # 2) Convert the board into *standard* orientation (rank-8 first, file-a first)
        std_board = [row[:] for row in labels_2d]
        if is_flipped:
            std_board.reverse()
            for row in std_board:
                row.reverse()

        # 3) Build two FENs – one assuming White to move, one Black to move – and
        #    check whose king is currently attacked.
        fen_white = labels_to_fen(std_board, side_to_move='w', castling='-', ep_target='-')
        fen_black = labels_to_fen(std_board, side_to_move='b', castling='-', ep_target='-')

        board_w = chess.Board(fen_white)
        board_b = chess.Board(fen_black)

        white_in_check = board_w.is_check()
        black_in_check = board_b.is_check()

        # If exactly one side is in check, that side is to move.
        if white_in_check and not black_in_check:
            return 'w'
        if black_in_check and not white_in_check:
            return 'b'

        # Otherwise, default to side matching the player's POV.
        return 'b' if is_flipped else 'w' 