import chess
from typing import List, Tuple, Optional, Dict
from labels import labels_to_fen
#########################################
# ChessBoardModel.py
#########################################
class ChessBoardModel:
    """
    Single source of truth for chess board state.
    Handles all coordinate transformations and leverages python-chess.Board.
    """
    
    def __init__(self, labels_2d: List[List[str]]):
        """
        Initialize the board model. Orientation is auto-detected from labels.
        
        Args:
            labels_2d: 2D list of piece labels in current display orientation
        """
        # Auto-detect orientation from the labels themselves
        self.is_display_flipped = self._detect_orientation_from_labels(labels_2d)
        
        # Convert labels to standard orientation and create chess.Board
        standard_labels = self._convert_to_standard_orientation(labels_2d)
        fen = labels_to_fen(standard_labels, 'w', 'KQkq', '-')  # Default values
        self._board = chess.Board(fen)
    
    def _detect_orientation_from_labels(self, labels_2d: List[List[str]]) -> bool:
        """Detect if board is flipped by looking for white pieces on top vs bottom."""
        white_top = sum(1 for piece in labels_2d[0] + labels_2d[1] if piece.startswith('w'))
        white_bottom = sum(1 for piece in labels_2d[6] + labels_2d[7] if piece.startswith('w'))
        return white_top > white_bottom  # True if flipped (white pieces mostly on top)
        
    def _convert_to_standard_orientation(self, labels_2d: List[List[str]]) -> List[List[str]]:
        """Convert from display orientation to standard chess orientation (rank 8 first, file a first)"""
        result = [row[:] for row in labels_2d]  # Deep copy
        if self.is_display_flipped:
            result.reverse()
            for row in result:
                row.reverse()
        return result
    
    def _convert_to_display_orientation(self, standard_labels: List[List[str]]) -> List[List[str]]:
        """Convert from standard orientation to current display orientation"""
        result = [row[:] for row in standard_labels]  # Deep copy
        if self.is_display_flipped:
            result.reverse()
            for row in result:
                row.reverse()
        return result
    

    
    def _chess_board_to_labels(self, board: chess.Board) -> List[List[str]]:
        """Convert python-chess Board to standard orientation label matrix"""
        # Map piece symbols to our labels
        piece_map = {
            'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
        }
        
        labels = []
        for rank in range(8):
            row = []
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Convert to chess.Square
                piece = board.piece_at(square)
                if piece:
                    row.append(piece_map[piece.symbol()])
                else:
                    row.append('empty')
            labels.append(row)
        return labels
    
    def get_display_labels(self) -> List[List[str]]:
        """Get board labels in current display orientation"""
        standard_labels = self._chess_board_to_labels(self._board)
        return self._convert_to_display_orientation(standard_labels)
    
    def set_piece_at_display_coords(self, row: int, col: int, piece_label: str):
        """Set piece at display coordinates"""
        # Convert display coordinates to standard coordinates
        if self.is_display_flipped:
            std_row, std_col = 7 - row, 7 - col
        else:
            std_row, std_col = row, col
        
        # Convert to chess square
        square = chess.square(std_col, 7 - std_row)
        
        # Convert label to piece
        if piece_label == 'empty':
            piece = None
        else:
            color = chess.WHITE if piece_label[0] == 'w' else chess.BLACK
            piece_type_map = {'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP, 
                            'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING}
            piece_type = piece_type_map[piece_label[1]]
            piece = chess.Piece(piece_type, color)
        
        # Set piece on board
        self._board.set_piece_at(square, piece)
    
    def get_piece_at_display_coords(self, row: int, col: int) -> str:
        """Get piece label at display coordinates"""
        labels = self.get_display_labels()
        return labels[row][col]
    
    def flip_display_orientation(self):
        """Flip the display orientation (toggle perspective)"""
        self.is_display_flipped = not self.is_display_flipped
    
    def clear_board(self):
        """Clear all pieces from the board"""
        self._board.clear()
    
    def reset_to_starting_position(self):
        """Reset to standard chess starting position"""
        self._board = chess.Board()
    
    def get_fen(self) -> str:
        """Get FEN string for current position"""
        return self._board.fen()
    
    def set_from_fen(self, fen: str):
        """Set board state from FEN string"""
        try:
            self._board = chess.Board(fen)
        except Exception as e:
            raise ValueError(f"Invalid FEN: {e}")
    
    def get_side_to_move(self) -> str:
        """Get side to move ('w' or 'b')"""
        return 'w' if self._board.turn else 'b'
    
    def set_side_to_move(self, side: str):
        """Set side to move"""
        self._board.turn = (side == 'w')
    
    def get_castling_rights(self) -> str:
        """Get castling rights string"""
        rights = ""
        if self._board.has_kingside_castling_rights(chess.WHITE):
            rights += "K"
        if self._board.has_queenside_castling_rights(chess.WHITE):
            rights += "Q"
        if self._board.has_kingside_castling_rights(chess.BLACK):
            rights += "k"
        if self._board.has_queenside_castling_rights(chess.BLACK):
            rights += "q"
        return rights or "-"
    
    def set_castling_rights(self, rights: str):
        """Set castling rights"""
        # Use python-chess's proper method to set castling rights
        if rights == "-":
            self._board.castling_rights = chess.BB_EMPTY
        else:
            # Build castling rights step by step
            castling_rights = chess.BB_EMPTY
            if 'K' in rights:
                castling_rights |= chess.BB_H1
            if 'Q' in rights:
                castling_rights |= chess.BB_A1
            if 'k' in rights:
                castling_rights |= chess.BB_H8
            if 'q' in rights:
                castling_rights |= chess.BB_A8
            self._board.castling_rights = castling_rights
    
    def can_castle(self, side: str, kingside: bool) -> bool:
        """Check if castling is possible"""
        color = chess.WHITE if side == 'w' else chess.BLACK
        if kingside:
            return self._board.has_kingside_castling_rights(color)
        else:
            return self._board.has_queenside_castling_rights(color)
    
    def is_in_check(self) -> bool:
        """Check if current side to move is in check"""
        return self._board.is_check()
    
    def get_en_passant_targets(self) -> Dict[Tuple[int, int], str]:
        """Get possible en passant targets in display coordinates"""
        targets = {}
        
        # Get current side to move
        side = self.get_side_to_move()
        
        # Get board in standard orientation
        standard_labels = self._chess_board_to_labels(self._board)
        
        def alg(r, c): 
            return "abcdefgh"[c] + str(8 - r)

        if side == 'w':  # White to move
            enpassant_rank = 2  # Target rank (6th rank)
            pawn_rank = 3      # Where the black pawn is (5th rank)
            
            for col in range(8):
                # Check if there's a black pawn on the 5th rank
                if standard_labels[pawn_rank][col] == "bp":
                    
                    # Check if there are white pawns on either side that could capture
                    for dcol in [-1, 1]:
                        if 0 <= col + dcol < 8 and standard_labels[pawn_rank][col + dcol] == "wp":
                            # Found potential en passant - now validate with python-chess
                            pawn_square = alg(pawn_rank, col + dcol)
                            target_square = alg(enpassant_rank, col)
                            ep_move = chess.Move.from_uci(pawn_square+target_square)
                            # Create temporary FEN with this en passant target
                            current_fen = self._board.fen()
                            fen_parts = current_fen.split()
                            fen_parts[3] = target_square  # Set en passant field
                            test_fen = " ".join(fen_parts)
                            
                            # Test if python-chess would accept the en passant move
                            
                            test_board = chess.Board(test_fen)

                            # If the FEN is invalid, clear targets and stop
                            if not test_board.is_legal(ep_move) or not test_board.is_valid():
                                break
                            
                            # If we get here, the en passant is valid
                            trg_r, trg_c = enpassant_rank, col
                            if self.is_display_flipped:
                                disp_r, disp_c = 7 - trg_r, 7 - trg_c
                            else:
                                disp_r, disp_c = trg_r, trg_c
                            targets[(disp_r, disp_c)] = target_square
                            break  # One target per pawn
                                
        else:  # Black to move (same logic)
            enpassant_rank = 5  # Target rank (3rd rank)
            pawn_rank = 4      # Where the white pawn is (4th rank) 
            
            for col in range(8):
                if standard_labels[pawn_rank][col] == "wp":
                                       
                    for dcol in [-1, 1]:
                        if 0 <= col + dcol < 8 and standard_labels[pawn_rank][col + dcol] == "bp":
                            pawn_square = alg(pawn_rank, col + dcol)
                            target_square = alg(enpassant_rank, col)
                            ep_move = chess.Move.from_uci(pawn_square+target_square)
                            current_fen = self._board.fen()
                            fen_parts = current_fen.split()
                            fen_parts[3] = target_square
                            test_fen = " ".join(fen_parts)
                            
                            test_board = chess.Board(test_fen)
                            
                            # If the FEN is invalid, clear targets and stop
                            if not test_board.is_legal(ep_move) or not test_board.is_valid():
                                break
                            
                            trg_r, trg_c = enpassant_rank, col
                            if self.is_display_flipped:
                                disp_r, disp_c = 7 - trg_r, 7 - trg_c
                            else:
                                disp_r, disp_c = trg_r, trg_c
                            targets[(disp_r, disp_c)] = target_square
                            break

        return targets
    

    
    def set_en_passant_square(self, square_name: Optional[str]):
        """Set en passant target square"""
        if square_name and square_name != '-':
            self._board.ep_square = chess.parse_square(square_name)
        else:
            self._board.ep_square = None
    
    def copy(self) -> 'ChessBoardModel':
        """Create a deep copy of this board model"""
        # Create a new model using the current display labels
        current_labels = self.get_display_labels()
        new_model = ChessBoardModel(current_labels)
        
        # Copy the internal chess board state directly to preserve all details
        new_model._board = self._board.copy()
        new_model.is_display_flipped = self.is_display_flipped
        return new_model
    
    def validate_position(self) -> Tuple[bool, str]:
        """
        Validate current position for legality.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if board has exactly one king per side
            white_kings = len(self._board.pieces(chess.KING, chess.WHITE))
            black_kings = len(self._board.pieces(chess.KING, chess.BLACK))
            
            if white_kings != 1:
                return False, f"White must have exactly 1 king (found {white_kings})"
            if black_kings != 1:
                return False, f"Black must have exactly 1 king (found {black_kings})"
            
            # Check if the opponent's king is in check
            # (it's illegal to leave opponent in check)
            self._board.turn = not self._board.turn  # Switch to opponent
            if self._board.is_check():
                self._board.turn = not self._board.turn  # Switch back
                opponent = "Black" if self._board.turn else "White"
                return False, f"{opponent}'s king is in check (illegal position)"
            self._board.turn = not self._board.turn  # Switch back
            
            # Check for pawns on first/last rank
            for square in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
                piece = self._board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    rank = "1st" if square in chess.SquareSet(chess.BB_RANK_1) else "8th"
                    return False, f"Pawns cannot be on {rank} rank"
            
            # Use python-chess built-in validation
            if not self._board.is_valid():
                return False, "Position is invalid (general board constraints violated)"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_internal_board(self) -> chess.Board:
        """
        Get reference to internal python-chess Board.
        Use with caution - prefer public methods when possible.
        
        Returns:
            The internal chess.Board instance
        """
        return self._board
    
 