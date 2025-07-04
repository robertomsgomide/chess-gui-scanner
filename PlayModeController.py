import chess
from typing import List, Optional, Tuple
from ChessBoardModel import ChessBoardModel
from PgnManager import PgnManager

class PlayModeController:
    """
    Handles all play mode logic including move validation, legal move highlighting,
    and move execution. This separates the game logic from the UI management.
    """
    
    def __init__(self, board_model: ChessBoardModel, pgn_manager: PgnManager, ui_manager):
        """
        Initialize the play mode controller.
        
        Args:
            board_model: The chess board model
            pgn_manager: PGN manager for move recording
            ui_manager: UI manager for accessing board squares
        """
        self.board_model = board_model
        self.pgn_manager = pgn_manager
        self.ui_manager = ui_manager
        
        # Play mode state
        self.selected_square: Optional[int] = None
        self.legal_moves: List[chess.Move] = []
        self.highlighted_squares: List[Tuple[int, int]] = []
        
        # Drag state
        self.play_mode_drag_legal_moves: List[chess.Move] = []
        self.play_mode_drag_highlighted_squares: List[Tuple[int, int]] = []
        
    def handle_square_click(self, row: int, col: int):
        """
        Handle square clicks in Play mode for legal moves.
        This now correctly handles piece selection, deselection, and move execution.
        """
        if self.board_model.is_display_flipped:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row
        
        clicked_square = chess.square(file_idx, rank_idx)
        internal_board = self.board_model.get_internal_board()

        # If a piece is already selected
        if self.selected_square is not None:
            # Check if the click is on a legal destination square
            move_to_execute = None
            for move in self.legal_moves:
                if move.to_square == clicked_square:
                    # Found a legal move to the clicked square.
                    # Handle pawn promotions: if it's a promotion, we prefer the queen.
                    # A more advanced UI could ask the user, but this is a robust default.
                    if move.promotion:
                        if move.promotion == chess.QUEEN:
                            move_to_execute = move
                            break  # Queen promotion is the best choice
                        # Otherwise, store the first promotion option found
                        if not move_to_execute:
                            move_to_execute = move
                    else:
                        # This is a non-promotion move (including en passant)
                        move_to_execute = move
                        break

            if move_to_execute:
                # A legal move was found, so execute it
                self._clear_highlights()
                self._execute_move(move_to_execute)
                self.selected_square = None
                self.legal_moves = []
                return

            # If the click was not on a legal destination, it might be on another piece
            self._clear_highlights()
            self.selected_square = None
            self.legal_moves = []
            
            piece_on_clicked_square = internal_board.piece_at(clicked_square)
            if piece_on_clicked_square and piece_on_clicked_square.color == internal_board.turn:
                # The user clicked another one of their own pieces, so select that one instead
                self._select_piece(clicked_square)
            # Otherwise, the click was on an empty/opponent square, effectively deselecting.

        else:  # No piece is currently selected
            piece_on_clicked_square = internal_board.piece_at(clicked_square)
            if piece_on_clicked_square and piece_on_clicked_square.color == internal_board.turn:
                # Select the clicked piece
                self._select_piece(clicked_square)

    def _select_piece(self, square: int):
        """Helper method to select a piece and highlight its legal moves."""
        self.selected_square = square
        internal_board = self.board_model.get_internal_board()

        # Clear any previous highlights
        self._clear_highlights()

        # Get and highlight all legal moves for the selected piece
        self.legal_moves = [move for move in internal_board.legal_moves if move.from_square == square]

        for move in self.legal_moves:
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)

            if self.board_model.is_display_flipped:
                disp_row, disp_col = to_rank, 7 - to_file
            else:
                disp_row, disp_col = 7 - to_rank, to_file
            
            self.ui_manager.squares[disp_row][disp_col].set_highlight(True)
            self.highlighted_squares.append((disp_row, disp_col))
            
    def start_drag(self, from_row: int, from_col: int) -> bool:
        """
        Start a drag operation in Play mode by highlighting legal moves.
        
        Args:
            from_row: Display row of source square
            from_col: Display column of source square
            
        Returns:
            True if drag can start (piece can move), False otherwise
        """
        # Convert display coords to chess square
        if self.board_model.is_display_flipped:
            file_idx = 7 - from_col
            rank_idx = from_row
        else:
            file_idx = from_col
            rank_idx = 7 - from_row
        
        from_square = chess.square(file_idx, rank_idx)
        internal_board = self.board_model.get_internal_board()
        
        # Check if there's a piece of the current player at this square
        piece = internal_board.piece_at(from_square)
        if not piece or piece.color != internal_board.turn:
            return False
        
        # Clear any existing highlights
        self.clear_drag_highlights()
        
        # Find legal moves from this square
        self.play_mode_drag_legal_moves = [move for move in internal_board.legal_moves 
                                         if move.from_square == from_square]
        
        if not self.play_mode_drag_legal_moves:
            return False
        
        # Highlight legal destination squares in green
        for move in self.play_mode_drag_legal_moves:
            # Convert to display coords
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            
            if self.board_model.is_display_flipped:
                disp_row = to_rank
                disp_col = 7 - to_file
            else:
                disp_row = 7 - to_rank
                disp_col = to_file
            
            self.ui_manager.squares[disp_row][disp_col].set_highlight(True)
            self.play_mode_drag_highlighted_squares.append((disp_row, disp_col))
        
        return True
        
    def is_move_legal(self, from_row: int, from_col: int, to_row: int, to_col: int) -> Optional[chess.Move]:
        """
        Check if a move is legal in Play mode and return the move object.
        
        Args:
            from_row, from_col: Display coordinates of source square
            to_row, to_col: Display coordinates of destination square
            
        Returns:
            chess.Move object if legal, None otherwise
        """
        # Convert destination display coords to chess square
        if self.board_model.is_display_flipped:
            to_file = 7 - to_col
            to_rank = to_row
        else:
            to_file = to_col
            to_rank = 7 - to_row
        
        to_square = chess.square(to_file, to_rank)
        
        # Find matching legal move
        for move in self.play_mode_drag_legal_moves:
            if move.to_square == to_square:
                # For pawn promotion, auto-promote to queen
                internal_board = self.board_model.get_internal_board()
                if (internal_board.piece_at(move.from_square).piece_type == chess.PAWN and
                    chess.square_rank(to_square) in [0, 7]):
                    if move.promotion == chess.QUEEN:
                        return move
                else:
                    return move
        
        return None
        
    def execute_drop(self, move: chess.Move):
        """
        Execute a move after a successful drop operation.
        
        Args:
            move: The chess.Move to execute
        """
        self._execute_move(move)
        self.clear_drag_highlights()
        
    def clear_selection(self):
        """Clear any current piece selection and highlights"""
        self._clear_highlights()
        self.selected_square = None
        self.legal_moves = []
        self.clear_drag_highlights()
        
    def clear_drag_highlights(self):
        """Clear all play mode drag highlights"""
        for sq_coords in self.play_mode_drag_highlighted_squares:
            self.ui_manager.squares[sq_coords[0]][sq_coords[1]].set_highlight(False)
        self.play_mode_drag_highlighted_squares.clear()
        self.play_mode_drag_legal_moves.clear()
        
    def _clear_highlights(self):
        """Clear click-based move highlights"""
        for sq_coords in self.highlighted_squares:
            self.ui_manager.squares[sq_coords[0]][sq_coords[1]].set_highlight(False)
        self.highlighted_squares.clear()
        
    def _execute_move(self, move: chess.Move):
        """
        Execute a chess move and update all relevant states.
        
        Args:
            move: The chess.Move to execute
        """
        internal_board = self.board_model.get_internal_board()
        san = internal_board.san(move)
        internal_board.push(move)
        
        # Record in PGN manager
        self.pgn_manager.add_move(move, san)
        
        # Enable undo button
        self.ui_manager.play_undo_btn.setEnabled(True) 