import chess
import chess.pgn
from typing import List, Optional, Callable
from io import StringIO

#########################################
# PgnManager.py
#########################################

class PgnManager:
    """
    Manages PGN move recording and history during Play mode.
    Tracks moves, generates PGN snippets, and handles move undo.
    """
    
    def __init__(self):
        """Initialize with empty move list"""
        self.moves: List[chess.Move] = []
        self.move_sans: List[str] = []  # Store SAN notation for display
        self.initial_fen: Optional[str] = None
        self.update_callback: Optional[Callable[[], None]] = None
    
    def set_update_callback(self, callback: Callable[[], None]):
        """Set callback to notify when moves list changes"""
        self.update_callback = callback
    
    def start_new_game(self, initial_fen: str):
        """
        Start recording moves from a given position.
        
        Args:
            initial_fen: Starting position in FEN format
        """
        self.moves.clear()
        self.move_sans.clear()
        self.initial_fen = initial_fen
        self._notify_update()
    
    def add_move(self, move: chess.Move, san: str):
        """
        Add a move to the history.
        
        Args:
            move: The chess.Move object
            san: The move in Standard Algebraic Notation
        """
        self.moves.append(move)
        self.move_sans.append(san)
        self._notify_update()
    
    def undo_last_move(self) -> Optional[chess.Move]:
        """
        Remove and return the last move.
        
        Returns:
            The removed move, or None if no moves to undo
        """
        if self.moves:
            move = self.moves.pop()
            self.move_sans.pop()
            self._notify_update()
            return move
        return None
    
    def get_move_list_text(self) -> str:
        """
        Get formatted move list for display.
        
        Returns:
            Formatted string with move numbers and moves
        """
        if not self.move_sans:
            return "No moves yet"
        
        lines = []
        for i in range(0, len(self.move_sans), 2):
            move_num = (i // 2) + 1
            white_move = self.move_sans[i]
            black_move = self.move_sans[i + 1] if i + 1 < len(self.move_sans) else ""
            
            if black_move:
                lines.append(f"{move_num}. {white_move} {black_move}")
            else:
                lines.append(f"{move_num}. {white_move}")
        
        return "\n".join(lines)
    
    def get_pgn_text(self) -> str:
        """
        Generate PGN text for the current game.
        
        Returns:
            PGN formatted string
        """
        if not self.initial_fen:
            return ""
        
        # Create a game from the initial position
        game = chess.pgn.Game()
        board = chess.Board(self.initial_fen)
        
        # Set up headers
        if self.initial_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.initial_fen
        
        # Add moves
        node = game
        for move in self.moves:
            node = node.add_variation(move)
        
        # Export to string
        output = StringIO()
        exporter = chess.pgn.FileExporter(output)
        game.accept(exporter)
        return output.getvalue()
    
    def has_moves(self) -> bool:
        """Check if any moves have been recorded"""
        return len(self.moves) > 0
    
    def clear(self):
        """Clear all moves and reset"""
        self.moves.clear()
        self.move_sans.clear()
        self.initial_fen = None
        self._notify_update()
    
    def _notify_update(self):
        """Notify that move list has changed"""
        if self.update_callback:
            self.update_callback() 