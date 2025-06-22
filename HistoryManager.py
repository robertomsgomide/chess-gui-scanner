from typing import List, Optional, Callable
from ChessBoardModel import ChessBoardModel

#########################################
# HistoryManager.py
#########################################
class HistoryManager:
    """
    Manages undo/redo history for the chess board.
    """
    
    def __init__(self, initial_state: ChessBoardModel, max_history: int = 100):
        """
        Initialize history manager with an initial state.
        
        Args:
            initial_state: Initial board state
            max_history: Maximum number of states to keep in history
        """
        self.history: List[ChessBoardModel] = [initial_state.copy()]
        self.current_index = 0
        self.max_history = max_history
        self.update_callback: Optional[Callable] = None
    
    def set_update_callback(self, callback: Callable):
        """Set callback to call when undo/redo buttons need updating"""
        self.update_callback = callback
    
    def save_state(self, state: ChessBoardModel):
        """
        Save a new state to history.
        
        Args:
            state: Current board state to save
        """
        # If we're not at the end of history, truncate future states
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append(state.copy())
        self.current_index = len(self.history) - 1
        
        # Keep history within limits
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1
        
        self._notify_update()
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current_index < len(self.history) - 1
    
    def undo(self) -> Optional[ChessBoardModel]:
        """
        Undo to previous state.
        
        Returns:
            Previous state if available, None otherwise
        """
        if self.can_undo():
            self.current_index -= 1
            self._notify_update()
            return self.history[self.current_index].copy()
        return None
    
    def redo(self) -> Optional[ChessBoardModel]:
        """
        Redo to next state.
        
        Returns:
            Next state if available, None otherwise
        """
        if self.can_redo():
            self.current_index += 1
            self._notify_update()
            return self.history[self.current_index].copy()
        return None
    
    def get_current_state(self) -> ChessBoardModel:
        """Get current state"""
        return self.history[self.current_index].copy()
    
    def _notify_update(self):
        """Notify that undo/redo button states should be updated"""
        if self.update_callback:
            self.update_callback()
    
    def clear_history(self, new_initial_state: ChessBoardModel):
        """
        Clear history and start fresh with a new initial state.
        
        Args:
            new_initial_state: New initial state
        """
        self.history = [new_initial_state.copy()]
        self.current_index = 0
        self._notify_update() 