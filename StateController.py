from enum import Enum
from typing import Callable, Optional

#########################################
# StateController.py
#########################################

class BoardState(Enum):
    """Enumeration of board states"""
    EDIT = "edit"
    PLAY = "play"


class StateController:
    """
    Controls the state transitions between Edit and Play modes.
    Provides a clean interface for state management.
    """
    
    def __init__(self):
        """Initialize in Edit state (default)"""
        self._current_state = BoardState.EDIT
        self._state_change_callback: Optional[Callable[[BoardState], None]] = None
    
    @property
    def current_state(self) -> BoardState:
        """Get the current board state"""
        return self._current_state
    
    @property
    def is_edit_mode(self) -> bool:
        """Check if currently in edit mode"""
        return self._current_state == BoardState.EDIT
    
    @property 
    def is_play_mode(self) -> bool:
        """Check if currently in play mode"""
        return self._current_state == BoardState.PLAY
    
    def set_state_change_callback(self, callback: Callable[[BoardState], None]):
        """
        Set callback to be called when state changes.
        
        Args:
            callback: Function that takes the new BoardState as parameter
        """
        self._state_change_callback = callback
    
    def transition_to_edit(self) -> bool:
        """
        Transition from Play to Edit state.
        
        Returns:
            True if transition was successful
        """
        if self._current_state == BoardState.PLAY:
            self._current_state = BoardState.EDIT
            if self._state_change_callback:
                self._state_change_callback(self._current_state)
            return True
        return False
    
    def transition_to_play(self) -> bool:
        """
        Transition from Edit to Play state.
        
        Returns:
            True if transition was successful
        """
        if self._current_state == BoardState.EDIT:
            self._current_state = BoardState.PLAY
            if self._state_change_callback:
                self._state_change_callback(self._current_state)
            return True
        return False
    
    def get_state_display_name(self) -> str:
        """Get human-readable name for current state"""
        return "Edit Mode" if self.is_edit_mode else "Play Mode" 