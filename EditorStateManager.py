from BoardUIManager import BoardUIManager
from StateController import BoardState

class EditorStateManager:
    """
    Manages UI state transitions between Edit and Play modes.
    This class is responsible for showing/hiding/enabling/disabling widgets
    based on the current board state.
    """
    
    def __init__(self, ui_manager: BoardUIManager, history_manager, pgn_manager):
        """
        Initialize the state manager.
        
        Args:
            ui_manager: BoardUIManager instance containing all UI widgets
            history_manager: HistoryManager instance for undo/redo state
            pgn_manager: PgnManager instance for move history
        """
        self.ui_manager = ui_manager
        self.history_manager = history_manager
        self.pgn_manager = pgn_manager
        
    def handle_state_change(self, new_state: BoardState):
        """
        Handle state change callback from StateController.
        
        Args:
            new_state: The new board state
        """
        if new_state == BoardState.EDIT:
            self.apply_edit_state_ui()
        else:
            self.apply_play_state_ui()
            
    def apply_edit_state_ui(self):
        """Apply UI changes for Edit state - show editing controls, hide play controls"""
        ui = self.ui_manager
        
        # Update button text
        ui.state_toggle_btn.setText("Finish Edit")
        ui.state_toggle_btn.setIcon(ui.icons['done'])
        
        # Show editing controls
        ui.clear_btn.show()
        ui.switch_btn.show()
        ui.reset_btn.show()
        ui.redo_btn.show()
        ui.redo_btn.setEnabled(self.history_manager.can_redo())
        ui.undo_btn.show()
        ui.undo_btn.setEnabled(self.history_manager.can_undo())
        
        # Show editing controls
        ui.flip_btn.show()
        ui.paste_btn.show()
        
        # Hide Copy FEN (only available in Play mode)
        ui.copy_btn.hide()
        
        # Set Edit mode button text
        ui.undo_btn.setText("Undo")
        ui.undo_btn.setToolTip("Undo last board edit")
        
        # Hide Play mode undo button
        ui.play_undo_btn.hide()
        
        # Show castling and en passant controls
        ui.white_rb.show()
        ui.black_rb.show()
        ui.w_k_cb.show()
        ui.w_q_cb.show()
        ui.b_k_cb.show()
        ui.b_q_cb.show()
        ui.ep_cb.show()
        
        # Hide Learn and Analysis
        ui.learn_btn.hide()
        # Hide the analysis container (which contains analysis button and dropdown)
        analysis_container = ui.analysis_btn.parent()
        if analysis_container:
            analysis_container.hide()
        ui.reset_analysis_btn.hide()
        
        # Hide Copy PGN button (will be shown in Play mode)
        ui.copy_pgn_btn.hide()
        
        # Show palette
        if ui.palette_container:
            ui.palette_container.show()
        
        # Hide move list
        ui.move_list_container.hide()
        
        # Enable piece dragging on board
        for row in ui.squares:
            for square in row:
                if square:
                    square.setAcceptDrops(True)
                    
    def apply_play_state_ui(self):
        """Apply UI changes for Play state - hide editing controls, show play controls"""
        ui = self.ui_manager
        
        # Update button text
        ui.state_toggle_btn.setText("Edit Board")
        ui.state_toggle_btn.setIcon(ui.icons['empty'])
        
        # Hide most editing controls
        ui.clear_btn.hide()
        ui.switch_btn.hide()
        ui.reset_btn.hide()
        ui.redo_btn.hide()
        
        # These controls remain visible in Play mode
        ui.flip_btn.show()
        ui.copy_btn.show()
        ui.paste_btn.hide()  # Hide Paste FEN in Play mode
        ui.copy_pgn_btn.show()  # Show Copy PGN in main button bar
        
        # Hide main undo button from button bar, use the one in move list
        ui.undo_btn.hide()
        
        # Show and enable Play mode undo button in move list
        ui.play_undo_btn.show()
        ui.play_undo_btn.setEnabled(self.pgn_manager.has_moves())
        
        # Hide castling and en passant controls
        ui.white_rb.hide()
        ui.black_rb.hide()
        ui.w_k_cb.hide()
        ui.w_q_cb.hide()
        ui.b_k_cb.hide()
        ui.b_q_cb.hide()
        ui.ep_cb.hide()
        
        # Show Learn and Analysis
        ui.learn_btn.show()
        # Show the analysis container (which contains analysis button and dropdown)
        analysis_container = ui.analysis_btn.parent()
        if analysis_container:
            analysis_container.show()
        ui.reset_analysis_btn.show()
        
        # Hide palette
        if ui.palette_container:
            ui.palette_container.hide()
        
        # Show move list
        ui.move_list_container.show()
        
        # Enable piece dragging for Play mode (legal moves only)
        for row in ui.squares:
            for square in row:
                if square:
                    square.setAcceptDrops(True) 