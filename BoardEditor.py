import os
import sys
from BoardSquareWidget import BoardSquareWidget
from BoardUIManager import BoardUIManager
from EditorStateManager import EditorStateManager
from PlayModeController import PlayModeController
from labels import (PIECE_LABELS, get_piece_pixmap)
from ChessBoardModel import ChessBoardModel
from HistoryManager import HistoryManager
from AnalysisManager import AnalysisManager
from StateController import StateController, BoardState
from PgnManager import PgnManager
from PyQt5.QtCore import Qt, QEvent, QSettings, QTimer
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMessageBox, QWhatsThis, QAction, QMenu, 
    QFileDialog
)


#########################################
# BoardEditor
#########################################

class BoardEditor(QDialog):
    """
    The main dialog that orchestrates the chess board editor.
    Now refactored to delegate responsibilities to specialized managers and controllers.
    """
    def __init__(self, labels_2d, predicted_side_to_move='w'):
        super().__init__()
        self.setWindowTitle("Chess Viewer")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint | Qt.WindowContextHelpButtonHint)
        
        # Initialize the board model (single source of truth)
        self.board_model = ChessBoardModel(labels_2d)
        self.board_model.set_side_to_move(predicted_side_to_move)
        
        # Initialize managers
        self.history_manager = HistoryManager(self.board_model)
        self.analysis_manager = AnalysisManager()
        self.state_controller = StateController()
        self.pgn_manager = PgnManager()
        
        # Initialize UI manager
        self.ui_manager = BoardUIManager(self, predicted_side_to_move)
        self.setLayout(self.ui_manager.main_layout)
        
        # Apply board model orientation to UI
        if self.board_model.is_display_flipped:
            self.ui_manager.file_labels.reverse()
            self.ui_manager.rank_labels.reverse()
            self.ui_manager.update_coordinate_labels(
                self.ui_manager.file_labels, 
                self.ui_manager.rank_labels
            )
        
        # Create board squares from model
        self._create_board_squares()
        
        # Create palette squares
        self._create_palette_squares()
        
        # Settings storage
        self.settings = QSettings("ChessAIScanner", "Settings")
        
        # Initialize state manager
        self.editor_state_manager = EditorStateManager(
            self.ui_manager, 
            self.history_manager,
            self.pgn_manager
        )
        
        # Initialize play mode controller
        self.play_mode_controller = PlayModeController(
            self.board_model,
            self.pgn_manager,
            self.ui_manager
        )
        
        # Set up callbacks
        self.history_manager.set_update_callback(self.update_undo_redo_buttons)
        self.analysis_manager.set_update_display_callback(self.update_analysis_display)
        self.analysis_manager.set_update_board_callback(self.update_board_from_model)
        self.state_controller.set_state_change_callback(self.on_state_changed)
        self.pgn_manager.set_update_callback(self.update_move_list_display)
        
        # Initialize piece memory feature
        self.remembered_piece = None
        
        # Timer to handle double-click vs single-click on palette pieces
        self.palette_click_timer = QTimer()
        self.palette_click_timer.setSingleShot(True)
        self.palette_click_timer.timeout.connect(self._process_delayed_palette_click)
        self.pending_palette_click = None
        self.double_click_in_progress = False
        
        # En passant state
        self.ep_possible = {}
        self.ep_selected = None
        self.ep_highlight_on = False
        
        # Help system
        self._setup_help_system()
        
        # Connect UI signals
        self._connect_ui_signals()
        
        # Window icon
        base_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
        self.setWindowIcon(QIcon(os.path.join(base_icon, "Chess_icon.png")))
        
        # Initialize UI state
        self.refresh_ui_from_model()
        self.save_state()
        self.editor_state_manager.apply_edit_state_ui()
        
        # Update analysis button text if engine is loaded
        engine_name = self.analysis_manager.get_selected_engine_name()
        if engine_name:
            self.ui_manager.analysis_btn.setText(f"Analysis ({engine_name})")
    
    def _create_board_squares(self):
        """Create board square widgets from the model"""
        display_labels = self.board_model.get_display_labels()
        for r in range(8):
            for c in range(8):
                square = BoardSquareWidget(r, c, display_labels[r][c], parent=self)
                self.ui_manager.set_board_square(r, c, square)
                self._connect_square_signals(square)
    
    def _create_palette_squares(self):
        """Create palette square widgets"""
        for i, plbl in enumerate(PIECE_LABELS[1:]):
            palette_square = BoardSquareWidget(-1, i, plbl, parent=self, is_palette=True)
            self.ui_manager.add_palette_square(palette_square)
            self._connect_square_signals(palette_square)
        self.ui_manager.finalize_palette()
    
    def _connect_square_signals(self, square):
        """Connect signals from a board square widget"""
        square.squareClicked.connect(self.on_square_clicked)
        square.squareRightClicked.connect(self.on_square_right_clicked)
        square.squareDoubleClicked.connect(self.on_square_double_clicked)
        square.pieceDragStarted.connect(self.on_piece_drag_started)
        square.pieceDropped.connect(self.on_piece_dropped)
        square.enPassantSquareClicked.connect(self.on_ep_square_clicked)
    
    def _connect_ui_signals(self):
        """Connect signals from UI widgets"""
        ui = self.ui_manager
        
        # Radio buttons
        ui.white_rb.clicked.connect(lambda: (self.clear_remembered_piece(), self.refresh_en_passant()))
        ui.black_rb.clicked.connect(lambda: (self.clear_remembered_piece(), self.refresh_en_passant()))
        
        # En passant
        ui.ep_cb.stateChanged.connect(lambda state: (self.clear_remembered_piece(), self.on_ep_toggled(state)))
        
        # Buttons
        ui.clear_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_clear_board()))
        ui.flip_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_flip_board()))
        ui.switch_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_switch_coords()))
        ui.reset_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_reset_to_start()))
        ui.copy_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_copy_fen()))
        ui.paste_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_paste_fen()))
        ui.copy_pgn_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_copy_pgn()))
        ui.learn_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.accept()))
        ui.undo_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_undo()))
        ui.redo_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_redo()))
        ui.play_undo_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_undo()))
        
        # State toggle
        ui.state_toggle_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_state_toggle()))
        
        # Analysis
        ui.analysis_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_analysis()))
        ui.engine_dropdown_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.show_engine_menu()))
        ui.reset_analysis_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.restore_original_position()))
        
        # Analysis line selection
        for i, widget in enumerate(ui.analysis_line_widgets):
            widget.mousePressEvent = lambda event, idx=i: self.select_analysis_line(idx)
            widget.installEventFilter(self)
    
    def _setup_help_system(self):
        """Set up the help system"""
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.help_action = QAction("Help", self)
        self.help_action.triggered.connect(self.show_help)
        self.addAction(self.help_action)
        
        self.help_text = """<b>Chess Board Editor - quick help</b><br>
<br>
<b>Edit Mode (default):</b><br>
• Drag pieces from the palette onto the board.<br>
• Right-click a square to erase it.<br>
• Double-click a piece in the palette to select it for repeated placement.<br>
• <i>Flip Board</i> changes the point of view.<br>
• <i>Undo/Redo</i> restore previous board positions.<br>
• Use the castling checkboxes to control O-O and O-O-O rights.<br>
• To enable <i>en passant</i>, tick the box when a capture is available.<br>
• Click <i>Finish Edit</i> to enter Play mode.<br>
<br>
<b>Play Mode:</b><br>
• Click pieces to select and see legal moves highlighted.<br>
• Click a highlighted square to make the move.<br>
• <i>Undo</i> takes back the last move.<br>
• <i>Copy PGN</i> copies the game notation to clipboard.<br>
• <i>Analysis</i> runs engine on the current position.<br>
• Click <i>Edit Board</i> to return to Edit mode.<br>"""
        
        self.setWhatsThis(self.help_text)
    
    # Square event handlers
    def on_square_clicked(self, row, col, button):
        """Handle square click events"""
        # Left-click on palette piece with remembered piece - delay to allow double-click detection
        if (button == Qt.LeftButton and row < 0 and 
            self.remembered_piece and not self.double_click_in_progress):
            # Store the click info and start timer
            self.pending_palette_click = (row, col, button)
            self.palette_click_timer.start(300)  # 300ms delay
            return
        
        # Right-click with remembered piece - clear it immediately
        if button == Qt.RightButton and self.remembered_piece:
            self.clear_remembered_piece()
            # Continue to handle right-click for erasing
        
        # Left-click on board square with remembered piece (Edit mode only)
        if (button == Qt.LeftButton and row >= 0 and 
            self.remembered_piece and 
            self.state_controller.is_edit_mode):
            # Cancel any pending palette click since we're using the remembered piece
            if self.palette_click_timer.isActive():
                self.palette_click_timer.stop()
                self.pending_palette_click = None
                
            square = self.ui_manager.squares[row][col]
            square.piece_label = self.remembered_piece
            square.update()
            self.sync_squares_to_model()
            self.save_state()
            return
        
        # Play mode click handling
        if (self.state_controller.is_play_mode and 
            button == Qt.LeftButton and row >= 0):
            self.play_mode_controller.handle_square_click(row, col)
            self.sync_model_to_squares()
            self.refresh_ui_from_model()
    
    def _process_delayed_palette_click(self):
        """Process delayed palette click if it wasn't a double-click"""
        if self.pending_palette_click:
            row, col, button = self.pending_palette_click
            # Clear the remembered piece since this was a single click
            self.clear_remembered_piece()
            self.pending_palette_click = None
    
    def on_square_right_clicked(self, row, col):
        """Handle right-click on square"""
        if row >= 0 and self.state_controller.is_edit_mode:
            square = self.ui_manager.squares[row][col]
            if square.piece_label != "empty":
                square.piece_label = "empty"
                square.update()
                self.sync_squares_to_model()
                self.save_state()
    
    def on_square_double_clicked(self, row, col):
        """Handle double-click on square"""
        if row < 0:  # Palette piece
            # Mark that we're processing a double-click
            self.double_click_in_progress = True
            
            # Cancel any pending single-click processing
            if self.palette_click_timer.isActive():
                self.palette_click_timer.stop()
                self.pending_palette_click = None
            
            square = self.ui_manager.palette_squares[col]
            self.set_remembered_piece(square.piece_label)
            
            # Reset the double-click flag after a short delay
            QTimer.singleShot(100, lambda: setattr(self, 'double_click_in_progress', False))
    
    def on_piece_drag_started(self, row, col, piece_label):
        """Handle start of piece drag"""
        if self.state_controller.is_play_mode and row >= 0:
            # Check if this is a valid drag in play mode
            can_drag = self.play_mode_controller.start_drag(row, col)
            if not can_drag:
                # If we can't drag, treat as a click
                self.play_mode_controller.handle_square_click(row, col)
                self.sync_model_to_squares()
                self.refresh_ui_from_model()
    
    def on_piece_dropped(self, from_row, from_col, to_row, to_col, piece_label):
        """Handle piece drop event"""
        # Play mode validation
        if (self.state_controller.is_play_mode and from_row >= 0):
            move = self.play_mode_controller.is_move_legal(from_row, from_col, to_row, to_col)
            if not move:
                # Illegal move - restore original piece
                if from_row >= 0:
                    origin_sq = self.ui_manager.squares[from_row][from_col]
                    origin_sq.piece_label = piece_label
                    origin_sq.update()
                return
            
            # Execute the move
            self.play_mode_controller.execute_drop(move)
            self.sync_model_to_squares()
            self.refresh_ui_from_model()
            return
        
        # Edit mode - allow any drop
        # Clear origin square if it was on the board
        if from_row >= 0:
            origin_sq = self.ui_manager.squares[from_row][from_col]
            origin_sq.piece_label = "empty"
            origin_sq.update()
        
        # Place piece on target square
        target_sq = self.ui_manager.squares[to_row][to_col]
        target_sq.piece_label = piece_label
        target_sq.update()
        
        # Sync and save in Edit mode
        self.sync_squares_to_model()
        if self.state_controller.is_edit_mode:
            self.save_state()
    
    # Helper methods
    def sync_squares_to_model(self):
        """Sync the UI squares with the board model"""
        for r in range(8):
            for c in range(8):
                piece_label = self.ui_manager.squares[r][c].piece_label
                self.board_model.set_piece_at_display_coords(r, c, piece_label)
        self.refresh_ui_from_model()

    def sync_model_to_squares(self):
        """Sync the board model to UI squares"""
        display_labels = self.board_model.get_display_labels()
        for r in range(8):
            for c in range(8):
                self.ui_manager.squares[r][c].piece_label = display_labels[r][c]
                self.ui_manager.squares[r][c].update()

    def refresh_ui_from_model(self):
        """Refresh UI elements from the current board model state"""
        self.refresh_castling_checkboxes()
        self.refresh_en_passant()
        
        # Update side to move radio buttons
        side = self.board_model.get_side_to_move()
        self.ui_manager.white_rb.setChecked(side == 'w')
        self.ui_manager.black_rb.setChecked(side == 'b')

    def save_state(self):
        """Save current board state to history"""
        self.sync_squares_to_model()
        self.history_manager.save_state(self.board_model)

    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons"""
        self.ui_manager.undo_btn.setEnabled(self.history_manager.can_undo())
        self.ui_manager.redo_btn.setEnabled(self.history_manager.can_redo())

    def on_undo(self):
        """Restore the previous board state or undo last move"""
        if self.state_controller.is_edit_mode:
            # Edit mode: use history manager for board state undo
            restored_state = self.history_manager.undo()
            if restored_state:
                self.board_model = restored_state
                self.sync_model_to_squares()
                self.refresh_ui_from_model()
        else:
            # Play mode: undo last move
            move = self.pgn_manager.undo_last_move()
            if move:
                # Revert the move on the internal board
                internal_board = self.board_model.get_internal_board()
                internal_board.pop()
                
                # Update display
                self.sync_model_to_squares()
                self.refresh_ui_from_model()
                
                # Update undo button state
                self.ui_manager.play_undo_btn.setEnabled(self.pgn_manager.has_moves())

    def on_redo(self):
        """Restore the next board state"""
        restored_state = self.history_manager.redo()
        if restored_state:
            self.board_model = restored_state
            self.sync_model_to_squares()
            self.refresh_ui_from_model()

    def on_clear_board(self):
        """Clear all pieces from the board"""
        self.board_model.clear_board()
        self.sync_model_to_squares()
        self.refresh_ui_from_model()
        self.save_state()

    def on_flip_board(self):
        """Flip the board orientation and update display"""
        self.board_model.flip_display_orientation()
        
        # Update coordinate labels
        self.ui_manager.file_labels.reverse()
        self.ui_manager.rank_labels.reverse()
        self.ui_manager.update_coordinate_labels(
            self.ui_manager.file_labels,
            self.ui_manager.rank_labels
        )
        
        # Update squares to reflect new orientation
        self.sync_model_to_squares()
        self.refresh_ui_from_model()
        self.save_state()

    def on_switch_coords(self):
        """Switch coordinate labels only (not board orientation)"""
        self.ui_manager.file_labels.reverse()
        self.ui_manager.rank_labels.reverse()
        self.ui_manager.update_coordinate_labels(
            self.ui_manager.file_labels,
            self.ui_manager.rank_labels
        )
        self.refresh_ui_from_model()

    def on_reset_to_start(self):
        """Reset to standard chess starting position"""
        self.board_model.reset_to_starting_position()
        self.sync_model_to_squares()
        self.refresh_ui_from_model()
        self.save_state()

    def on_copy_fen(self):
        """Copy current position as FEN to clipboard"""
        self.sync_squares_to_model()
        
        # Update board model with current UI state
        self.board_model.set_side_to_move(self.get_side_to_move())
        castling_rights = self.get_castling_rights()
        self.board_model.set_castling_rights(castling_rights)
        if hasattr(self, 'ep_selected') and self.ep_selected:
            self.board_model.set_en_passant_square(self.ep_selected)
        else:
            self.board_model.set_en_passant_square(None)
            
        fen = self.board_model.get_fen()
        QApplication.clipboard().setText(fen)
        QMessageBox.information(self, "FEN copied", fen)
        
    def on_paste_fen(self):
        """Parse a FEN string from clipboard and update the board"""
        clipboard = QApplication.clipboard()
        fen_text = clipboard.text().strip()
        
        try:
            # Set board from FEN (this validates it automatically)
            self.board_model.set_from_fen(fen_text)
            
            # Reset to standard view orientation (White's POV) for consistency
            self.ui_manager.file_labels = list("abcdefgh")
            self.ui_manager.rank_labels = list("87654321")
            self.board_model.is_display_flipped = False
            
            # Update coordinate labels in the UI
            self.ui_manager.update_coordinate_labels(
                self.ui_manager.file_labels,
                self.ui_manager.rank_labels
            )
            
            # Sync model to UI
            self.sync_model_to_squares()
            self.refresh_ui_from_model()
            
            # In Edit mode, save state to history
            if self.state_controller.is_edit_mode:
                self.save_state()
            else:
                # In Play mode, restart the game with new position
                self.pgn_manager.start_new_game(self.board_model.get_fen())
                self.update_move_list_display()
                self.ui_manager.play_undo_btn.setEnabled(False)
            
            QMessageBox.information(self, "FEN Loaded", "FEN position loaded successfully")
        
        except Exception as e:
            QMessageBox.warning(self, "Invalid FEN", f"Could not parse FEN: {str(e)}")

    def show_engine_menu(self):
        """Display a popup menu with available UCI engines"""
        engine_menu = QMenu(self)
        engines = self.analysis_manager.get_available_engines()
        
        if engines:
            # Create menu items for each engine
            for engine in engines:
                # Format display name
                if os.path.sep in engine:
                    display_name = engine
                else:
                    display_name = engine
                    
                action = engine_menu.addAction(display_name)
                action.triggered.connect(lambda checked, e=engine: self.select_engine(e))
        else:
            # No engines found
            no_engine_action = engine_menu.addAction("No engines found")
            no_engine_action.setEnabled(False)
            
            # Add option to add a new engine
            engine_menu.addSeparator()
            add_engine_action = engine_menu.addAction("Add engine...")
            add_engine_action.triggered.connect(self.add_engine)
        
        # Show the menu
        pos = self.ui_manager.engine_dropdown_btn.mapToGlobal(
            self.ui_manager.engine_dropdown_btn.rect().bottomLeft()
        )
        engine_menu.exec_(pos)
    
    def select_engine(self, engine_name):
        """Set the selected engine"""
        success = self.analysis_manager.select_engine(engine_name)
        
        if success:
            # Update the analysis button text
            display_name = self.analysis_manager.get_selected_engine_name()
            if display_name:
                self.ui_manager.analysis_btn.setText(f"Analysis ({display_name})")
        else:
            QMessageBox.warning(
                self, 
                "Engine Error", 
                f"Could not select engine '{engine_name}'. File may not exist."
            )

    def add_engine(self):
        """Open a file dialog to select an engine executable"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        if sys.platform.startswith("win"):
            file_dialog.setNameFilter("Executable files (*.exe)")
        else:
            file_dialog.setNameFilter("All files (*)")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                source_path = selected_files[0]
                engine_name = os.path.basename(source_path)
                
                # Create engine directory if it doesn't exist
                engine_dir = os.path.join(os.path.dirname(__file__), "engine")
                os.makedirs(engine_dir, exist_ok=True)
                
                # Copy the engine to the engine directory
                destination_path = os.path.join(engine_dir, engine_name)
                try:
                    import shutil
                    shutil.copy2(source_path, destination_path)
                    
                    # Make it executable on Unix systems
                    if not sys.platform.startswith("win"):
                        os.chmod(destination_path, os.stat(destination_path).st_mode | 0o111)
                    
                    # Select the newly added engine
                    self.select_engine(engine_name)
                    QMessageBox.information(self, "Engine Added", f"Engine '{engine_name}' added successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to add engine: {str(e)}")

    def on_analysis(self):
        """Run UCI engine analysis"""
        # Show analyzing state
        self.ui_manager.placeholder_label.setText("Analyzing...")
        self.ui_manager.placeholder_label.show()
        for widget in self.ui_manager.analysis_line_widgets:
            widget.hide()
        
        # In Edit mode, sync current state to model and update UI state
        if self.state_controller.is_edit_mode:
            self.sync_squares_to_model()
            
            # Update board model with current UI state
            self.board_model.set_side_to_move(self.get_side_to_move())
            castling_rights = self.get_castling_rights()
            self.board_model.set_castling_rights(castling_rights)
            if hasattr(self, 'ep_selected') and self.ep_selected:
                self.board_model.set_en_passant_square(self.ep_selected)
            else:
                self.board_model.set_en_passant_square(None)
        
        # Run analysis
        success = self.analysis_manager.analyze_position(self.board_model)
        
        if success:
            # Enable reset button
            self.ui_manager.reset_analysis_btn.setEnabled(True)
            
            # Set focus to first line widget
            if self.ui_manager.analysis_line_widgets:
                self.ui_manager.analysis_line_widgets[0].setFocus()
        else:
            self.ui_manager.placeholder_label.setText("Analysis failed")

    def update_board_from_model(self, new_model: ChessBoardModel):
        """Callback to update board from a new model (used by analysis manager)"""
        self.board_model = new_model
        self.sync_model_to_squares()
        self.refresh_ui_from_model()
        
        # Update play mode controller's board model reference
        self.play_mode_controller.board_model = new_model

    def update_analysis_display(self):
        """Callback to update analysis display (used by analysis manager)"""
        lines, selected_index, move_indices, has_navigated = self.analysis_manager.get_analysis_display_data()
        
        if not lines:
            # Hide all line widgets and show placeholder
            for widget in self.ui_manager.analysis_line_widgets:
                widget.hide()
            self.ui_manager.placeholder_label.setText("No analysis available")
            self.ui_manager.placeholder_label.show()
            return
        
        # Hide placeholder and show line widgets
        self.ui_manager.placeholder_label.hide()
        
        # Update each line widget
        for i, line in enumerate(lines):
            if i < len(self.ui_manager.analysis_line_widgets):
                widget = self.ui_manager.analysis_line_widgets[i]
                
                # Parse and highlight the line
                highlighted_line = self._highlight_analysis_line(line, i, move_indices, has_navigated, selected_index)
                widget.setText(highlighted_line)
                widget.show()
                
                # Apply selection highlighting
                if i == selected_index:
                    widget.setStyleSheet("padding: 5px; border: 1px solid #666666; background-color: #e6f3ff; color: #000000;")
                else:
                    widget.setStyleSheet("padding: 5px; border: 1px solid transparent;")
        
        # Hide unused widgets
        for i in range(len(lines), len(self.ui_manager.analysis_line_widgets)):
            self.ui_manager.analysis_line_widgets[i].hide()

    def _highlight_analysis_line(self, line: str, line_index: int, move_indices: list, has_navigated: list, selected_index: int) -> str:
        """Apply highlighting to analysis line text with proper move-by-move highlighting"""
        parts = line.split('|')
        if len(parts) != 2:
            return line
        
        score_part = parts[0].strip()
        moves_part = parts[1].strip()
        
        # Only highlight if this is the selected line and user has navigated
        if line_index == selected_index and has_navigated[line_index]:
            current_move_idx = move_indices[line_index]
            
            # Parse moves from the algebraic notation
            highlighted_moves = self._parse_and_highlight_moves(moves_part, current_move_idx)
            return f"{score_part} | {highlighted_moves}"
        
        return line
    
    def _parse_and_highlight_moves(self, moves_text: str, current_move_idx: int) -> str:
        """Parse algebraic notation and highlight the current move"""
        if current_move_idx <= 0:
            return moves_text
            
        # Split the moves while preserving the original format
        import re
        
        # Pattern to match move numbers and moves
        pattern = r'(\d+\.\.?\.?)\s*([^\s]+)(?:\s+([^\s]+))?'
        matches = re.findall(pattern, moves_text)
        
        if not matches:
            # Fallback: simple space-based splitting if regex fails
            parts = moves_text.split()
            if current_move_idx-1 < len(parts):
                highlighted_parts = []
                for i, part in enumerate(parts):
                    if i == current_move_idx-1:
                        highlighted_parts.append(f'<span style="background-color: yellow; font-weight: bold;">{part}</span>')
                    else:
                        highlighted_parts.append(part)
                return ' '.join(highlighted_parts)
            return moves_text
        
        # Reconstruct the move text with highlighting
        highlighted_text = ""
        move_count = 0
        
        for match in matches:
            move_num, white_move, black_move = match
            
            # Add move number
            highlighted_text += move_num + " "
            
            # Add white move
            move_count += 1
            if move_count == current_move_idx:
                highlighted_text += f'<span style="background-color: yellow; font-weight: bold;">{white_move}</span>'
            else:
                highlighted_text += white_move
            
            # Add black move if it exists
            if black_move:
                highlighted_text += " "
                move_count += 1
                if move_count == current_move_idx:
                    highlighted_text += f'<span style="background-color: yellow; font-weight: bold;">{black_move}</span>'
                else:
                    highlighted_text += black_move
            
            highlighted_text += " "
        
        return highlighted_text.strip()

    def select_analysis_line(self, index):
        """Select a specific analysis line"""
        self.analysis_manager.select_line(index)
        if self.ui_manager.analysis_line_widgets and index < len(self.ui_manager.analysis_line_widgets):
            self.ui_manager.analysis_line_widgets[index].setFocus()

    def get_final_labels_2d(self):
        """Get final board state as 2D labels (for compatibility)"""
        self.sync_squares_to_model()
        return self.board_model.get_display_labels()
    
    def get_side_to_move(self):
        """Get side to move from UI"""
        return 'w' if self.ui_manager.white_rb.isChecked() else 'b'
        
    def get_castling_rights(self) -> str:
        """Get castling rights from UI checkboxes"""
        ui = self.ui_manager
        rights = ""
        if ui.w_k_cb.isChecked(): rights += "K"
        if ui.w_q_cb.isChecked(): rights += "Q"
        if ui.b_k_cb.isChecked(): rights += "k"
        if ui.b_q_cb.isChecked(): rights += "q"
        return rights or "-"
    
    def refresh_castling_checkboxes(self):
        """Enable castling boxes based on piece positions and update from model"""
        # Check which castling rights are possible based on current position
        WK = self.board_model.can_castle('w', True)
        WQ = self.board_model.can_castle('w', False)
        BK = self.board_model.can_castle('b', True)
        BQ = self.board_model.can_castle('b', False)

        ui = self.ui_manager
        # Update checkboxes
        for cb, ok in ((ui.w_k_cb, WK), (ui.w_q_cb, WQ),
                      (ui.b_k_cb, BK), (ui.b_q_cb, BQ)):
            cb.blockSignals(True)
            cb.setEnabled(ok)
            if not ok:
                cb.setChecked(False)
            else:
                # Set from model's current state
                current_rights = self.board_model.get_castling_rights()
                if cb == ui.w_k_cb:
                    cb.setChecked('K' in current_rights)
                elif cb == ui.w_q_cb:
                    cb.setChecked('Q' in current_rights)
                elif cb == ui.b_k_cb:
                    cb.setChecked('k' in current_rights)
                elif cb == ui.b_q_cb:
                    cb.setChecked('q' in current_rights)
            cb.blockSignals(False)

    def refresh_en_passant(self):
        """Re-evaluate EP candidates and update UI"""
        # First clear any existing highlights
        if self.ep_highlight_on:
            for (r,c) in self.ep_possible:
                self.ui_manager.squares[r][c].set_highlight(False)
            self.ep_highlight_on = False
            
        # Get en passant targets from the model
        self.ep_possible = self.board_model.get_en_passant_targets()
        ok = self.board_model.has_en_passant_candidates()

        ui = self.ui_manager
        ui.ep_cb.blockSignals(True)
        ui.ep_cb.setEnabled(ok)
        if not ok:
            ui.ep_cb.setChecked(False)
            self.ep_selected = None
        
        # If the checkbox is still checked (or was already checked), highlight the squares
        if ui.ep_cb.isChecked() and self.ep_possible:
            for (r, c) in self.ep_possible:
                self.ui_manager.squares[r][c].set_highlight(True)
            self.ep_highlight_on = True
            
        ui.ep_cb.blockSignals(False)
        
        # Update en passant state for all squares
        for row in self.ui_manager.squares:
            for square in row:
                square.set_ep_state(self.ep_highlight_on, self.ep_possible)

    def on_ep_toggled(self, state):
        """User ticked/unticked the EP box."""
        checked = state == Qt.Checked
        # clear any previous highlight
        for (r,c) in self.ep_possible:
            self.ui_manager.squares[r][c].set_highlight(False)
        self.ep_highlight_on = False
        if checked:
            for (r, c) in self.ep_possible:
                self.ui_manager.squares[r][c].set_highlight(True)
            self.ep_highlight_on = True
            self.ep_selected = None
        else:
            self.ep_selected = None
            
        # Update en passant state for all squares
        for row in self.ui_manager.squares:
            for square in row:
                square.set_ep_state(False, {})

    def on_ep_square_clicked(self, row, col):
        """User chose which EP target square goes into the FEN."""
        if (row, col) not in self.ep_possible:
            return

        # remember the algebraic square (e.g. "g6")
        self.ep_selected = self.ep_possible[(row, col)]

        # turn OFF highlights
        for (r, c) in list(self.ep_possible):
            self.ui_manager.squares[r][c].set_highlight(False)
        self.ep_highlight_on = False
        self.ep_possible.clear()

        # keep the check‑box ticked (block signals to avoid re‑entry)
        ui = self.ui_manager
        ui.ep_cb.blockSignals(True)
        ui.ep_cb.setChecked(True)
        ui.ep_cb.blockSignals(False)
        
        # Update en passant state for all squares
        for row in self.ui_manager.squares:
            for square in row:
                square.set_ep_state(False, {})

    def get_ep_field(self):
        """Get the en passant field for FEN string"""
        return self.ep_selected if (self.ui_manager.ep_cb.isChecked() and self.ep_selected) else "-"

    def show_help(self):
        """Show help information in a message box"""
        QMessageBox.information(self, "Board Editor Help", self.help_text)

    def event(self, e):
        if e.type() == QEvent.EnterWhatsThisMode:
            # When the ? button in the title bar is clicked, show our custom help instead
            QWhatsThis.leaveWhatsThisMode()  # Exit "What's This?" mode immediately 
            self.show_help()
            return True
        return super().event(e)

    def set_remembered_piece(self, piece_label):
        """Set or clear the currently remembered piece"""
        # Clear previous piece if any
        if self.remembered_piece == piece_label:
            # If clicking the same piece, toggle it off
            self.remembered_piece = None
            QApplication.restoreOverrideCursor()
            # Update palette squares
            for square in self.ui_manager.palette_squares:
                square.set_memory_highlight(False)
            return
            
        # Clear any existing remembered piece highlighting
        self.remembered_piece = None
        for palette_square in self.ui_manager.palette_squares:
            palette_square.set_memory_highlight(False)
            
        # Set new piece
        self.remembered_piece = piece_label
        
        # Highlight the palette square
        for square in self.ui_manager.palette_squares:
            if square.piece_label == piece_label:
                square.set_memory_highlight(True)
        
        # Create a custom cursor with the piece
        if piece_label != "empty":
            # Get the piece pixmap
            pixmap = get_piece_pixmap(piece_label).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create and set the cursor
            if QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(QCursor(pixmap, 0, 0))
        else:
            # Restore default cursor if selecting "empty"
            if QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()
            self.remembered_piece = None
    
    def clear_remembered_piece(self):
        """Clear the currently remembered piece"""
        # Cancel any pending palette click processing
        if self.palette_click_timer.isActive():
            self.palette_click_timer.stop()
            self.pending_palette_click = None
            
        self.remembered_piece = None
        if QApplication.overrideCursor():
            QApplication.restoreOverrideCursor()
            
        # Make sure to update palette squares to clear highlight
        for palette_square in self.ui_manager.palette_squares:
            palette_square.set_memory_highlight(False)
            
    def mousePressEvent(self, event):
        """Clear remembered piece if clicking outside of board squares"""
        if self.remembered_piece:
            # Clear remembered piece on any right-click anywhere in the window
            if event.button() == Qt.RightButton:
                self.clear_remembered_piece()
            # For left-clicks, only clear if clicking outside board squares
            elif event.button() == Qt.LeftButton:
                # Check if we clicked on something that isn't a board square
                widget = self.childAt(event.pos())
                if not widget or not isinstance(widget, BoardSquareWidget):
                    self.clear_remembered_piece()
        super().mousePressEvent(event)

    def eventFilter(self, obj, event):
        """Handle key events for analysis line navigation"""
        if obj in self.ui_manager.analysis_line_widgets and event.type() == QEvent.KeyPress:
            # Handle arrow keys for analysis navigation
            if event.key() == Qt.Key_Right:
                self.analysis_manager.navigate_line('next')
                return True
            elif event.key() == Qt.Key_Left:
                self.analysis_manager.navigate_line('previous')
                return True
            elif event.key() == Qt.Key_Up:
                self.analysis_manager.navigate_line('line_up')
                return True
            elif event.key() == Qt.Key_Down:
                self.analysis_manager.navigate_line('line_down')
                return True
            elif event.key() == Qt.Key_Home:
                self.analysis_manager.navigate_line('start')
                return True
            elif event.key() == Qt.Key_End:
                self.analysis_manager.navigate_line('end')
                return True
        return super().eventFilter(obj, event)
        
    def restore_original_position(self):
        """Restore the original position from before analysis"""
        success = self.analysis_manager.restore_original_position()
        if not success:
            QMessageBox.information(self, "No Original Position", "No original position to restore.")
    
    def on_state_toggle(self):
        """Handle state toggle button click"""
        if self.state_controller.is_edit_mode:
            # Transitioning to Play mode - sync ALL UI state before validation
            self.sync_squares_to_model()
            
            # Update board model with current UI state
            self.board_model.set_side_to_move(self.get_side_to_move())
            castling_rights = self.get_castling_rights()
            self.board_model.set_castling_rights(castling_rights)
            if hasattr(self, 'ep_selected') and self.ep_selected:
                self.board_model.set_en_passant_square(self.ep_selected)
            else:
                self.board_model.set_en_passant_square(None)
            
            # Now validate the complete position
            is_valid, error_msg = self.board_model.validate_position()
            if not is_valid:
                QMessageBox.warning(self, "Invalid Position", 
                                  f"Cannot switch to Play mode:\n\n{error_msg}")
                return
            
            # Transition to Play state
            self.state_controller.transition_to_play()
            
            # Initialize PGN manager with current position
            self.pgn_manager.start_new_game(self.board_model.get_fen())
            
        else:
            # Transitioning to Edit mode
            self.state_controller.transition_to_edit()
    
    def on_state_changed(self, new_state: BoardState):
        """Handle state change callback"""
        # Clear any play mode selection state
        self.play_mode_controller.clear_selection()
        
        # Apply UI state changes
        self.editor_state_manager.handle_state_change(new_state)
        
        # Update move list display if entering play mode
        if new_state == BoardState.PLAY:
            self.update_move_list_display()
    
    def update_move_list_display(self):
        """Update the move list text display"""
        move_text = self.pgn_manager.get_move_list_text()
        self.ui_manager.move_list_text.setText(move_text)
        
        # Scroll to bottom to show latest moves
        scrollbar = self.ui_manager.move_list_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_copy_pgn(self):
        """Copy PGN to clipboard"""
        pgn_text = self.pgn_manager.get_pgn_text()
        if pgn_text:
            QApplication.clipboard().setText(pgn_text)
            QMessageBox.information(self, "PGN Copied", "PGN text copied to clipboard")
        else:
            QMessageBox.information(self, "No Moves", "No moves to copy")