import os
import sys
import chess
from BoardSquareWidget import BoardSquareWidget
from labels import (PIECE_LABELS, get_piece_pixmap)
from ChessBoardModel import ChessBoardModel
from HistoryManager import HistoryManager
from AnalysisManager import AnalysisManager
from StateController import StateController, BoardState
from PgnManager import PgnManager
from PyQt5.QtCore import Qt, QEvent, QSettings
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QDialog, QGridLayout, QMessageBox, QHBoxLayout,
    QCheckBox, QRadioButton, QButtonGroup, QWhatsThis, QAction, QMenu, 
    QFileDialog, QSizePolicy, QTextEdit, QScrollArea
)


#########################################
# BoardEditor
#########################################

class BoardEditor(QDialog):
    """
    The dialog that shows the 8x8 board + palette + coordinate labels and:
        • Clear Board   • Flip Board   • Switch Coord   • Reset
        • Copy FEN      • Learn        • right-hand Stockfish Analysis panel
    """
    def __init__(self, labels_2d, predicted_side_to_move='w'):
        super().__init__()
        self.setWindowTitle("Board Editor")
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
        
        # Set up callbacks
        self.history_manager.set_update_callback(self.update_undo_redo_buttons)
        self.analysis_manager.set_update_display_callback(self.update_analysis_display)
        self.analysis_manager.set_update_board_callback(self.update_board_from_model)
        self.state_controller.set_state_change_callback(self.on_state_changed)
        self.pgn_manager.set_update_callback(self.update_move_list_display)
        
        # Settings storage for user preferences
        self.settings = QSettings("ChessAIScanner", "Settings")
        
        # Initialize piece memory feature
        self.remembered_piece = None  # Currently remembered piece
        
        # Disable automatic "What's this?" on right-click
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        # Create a help action for direct access via the help button
        self.help_action = QAction("Help", self)
        self.help_action.triggered.connect(self.show_help)
        self.addAction(self.help_action)

        # Define help text in just one place
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

        # icon paths
        base_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
        def ip(name): return os.path.join(base_icon, name)
        self.setWindowIcon(QIcon(ip("Chess_icon.png")))
        w_path, b_path        = ip("Chess_klt.png"),  ip("Chess_kdt.png")
        switch_path, flip_path = ip("Chess_switch.png"), ip("Chess_flip.png")
        clip_path, nn_path    = ip("Chess_clip.png"),  ip("Chess_nn.png")
        paste_path = ip("Chess_paste.png")
        bcastle_path, wcastle_path = ip("Chess_bcastle.png"), ip("Chess_wcastle.png")
        cengine_path = ip("Chess_engine.png")
        undo_path = ip("Chess_undo.png")
        redo_path = ip("Chess_redo.png")

        # Coordinate labels - sync with board model orientation
        self.file_labels = list("abcdefgh")
        self.rank_labels = list("87654321")
        
        # Apply board model orientation to coordinate labels
        if self.board_model.is_display_flipped:
            self.file_labels.reverse()
            self.rank_labels.reverse()

        outer = QHBoxLayout(self)
        left  = QVBoxLayout()       # board + controls
        outer.addLayout(left)
        self.setLayout(outer)

        # top row – side‑to‑move radio + castling boxes + en passant
        top_row = QHBoxLayout(); left.addLayout(top_row)
        self.white_rb = QRadioButton(); self.white_rb.setIcon(QIcon(w_path))
        self.black_rb = QRadioButton(); self.black_rb.setIcon(QIcon(b_path))
        self.white_rb.setChecked(predicted_side_to_move == 'w')  # Use predicted side
        self.black_rb.setChecked(predicted_side_to_move == 'b')  # Use predicted side
        side_group = QButtonGroup(self); side_group.addButton(self.white_rb); side_group.addButton(self.black_rb)
        
        # Connect side-to-move radio buttons to en passant refresh
        self.white_rb.clicked.connect(lambda: (self.clear_remembered_piece(), self.refresh_en_passant()))
        self.black_rb.clicked.connect(lambda: (self.clear_remembered_piece(), self.refresh_en_passant()))
        
        self.ep_cb = QCheckBox("en passant")
        self.ep_cb.setEnabled(False)
        self.ep_cb.stateChanged.connect(lambda state: (self.clear_remembered_piece(), self.on_ep_toggled(state)))
        top_row.addWidget(self.ep_cb)
        self.ep_possible   = {}   # {(r,c): "e3", ...} - changed from ep_pawn_candidates
        self.ep_selected   = None # "e3"  (FEN field) or None
        self.ep_highlight_on = False


        # castling rights check‑boxes
        self.w_k_cb = QCheckBox("\nO-O");   self.w_k_cb.setIcon(QIcon(wcastle_path))
        self.w_q_cb = QCheckBox("\nO-O-O"); self.w_q_cb.setIcon(QIcon(wcastle_path))
        self.b_k_cb = QCheckBox("\nO-O");   self.b_k_cb.setIcon(QIcon(bcastle_path))
        self.b_q_cb = QCheckBox("\nO-O-O"); self.b_q_cb.setIcon(QIcon(bcastle_path))
        for cb in (self.w_k_cb, self.w_q_cb, self.b_k_cb, self.b_q_cb):
            cb.setChecked(True)

        for w in (self.white_rb, self.black_rb, self.w_k_cb, self.w_q_cb, self.b_k_cb, self.b_q_cb):
            top_row.addWidget(w)
        top_row.addStretch()

        # chessboard grid
        grid_container = QWidget(); self.grid = QGridLayout(grid_container)
        left.addWidget(grid_container)
        
        # Set fixed size to ensure proper alignment (8 squares of 60px each plus labels)
        grid_container.setFixedSize(9*60 + 70, 10*60)  # Add 70 pixels for the Undo/Redo buttons
        
        # Make squares closer together by removing spacing
        self.grid.setSpacing(0)
        self.grid.setContentsMargins(0, 0, 0, 0)
        
        # Set specific style on the grid container to prevent any gap issues
        grid_container.setStyleSheet("padding: 0px; margin: 0px; border: 0px; spacing: 0px;")
        
        # Force grid to align squares perfectly
        self.grid.setVerticalSpacing(0)
        self.grid.setHorizontalSpacing(0)
        grid_container.setContentsMargins(0, 0, 0, 0)
        
        # Set stretch factors to fill all available space
        for i in range(9):  # 8 rows + 1 for labels
            self.grid.setRowStretch(i, 1)
            self.grid.setColumnStretch(i, 1)
        
        self.squares = []
        display_labels = self.board_model.get_display_labels()
        for r in range(8):
            # rank label
            rank_lbl = QLabel(self.rank_labels[r]); rank_lbl.setAlignment(Qt.AlignCenter)
            self.grid.addWidget(rank_lbl, r+1, 0)
            row_widgets = []
            for c in range(8):
                sq = BoardSquareWidget(r, c, display_labels[r][c], parent=self)
                self.grid.addWidget(sq, r+1, c+1)
                row_widgets.append(sq)
            self.squares.append(row_widgets)
        
        # file labels bottom
        for c in range(8):
            file_lbl = QLabel(self.file_labels[c]); file_lbl.setAlignment(Qt.AlignCenter)
            self.grid.addWidget(file_lbl, 9, c+1)
        # blank bottom‑left pad
        self.grid.addWidget(QLabel(""), 9, 0)
        
        # Add Undo/Redo buttons to the right of ranks 8 and 7
        self.undo_btn = QPushButton("Undo"); self.undo_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_undo()))
        self.redo_btn = QPushButton("Redo"); self.redo_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_redo()))
        
        # Set minimum width to prevent text from being cut off
        self.undo_btn.setMinimumWidth(60)
        self.redo_btn.setMinimumWidth(60)
        
        self.undo_btn.setIcon(QIcon(undo_path))
        self.redo_btn.setIcon(QIcon(redo_path))           
        # Initialize buttons as disabled
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        
        # Add buttons to the grid - red region (rank 8) for Undo, blue region (rank 7) for Redo
        self.grid.addWidget(self.undo_btn, 1, 9)  # row 1 = rank 8, column 9 = right of board
        self.grid.addWidget(self.redo_btn, 2, 9)  # row 2 = rank 7, column 9 = right of board
        
        # Set the column stretch for the button column
        self.grid.setColumnStretch(9, 1)

        # palette
        bottom_container = QWidget()
        palette_layout = QHBoxLayout(bottom_container)
        palette_layout.setSpacing(0)  # Remove spacing between palette pieces
        palette_layout.setContentsMargins(0, 0, 0, 0)  # Remove padding around palette
        
        # Apply stylesheet to ensure no gaps in palette
        bottom_container.setStyleSheet("padding: 0px; margin: 0px; border: 0px; spacing: 0px;")
        
        # Ensure layout has no spacing
        palette_layout.setSpacing(0)
        
        # Remove any margins from the container widget
        bottom_container.setContentsMargins(0, 0, 0, 0)
        
        left.addWidget(bottom_container)
        
        # Add pieces to palette with no spacing
        self.palette_squares = []
        for i, plbl in enumerate(PIECE_LABELS[1:]):
            palette_square = BoardSquareWidget(-1, i, plbl, parent=self)
            palette_square.is_palette = True
            palette_layout.addWidget(palette_square, 0, Qt.AlignLeft)
            self.palette_squares.append(palette_square)
        
        # Add stretch at the end to keep pieces left-aligned
        palette_layout.addStretch()

        # bottom button bar
        btn_bar = QHBoxLayout(); left.addLayout(btn_bar)
        def add_btn(txt, icon_path, slot):
            def wrapped_slot():
                self.clear_remembered_piece()  # Clear remembered piece on any button click
                slot()
            b = QPushButton(txt); b.clicked.connect(wrapped_slot)
            if icon_path: b.setIcon(QIcon(icon_path))
            btn_bar.addWidget(b); return b

        self.clear_btn  = add_btn("Clear Board",  None,        self.on_clear_board)
        self.flip_btn   = add_btn("Flip Board",   flip_path,   self.on_flip_board)
        self.switch_btn = add_btn("Switch",       switch_path, self.on_switch_coords)
        
        self.reset_btn  = add_btn("Set To Opening",        None,        self.on_reset_to_start)
        self.copy_btn   = add_btn("Copy FEN",     clip_path,   self.on_copy_fen)
        self.paste_btn  = add_btn("Paste FEN",    paste_path,  self.on_paste_fen)
        self.copy_pgn_btn = add_btn("Copy PGN",   clip_path,   self.on_copy_pgn)
        self.learn_btn  = add_btn("Learn",        nn_path,     self.accept)
        btn_bar.addStretch()

        # analysis column
        right = QVBoxLayout(); outer.addLayout(right)
        
        # Add Finish Edit / Edit Board button
        self.state_toggle_btn = QPushButton("Finish Edit")
        self.state_toggle_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_state_toggle()))
        right.addWidget(self.state_toggle_btn)
        
        # Create analysis button group with dropdown
        analysis_container = QWidget()
        analysis_layout = QHBoxLayout(analysis_container)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(0)
        
        # Main Analysis button
        self.analysis_btn = QPushButton("Analysis")
        self.analysis_btn.setIcon(QIcon(cengine_path))
        self.analysis_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_analysis()))
        
        # Dropdown button for engine selection
        self.engine_dropdown_btn = QPushButton("▼")
        self.engine_dropdown_btn.setMaximumWidth(20)  # Make dropdown button narrow
        
        # Ensure the dropdown button has the same height and styling as the Analysis button
        self.engine_dropdown_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.engine_dropdown_btn.setFixedHeight(self.analysis_btn.sizeHint().height())
        self.engine_dropdown_btn.setStyleSheet("QPushButton { padding: 0px; }")
        
        self.engine_dropdown_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.show_engine_menu()))
        
        # Add buttons to container
        analysis_layout.addWidget(self.analysis_btn, 1)  # Main button takes most space
        analysis_layout.addWidget(self.engine_dropdown_btn, 0)  # Dropdown is small
        
        # Add container to layout
        right.addWidget(analysis_container)
        
        # Add a reset button for analysis
        self.reset_analysis_btn = QPushButton("Restore Analysis Board")
        self.reset_analysis_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.restore_original_position()))
        self.reset_analysis_btn.setEnabled(False)  # Initially disabled until analysis is run
        right.addWidget(self.reset_analysis_btn)
        
        # Update analysis button text if engine is loaded
        engine_name = self.analysis_manager.get_selected_engine_name()
        if engine_name:
            self.analysis_btn.setText(f"Analysis ({engine_name})")
        
        # Create navigable analysis lines
        self.analysis_lines = []
        self.analysis_line_widgets = []
        self.selected_line_index = -1
        self.current_move_indices = [0, 0, 0]  # Current position in each line
        self.has_navigated = [False, False, False]  # Track if user has navigated in each line
        
        # Create analysis container
        analysis_view_container = QWidget()
        analysis_view_container.setFixedWidth(280)
        analysis_view_layout = QVBoxLayout(analysis_view_container)
        analysis_view_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create placeholder text
        placeholder_label = QLabel("Engine lines will appear here")
        placeholder_label.setAlignment(Qt.AlignCenter)
        analysis_view_layout.addWidget(placeholder_label)
        self.placeholder_label = placeholder_label
        
        # Create three line widgets
        for i in range(3):
            line_widget = QLabel()
            line_widget.setWordWrap(True)
            line_widget.setTextFormat(Qt.RichText)
            line_widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            line_widget.setMinimumHeight(60)  # Give enough height for the line
            line_widget.setStyleSheet("padding: 5px; border: 1px solid transparent;")
            line_widget.mousePressEvent = lambda event, idx=i: self.select_analysis_line(idx)
            # Make line widgets focusable to receive key events
            line_widget.setFocusPolicy(Qt.StrongFocus)
            
            # Install event filter for key presses
            line_widget.installEventFilter(self)
            
            analysis_view_layout.addWidget(line_widget)
            self.analysis_line_widgets.append(line_widget)
            # Initially hide the line widgets
            line_widget.hide()
        
        right.addWidget(analysis_view_container, 1)
        self.analysis_view_container = analysis_view_container
        
        # Move list panel (for Play mode)
        self.move_list_container = QWidget()
        move_list_layout = QVBoxLayout(self.move_list_container)
        move_list_layout.setContentsMargins(5, 5, 5, 5)
        
        # Move list header
        move_list_header = QLabel("Move List")
        move_list_header.setAlignment(Qt.AlignCenter)
        move_list_header.setStyleSheet("font-weight: bold;")
        move_list_layout.addWidget(move_list_header)
        
        # Move list text area
        self.move_list_text = QTextEdit()
        self.move_list_text.setReadOnly(True)
        self.move_list_text.setFixedWidth(260)
        self.move_list_text.setMinimumHeight(150)
        move_list_layout.addWidget(self.move_list_text)
        
        # Undo button for Play mode (will be moved here from main button bar)
        self.play_undo_btn = QPushButton("Undo Last Move")
        self.play_undo_btn.clicked.connect(lambda: (self.clear_remembered_piece(), self.on_undo()))
        self.play_undo_btn.setToolTip("Undo the last move played")
        move_list_layout.addWidget(self.play_undo_btn)
        
        right.addWidget(self.move_list_container)
        self.move_list_container.hide()  # Initially hidden
        
        right.addStretch()

        # Initialize UI state variables needed for en passant
        self.ep_possible = {}  # Changed from ep_pawn_candidates to match provided implementation
        self.ep_selected = None
        self.ep_highlight_on = False
        
        # Initialize UI-dependent states
        self.refresh_ui_from_model()
        
        # Save initial state to history
        self.save_state()
        
        # Use the same help text for What's This instead of duplicating it
        self.setWhatsThis(self.help_text)
        
        # Apply initial Edit state UI settings
        self.apply_edit_state_ui()

    # Helper methods
    def sync_squares_to_model(self):
        """Sync the UI squares with the board model"""
        for r in range(8):
            for c in range(8):
                piece_label = self.squares[r][c].piece_label
                self.board_model.set_piece_at_display_coords(r, c, piece_label)
        self.refresh_ui_from_model()

    def sync_model_to_squares(self):
        """Sync the board model to UI squares"""
        display_labels = self.board_model.get_display_labels()
        for r in range(8):
            for c in range(8):
                self.squares[r][c].piece_label = display_labels[r][c]
                self.squares[r][c].update()

    def refresh_ui_from_model(self):
        """Refresh UI elements from the current board model state"""
        self.refresh_castling_checkboxes()
        self.refresh_en_passant()
        
        # Update side to move radio buttons
        side = self.board_model.get_side_to_move()
        self.white_rb.setChecked(side == 'w')
        self.black_rb.setChecked(side == 'b')

    def save_state(self):
        """Save current board state to history"""
        self.sync_squares_to_model()
        self.history_manager.save_state(self.board_model)

    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons"""
        self.undo_btn.setEnabled(self.history_manager.can_undo())
        self.redo_btn.setEnabled(self.history_manager.can_redo())

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
                self.play_undo_btn.setEnabled(self.pgn_manager.has_moves())

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
        self.file_labels.reverse()
        self.rank_labels.reverse()
        for r in range(8):
            self.grid.itemAtPosition(r+1, 0).widget().setText(self.rank_labels[r])
        for c in range(8):
            self.grid.itemAtPosition(9, c+1).widget().setText(self.file_labels[c])
        
        # Update squares to reflect new orientation
        self.sync_model_to_squares()
        self.refresh_ui_from_model()
        self.save_state()

    def on_switch_coords(self):
        """Switch coordinate labels only (not board orientation)"""
        self.file_labels.reverse()
        self.rank_labels.reverse()
        for r in range(8):
            self.grid.itemAtPosition(r+1, 0).widget().setText(self.rank_labels[r])
        for c in range(8):
            self.grid.itemAtPosition(9, c+1).widget().setText(self.file_labels[c])
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
            self.file_labels = list("abcdefgh")
            self.rank_labels = list("87654321")
            self.board_model.is_display_flipped = False
            
            # Update coordinate labels in the UI
            for r in range(8):
                self.grid.itemAtPosition(r+1, 0).widget().setText(self.rank_labels[r])
            for c in range(8):
                self.grid.itemAtPosition(9, c+1).widget().setText(self.file_labels[c])
            
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
                self.play_undo_btn.setEnabled(False)
            
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
                # Format display name - use just filename for top-level, show path for subdirectories
                if os.path.sep in engine:
                    display_name = engine  # Show full relative path for nested engines
                else:
                    display_name = engine  # Just filename for top level engines
                    
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
        pos = self.engine_dropdown_btn.mapToGlobal(self.engine_dropdown_btn.rect().bottomLeft())
        engine_menu.exec_(pos)
    
    def select_engine(self, engine_name):
        """Set the selected engine"""
        success = self.analysis_manager.select_engine(engine_name)
        
        if success:
            # Update the analysis button text
            display_name = self.analysis_manager.get_selected_engine_name()
            if display_name:
                self.analysis_btn.setText(f"Analysis ({display_name})")
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
    
    def create_engine_directory(self):
        """Create the engine directory and prompt to add an engine"""
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        try:
            os.makedirs(engine_dir, exist_ok=True)
            QMessageBox.information(self, "Directory Created", "Engine directory created. Now you can add an engine.")
            self.add_engine()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create engine directory: {str(e)}")

    def on_analysis(self):
        """Run UCI engine analysis"""
        # Show analyzing state
        self.placeholder_label.setText("Analyzing...")
        self.placeholder_label.show()
        for widget in self.analysis_line_widgets:
            widget.hide()
        
        # In Edit mode, sync current state to model and update UI state
        if self.state_controller.is_edit_mode:
            self.sync_squares_to_model()
            
            # Update board model with current UI state
            self.board_model.set_side_to_move(self.get_side_to_move())
            castling_rights = self.get_castling_rights()
            self.board_model.set_castling_rights(castling_rights)  # Always set, even if "-"
            if hasattr(self, 'ep_selected') and self.ep_selected:
                self.board_model.set_en_passant_square(self.ep_selected)
            else:
                self.board_model.set_en_passant_square(None)  # Clear en passant if not selected
        
        # In Play mode, the board model is already up to date
        
        # Run analysis
        success = self.analysis_manager.analyze_position(self.board_model)
        
        if success:
            # Enable reset button
            self.reset_analysis_btn.setEnabled(True)
            
            # Update display will be called by the analysis manager callback
            # Set focus to first line widget
            if self.analysis_line_widgets:
                self.analysis_line_widgets[0].setFocus()
        else:
            self.placeholder_label.setText("Analysis failed")

    def update_board_from_model(self, new_model: ChessBoardModel):
        """Callback to update board from a new model (used by analysis manager)"""
        self.board_model = new_model
        self.sync_model_to_squares()
        self.refresh_ui_from_model()

    def update_analysis_display(self):
        """Callback to update analysis display (used by analysis manager)"""
        lines, selected_index, move_indices, has_navigated = self.analysis_manager.get_analysis_display_data()
        
        if not lines:
            # Hide all line widgets and show placeholder
            for widget in self.analysis_line_widgets:
                widget.hide()
            self.placeholder_label.setText("No analysis available")
            self.placeholder_label.show()
            return
        
        # Hide placeholder and show line widgets
        self.placeholder_label.hide()
        
        # Update each line widget
        for i, line in enumerate(lines):
            if i < len(self.analysis_line_widgets):
                widget = self.analysis_line_widgets[i]
                
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
        for i in range(len(lines), len(self.analysis_line_widgets)):
            self.analysis_line_widgets[i].hide()

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
        # This regex matches move numbers (like "1.", "2.", etc.) and moves
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
        if self.analysis_line_widgets and index < len(self.analysis_line_widgets):
            self.analysis_line_widgets[index].setFocus()

    def get_final_labels_2d(self):
        """Get final board state as 2D labels (for compatibility)"""
        self.sync_squares_to_model()
        return self.board_model.get_display_labels()
    
    def get_side_to_move(self):
        """Get side to move from UI"""
        return 'w' if self.white_rb.isChecked() else 'b'
        
    def get_castling_rights(self) -> str:
        """Get castling rights from UI checkboxes"""
        rights = ""
        if self.w_k_cb.isChecked(): rights += "K"
        if self.w_q_cb.isChecked(): rights += "Q"
        if self.b_k_cb.isChecked(): rights += "k"
        if self.b_q_cb.isChecked(): rights += "q"
        return rights or "-"
    
    def refresh_castling_checkboxes(self):
        """Enable castling boxes based on piece positions and update from model"""
        # Check which castling rights are possible based on current position
        WK = self.board_model.can_castle('w', True)   # White kingside
        WQ = self.board_model.can_castle('w', False)  # White queenside
        BK = self.board_model.can_castle('b', True)   # Black kingside
        BQ = self.board_model.can_castle('b', False)  # Black queenside

        # Update checkboxes
        for cb, ok in ((self.w_k_cb, WK), (self.w_q_cb, WQ),
                      (self.b_k_cb, BK), (self.b_q_cb, BQ)):
            cb.blockSignals(True)
            cb.setEnabled(ok)
            if not ok:
                cb.setChecked(False)
            else:
                # Set from model's current state
                current_rights = self.board_model.get_castling_rights()
                if cb == self.w_k_cb:
                    cb.setChecked('K' in current_rights)
                elif cb == self.w_q_cb:
                    cb.setChecked('Q' in current_rights)
                elif cb == self.b_k_cb:
                    cb.setChecked('k' in current_rights)
                elif cb == self.b_q_cb:
                    cb.setChecked('q' in current_rights)
            cb.blockSignals(False)

    def refresh_en_passant(self):
        """Re-evaluate EP candidates and update UI"""
        # First clear any existing highlights
        if self.ep_highlight_on:
            for (r,c) in self.ep_possible:
                self.squares[r][c].set_highlight(False)
            self.ep_highlight_on = False
            
        # Get en passant targets from the model
        self.ep_possible = self.board_model.get_en_passant_targets()
        ok = self.board_model.has_en_passant_candidates()

        self.ep_cb.blockSignals(True)
        self.ep_cb.setEnabled(ok)
        if not ok:
            self.ep_cb.setChecked(False)
            self.ep_selected = None
        
        # If the checkbox is still checked (or was already checked), highlight the squares
        if self.ep_cb.isChecked() and self.ep_possible:
            for (r, c) in self.ep_possible:
                self.squares[r][c].set_highlight(True)
            self.ep_highlight_on = True
            
        self.ep_cb.blockSignals(False)

    def on_ep_toggled(self, state):
        """User ticked/unticked the EP box."""
        checked = state == Qt.Checked
        # clear any previous highlight
        for (r,c) in self.ep_possible:
            self.squares[r][c].set_highlight(False)
        self.ep_highlight_on = False
        if checked:
            for (r, c) in self.ep_possible:            # highlight *targets*
                self.squares[r][c].set_highlight(True)
            self.ep_highlight_on = True
            self.ep_selected = None
        else:
            self.ep_selected = None

    def on_ep_square_clicked(self, row, col):
        """User chose which EP target square goes into the FEN."""
        if (row, col) not in self.ep_possible:
            return

        # remember the algebraic square (e.g. "g6")
        self.ep_selected = self.ep_possible[(row, col)]

        # turn OFF highlights
        for (r, c) in list(self.ep_possible):
            self.squares[r][c].set_highlight(False)
        self.ep_highlight_on = False
        self.ep_possible.clear()            # <- guarantees no stale keys

        # keep the check‑box ticked (block signals to avoid re‑entry)
        self.ep_cb.blockSignals(True)
        self.ep_cb.setChecked(True)
        self.ep_cb.blockSignals(False)

    def get_ep_field(self):
        """Get the en passant field for FEN string"""
        return self.ep_selected if (self.ep_cb.isChecked() and self.ep_selected) else "-"

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
            return
            
        # Clear any existing remembered piece highlighting
        self.remembered_piece = None
        for palette_square in self.palette_squares:
            palette_square.update()
            
        # Set new piece
        self.remembered_piece = piece_label
        
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
        self.remembered_piece = None
        if QApplication.overrideCursor():
            QApplication.restoreOverrideCursor()
            
        # Make sure to update palette squares to clear highlight
        for palette_square in self.palette_squares:
            palette_square.update()
            
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
        if obj in self.analysis_line_widgets and event.type() == QEvent.KeyPress:
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
            
            # Update board model with current UI state (like in on_copy_fen)
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
        if hasattr(self, 'selected_square'):
            # Clear highlights
            if hasattr(self, 'highlighted_squares'):
                for sq_coords in self.highlighted_squares:
                    self.squares[sq_coords[0]][sq_coords[1]].set_highlight(False)
                self.highlighted_squares.clear()
            
            self.selected_square = None
            self.legal_moves = []
        
        # Clear any play mode drag highlights
        self.clear_play_mode_drag_highlights()
        
        if new_state == BoardState.EDIT:
            self.apply_edit_state_ui()
        else:
            self.apply_play_state_ui()
    
    def apply_edit_state_ui(self):
        """Apply UI changes for Edit state - show editing controls, hide play controls"""
        # Update button text
        self.state_toggle_btn.setText("Finish Edit")
        
        # Show editing controls
        self.clear_btn.show()
        self.switch_btn.show()
        self.reset_btn.show()
        self.redo_btn.show()
        self.redo_btn.setEnabled(self.history_manager.can_redo())
        self.undo_btn.show()
        self.undo_btn.setEnabled(self.history_manager.can_undo())
        
        # Show editing controls
        self.flip_btn.show()
        self.paste_btn.show()
        
        # Hide Copy FEN (only available in Play mode)
        self.copy_btn.hide()
        
        # Set Edit mode button text
        self.undo_btn.setText("Undo")
        self.undo_btn.setToolTip("Undo last board edit")
        
        # Hide Play mode undo button
        self.play_undo_btn.hide()
        
        # Show castling and en passant controls
        self.white_rb.show()
        self.black_rb.show()
        self.w_k_cb.show()
        self.w_q_cb.show()
        self.b_k_cb.show()
        self.b_q_cb.show()
        self.ep_cb.show()
        self.refresh_castling_checkboxes()
        self.refresh_en_passant()
        
        # Hide Learn and Analysis
        self.learn_btn.hide()
        # Hide the analysis container (which contains analysis button and dropdown)
        analysis_container = self.analysis_btn.parent()
        if analysis_container:
            analysis_container.hide()
        self.reset_analysis_btn.hide()
        
        # Hide Copy PGN button (will be shown in Play mode)
        self.copy_pgn_btn.hide()
        
        # Show palette
        bottom_container = self.palette_squares[0].parent()
        if bottom_container:
            bottom_container.show()
        
        # Hide move list
        self.move_list_container.hide()
        
        # Enable piece dragging on board
        for row in self.squares:
            for square in row:
                square.setAcceptDrops(True)
    
    def apply_play_state_ui(self):
        """Apply UI changes for Play state - hide editing controls, show play controls"""
        # Update button text
        self.state_toggle_btn.setText("Edit Board")
        
        # Hide most editing controls
        self.clear_btn.hide()
        self.switch_btn.hide()
        self.reset_btn.hide()
        self.redo_btn.hide()
        
        # These controls remain visible in Play mode
        self.flip_btn.show()
        self.copy_btn.show()
        self.paste_btn.hide()  # Hide Paste FEN in Play mode
        self.copy_pgn_btn.show()  # Show Copy PGN in main button bar
        
        # Hide main undo button from button bar, use the one in move list
        self.undo_btn.hide()
        
        # Show and enable Play mode undo button in move list
        self.play_undo_btn.show()
        self.play_undo_btn.setEnabled(self.pgn_manager.has_moves())
        
        # Hide castling and en passant controls
        self.white_rb.hide()
        self.black_rb.hide()
        self.w_k_cb.hide()
        self.w_q_cb.hide()
        self.b_k_cb.hide()
        self.b_q_cb.hide()
        self.ep_cb.hide()
        
        # Show Learn and Analysis
        self.learn_btn.show()
        # Show the analysis container (which contains analysis button and dropdown)
        analysis_container = self.analysis_btn.parent()
        if analysis_container:
            analysis_container.show()
        self.reset_analysis_btn.show()
        
        # Hide palette
        bottom_container = self.palette_squares[0].parent()
        if bottom_container:
            bottom_container.hide()
        
        # Show move list
        self.move_list_container.show()
        self.update_move_list_display()
        
        # Enable piece dragging for Play mode (legal moves only)
        for row in self.squares:
            for square in row:
                square.setAcceptDrops(True)
        
        # Initialize play mode drag state
        if not hasattr(self, 'play_mode_drag_legal_moves'):
            self.play_mode_drag_legal_moves = []
            self.play_mode_drag_highlighted_squares = []
    
    def update_move_list_display(self):
        """Update the move list text display"""
        move_text = self.pgn_manager.get_move_list_text()
        self.move_list_text.setText(move_text)
        
        # Scroll to bottom to show latest moves
        scrollbar = self.move_list_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_copy_pgn(self):
        """Copy PGN to clipboard"""
        pgn_text = self.pgn_manager.get_pgn_text()
        if pgn_text:
            QApplication.clipboard().setText(pgn_text)
            QMessageBox.information(self, "PGN Copied", "PGN text copied to clipboard")
        else:
            QMessageBox.information(self, "No Moves", "No moves to copy")
    
    def handle_play_mode_click(self, row: int, col: int):
        """
        Handle square clicks in Play mode for legal moves.
        
        Args:
            row: Display row clicked
            col: Display column clicked
        """
        if not hasattr(self, 'selected_square'):
            self.selected_square = None
            self.legal_moves = []
            self.highlighted_squares = []
        
        # Clear any existing drag highlights
        self.clear_play_mode_drag_highlights()
        
        # Convert display coords to algebraic notation
        if self.board_model.is_display_flipped:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row
        
        clicked_square = chess.square(file_idx, rank_idx)
        internal_board = self.board_model.get_internal_board()
        
        # If no piece selected
        if self.selected_square is None:
            piece = internal_board.piece_at(clicked_square)
            if piece and piece.color == internal_board.turn:
                # Select this piece
                self.selected_square = clicked_square
                
                # Clear previous highlights
                for sq_coords in self.highlighted_squares:
                    self.squares[sq_coords[0]][sq_coords[1]].set_highlight(False)
                self.highlighted_squares.clear()
                
                # Find and highlight legal moves
                self.legal_moves = [move for move in internal_board.legal_moves 
                                  if move.from_square == clicked_square]
                
                for move in self.legal_moves:
                    # Convert to display coords
                    to_file = chess.square_file(move.to_square)
                    to_rank = chess.square_rank(move.to_square)
                    
                    if self.board_model.is_display_flipped:
                        disp_row = to_rank
                        disp_col = 7 - to_file
                    else:
                        disp_row = 7 - to_rank
                        disp_col = to_file
                    
                    self.squares[disp_row][disp_col].set_highlight(True)
                    self.highlighted_squares.append((disp_row, disp_col))
        
        else:
            # Clear highlights
            for sq_coords in self.highlighted_squares:
                self.squares[sq_coords[0]][sq_coords[1]].set_highlight(False)
            self.highlighted_squares.clear()
            
            # Check if this is a legal move
            move = None
            for legal_move in self.legal_moves:
                if legal_move.to_square == clicked_square:
                    # Check for promotion
                    if (internal_board.piece_at(self.selected_square).piece_type == chess.PAWN and
                        chess.square_rank(clicked_square) in [0, 7]):
                        # For simplicity, auto-promote to queen
                        if legal_move.promotion == chess.QUEEN:
                            move = legal_move
                            break
                    else:
                        move = legal_move
                        break
            
            if move:
                # Make the move
                san = internal_board.san(move)
                internal_board.push(move)
                
                # Record in PGN manager
                self.pgn_manager.add_move(move, san)
                
                # Update display
                self.sync_model_to_squares()
                self.refresh_ui_from_model()
                
                # Update undo button (use play mode button if in play state)
                if self.state_controller.is_play_mode:
                    self.play_undo_btn.setEnabled(True)
                else:
                    self.undo_btn.setEnabled(True)
            
            # Clear selection
            self.selected_square = None
            self.legal_moves = []

    def start_play_mode_drag(self, from_row: int, from_col: int) -> bool:
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
        self.clear_play_mode_drag_highlights()
        
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
            
            self.squares[disp_row][disp_col].set_highlight(True)
            self.play_mode_drag_highlighted_squares.append((disp_row, disp_col))
        
        return True
    
    def clear_play_mode_drag_highlights(self):
        """Clear all play mode drag highlights"""
        if hasattr(self, 'play_mode_drag_highlighted_squares'):
            for sq_coords in self.play_mode_drag_highlighted_squares:
                self.squares[sq_coords[0]][sq_coords[1]].set_highlight(False)
            self.play_mode_drag_highlighted_squares.clear()
        
        if hasattr(self, 'play_mode_drag_legal_moves'):
            self.play_mode_drag_legal_moves.clear()
    
    def is_play_mode_move_legal(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """
        Check if a move is legal in Play mode and return the move object.
        
        Args:
            from_row, from_col: Display coordinates of source square
            to_row, to_col: Display coordinates of destination square
            
        Returns:
            chess.Move object if legal, None otherwise
        """
        if not hasattr(self, 'play_mode_drag_legal_moves'):
            return None
        
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