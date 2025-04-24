import os
import sys
import chess
import chess.engine
import copy
from BoardSquareWidget import BoardSquareWidget
from labels import (PIECE_LABELS, labels_to_fen, get_piece_pixmap)
from PyQt5.QtCore import Qt, QEvent, QSettings
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QDialog, QGridLayout, QMessageBox, QHBoxLayout,
    QCheckBox, QRadioButton, QButtonGroup, QTextEdit, QWhatsThis, QAction, QMenu, QFileDialog, QSizePolicy
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
    def __init__(self, labels_2d, predicted_is_flipped=False, predicted_side_to_move='w'):
        super().__init__()
        self.setWindowTitle("Board Editor")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint | Qt.WindowContextHelpButtonHint)
        
        # Settings storage for user preferences
        self.settings = QSettings("ChessAIScanner", "Settings")
        
        # Initialize piece memory feature
        self.remembered_piece = None  # Currently remembered piece
        self.piece_cursor = None  # Custom cursor for remembered piece
        
        # Initialize undo/redo history
        self.history = []  # List to store board states
        self.current_index = -1  # Current position in history
        
        # Disable automatic "What's this?" on right-click
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        # Create a help action for direct access via the help button
        self.help_action = QAction("Help", self)
        self.help_action.triggered.connect(self.show_help)
        self.addAction(self.help_action)

        # Define help text in just one place
        self.help_text = """<b>Chess Board Editor - quick help</b><br>
• Drag pieces from the palette onto the board.<br>
• Right-click a square to erase it.<br>
• Double-click a piece in the palette to select it for repeated placement.<br>
• <i>Flip Board</i> changes the point of view.<br>
• <i>Copy FEN</i> copies the current position to the clipboard.<br>
• <i>Analysis</i> runs engine on the position.<br>
• Click the ▼ button next to Analysis to select different chess engines.<br>
• <i>Undo</i> restores the previous board position.<br>
• <i>Redo</i> restores a previously undone position.<br>
• Use the castling checkboxes to control O-O and O-O-O rights.<br>
- They're only enabled when the king and rook are on their starting squares.<br>
• To enable <i>en passant</i>, tick the box when a capture is available.<br>
- Then click the highlighted square to select the target.<br>
• The board orientation and side to move are automatically predicted.<br>"""

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

        # internal board state
        self.labels_2d    = [row[:] for row in labels_2d]   # deep‑copy
        self.is_flipped   = False   # board orientation flag
        self.coords_switched = False # coordinate labels flipped?

        self.file_labels = list("abcdefgh")   # default from White side
        self.rank_labels = list("87654321")   # "

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
        self.ep_cb = QCheckBox("en passant")
        self.ep_cb.setEnabled(False)
        self.ep_cb.stateChanged.connect(self.on_ep_toggled)
        top_row.addWidget(self.ep_cb)
        self.ep_possible   = {}   # {(r,c): "e3", ...}
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
        for r in range(8):
            # rank label
            rank_lbl = QLabel(self.rank_labels[r]); rank_lbl.setAlignment(Qt.AlignCenter)
            self.grid.addWidget(rank_lbl, r+1, 0)
            row_widgets = []
            for c in range(8):
                sq = BoardSquareWidget(r, c, self.labels_2d[r][c], parent=self)
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
        self.undo_btn = QPushButton("Undo"); self.undo_btn.clicked.connect(self.on_undo)
        self.redo_btn = QPushButton("Redo"); self.redo_btn.clicked.connect(self.on_redo)
        
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
            b = QPushButton(txt); b.clicked.connect(slot)
            if icon_path: b.setIcon(QIcon(icon_path))
            btn_bar.addWidget(b); return b

        self.clear_btn  = add_btn("Clear Board",  None,        self.on_clear_board)
        self.flip_btn   = add_btn("Flip Board",   flip_path,   self.on_flip_board)
        self.switch_btn = add_btn("Switch",       switch_path, self.on_switch_coords)
        
        self.reset_btn  = add_btn("Reset",        None,        self.on_reset_to_start)
        self.copy_btn   = add_btn("Copy FEN",     clip_path,   self.on_copy_fen)
        self.paste_btn  = add_btn("Paste FEN",    paste_path,  self.on_paste_fen)
        self.learn_btn  = add_btn("Learn",        nn_path,     self.accept)
        btn_bar.addStretch()

        # analysis column
        right = QVBoxLayout(); outer.addLayout(right)
        
        # Create analysis button group with dropdown
        analysis_container = QWidget()
        analysis_layout = QHBoxLayout(analysis_container)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(0)
        
        # Main Analysis button
        self.analysis_btn = QPushButton("Analysis")
        self.analysis_btn.setIcon(QIcon(cengine_path))
        self.analysis_btn.clicked.connect(self.on_analysis)
        
        # Dropdown button for engine selection
        self.engine_dropdown_btn = QPushButton("▼")
        self.engine_dropdown_btn.setMaximumWidth(20)  # Make dropdown button narrow
        
        # Ensure the dropdown button has the same height and styling as the Analysis button
        self.engine_dropdown_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.engine_dropdown_btn.setFixedHeight(self.analysis_btn.sizeHint().height())
        self.engine_dropdown_btn.setStyleSheet("QPushButton { padding: 0px; }")
        
        self.engine_dropdown_btn.clicked.connect(self.show_engine_menu)
        
        # Add buttons to container
        analysis_layout.addWidget(self.analysis_btn, 1)  # Main button takes most space
        analysis_layout.addWidget(self.engine_dropdown_btn, 0)  # Dropdown is small
        
        # Add container to layout
        right.addWidget(analysis_container)
        
        # Variable to store the selected engine
        self.selected_engine = None
        
        # Load last used engine from settings
        last_engine = self.settings.value("last_used_engine", None)
        if last_engine:
            engine_dir = os.path.join(os.path.dirname(__file__), "engine")
            engine_path = os.path.join(engine_dir, last_engine)
            if os.path.exists(engine_path):
                self.selected_engine = last_engine
                # Update the UI to show the loaded engine
                if os.path.sep in last_engine:
                    display_name = os.path.basename(last_engine)
                    display_name = os.path.splitext(display_name)[0]
                else:
                    display_name = os.path.splitext(last_engine)[0]
                self.analysis_btn.setText(f"Analysis ({display_name})")
        
        self.analysis_view = QTextEdit(); self.analysis_view.setReadOnly(True)
        self.analysis_view.setPlaceholderText("Engine lines will appear here")
        self.analysis_view.setFixedWidth(280)
        right.addWidget(self.analysis_view, 1)
        right.addStretch()

        # Auto-apply predicted orientation
        if predicted_is_flipped:
            self.on_switch_coords()  # Apply predicted board orientation
        
        # initialise UI‑dependent states
        self.refresh_castling_checkboxes()
        
        # Save initial state to history
        self.save_state()
        
        # Use the same help text for What's This instead of duplicating it
        self.setWhatsThis(self.help_text)

    # just some helpers
    def sync_squares_to_labels(self):
        for r in range(8):
            for c in range(8):
                self.labels_2d[r][c] = self.squares[r][c].piece_label
        self.refresh_castling_checkboxes()

    def save_state(self):
        """Save current board state to history"""
        self.sync_squares_to_labels()
        
        # Create a state object with just the board position
        state = {
            'labels_2d': copy.deepcopy(self.labels_2d)
        }
        
        # If we're not at the end of the history, truncate it
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add new state to history
        self.history.append(state)
        self.current_index = len(self.history) - 1
        
        # Update button states
        self.update_undo_redo_buttons()

    def restore_state(self, state):
        """Restore board to a saved state"""
        # Restore board position only
        self.labels_2d = copy.deepcopy(state['labels_2d'])
        for r in range(8):
            for c in range(8):
                self.squares[r][c].piece_label = self.labels_2d[r][c]
                self.squares[r][c].update()
        
        # Refresh UI
        self.refresh_castling_checkboxes()
        
        # Update button states
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons"""
        self.undo_btn.setEnabled(self.current_index > 0)
        self.redo_btn.setEnabled(self.current_index < len(self.history) - 1)

    def on_undo(self):
        """Restore the previous board state"""
        if self.current_index > 0:
            self.current_index -= 1
            self.restore_state(self.history[self.current_index])

    def on_redo(self):
        """Restore the next board state"""
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.restore_state(self.history[self.current_index])

    def on_clear_board(self):
        for r in range(8):
            for c in range(8):
                self.squares[r][c].piece_label = "empty"
                self.squares[r][c].update()
        self.refresh_castling_checkboxes()
        self.save_state()

    def on_flip_board(self):
        self.labels_2d.reverse(); [row.reverse() for row in self.labels_2d]
        for r in range(8):
            for c in range(8):
                self.squares[r][c].piece_label = self.labels_2d[r][c]
                self.squares[r][c].update()
        self.file_labels.reverse(); self.rank_labels.reverse()
        for r in range(8):
            self.grid.itemAtPosition(r+1, 0).widget().setText(self.rank_labels[r])
        for c in range(8):
            self.grid.itemAtPosition(9, c+1).widget().setText(self.file_labels[c])
        self.is_flipped = not self.is_flipped
        self.save_state()

    def on_switch_coords(self):
        self.coords_switched = not self.coords_switched
        self.file_labels.reverse(); self.rank_labels.reverse()
        for r in range(8):
            self.grid.itemAtPosition(r+1, 0).widget().setText(self.rank_labels[r])
        for c in range(8):
            self.grid.itemAtPosition(9, c+1).widget().setText(self.file_labels[c])
        self.refresh_castling_checkboxes()
        self.save_state()

    def on_reset_to_start(self):
        standard = [
            ["br","bn","bb","bq","bk","bb","bn","br"],
            ["bp"]*8,
            ["empty"]*8,
            ["empty"]*8,
            ["empty"]*8,
            ["empty"]*8,
            ["wp"]*8,
            ["wr","wn","wb","wq","wk","wb","wn","wr"]
        ]
        if self.is_flipped:
            std = [row[::-1] for row in standard[::-1]]
        else:
            std = [row[:] for row in standard]
        self.labels_2d = [row[:] for row in std]
        for r in range(8):
            for c in range(8):
                self.squares[r][c].piece_label = self.labels_2d[r][c]
                self.squares[r][c].update()
        self.refresh_castling_checkboxes()
        self.save_state()

    def on_copy_fen(self):
        self.sync_squares_to_labels()
        effective = [row[:] for row in self.labels_2d]
        if self.rank_labels[0] == '1': effective.reverse()
        if self.file_labels[0] != 'a': [row.reverse() for row in effective]
        fen = labels_to_fen(effective, self.get_side_to_move(), self.get_castling_rights(), self.get_ep_field())
        QApplication.clipboard().setText(fen)
        QMessageBox.information(self, "FEN copied", fen)
        
    def on_paste_fen(self):
        """Parse a FEN string from clipboard and update the board"""
        clipboard = QApplication.clipboard()
        fen_text = clipboard.text().strip()
        
        try:
            # Attempt to create a chess board from the FEN to validate it
            chess.Board(fen_text)  # Just validate, don't store
            
            # Extract the piece placement part of FEN (first part before the first space)
            placement = fen_text.split(' ')[0]
            ranks = placement.split('/')
            
            if len(ranks) != 8:
                raise ValueError("Invalid FEN: must have 8 ranks")
            
            # Extract side to move (second part after first space)
            parts = fen_text.split(' ')
            if len(parts) >= 2:
                side_to_move = parts[1]
                if side_to_move == 'w':
                    self.white_rb.setChecked(True)
                    self.black_rb.setChecked(False)
                elif side_to_move == 'b':
                    self.white_rb.setChecked(False)
                    self.black_rb.setChecked(True)
            
            # Extract castling rights (third part after second space)
            if len(parts) >= 3:
                castling = parts[2]
                self.w_k_cb.setChecked('K' in castling)
                self.w_q_cb.setChecked('Q' in castling)
                self.b_k_cb.setChecked('k' in castling)
                self.b_q_cb.setChecked('q' in castling)
            
            # Convert to our internal format
            new_labels = []
            for rank in ranks:
                row = []
                for char in rank:
                    if char.isdigit():
                        # Add empty squares
                        row.extend(["empty"] * int(char))
                    else:
                        # Convert piece character to our label format
                        piece_color = 'w' if char.isupper() else 'b'
                        piece_type = char.lower()
                        if piece_type == 'p':
                            row.append(f"{piece_color}p")
                        elif piece_type == 'n':
                            row.append(f"{piece_color}n")
                        elif piece_type == 'b':
                            row.append(f"{piece_color}b")
                        elif piece_type == 'r':
                            row.append(f"{piece_color}r")
                        elif piece_type == 'q':
                            row.append(f"{piece_color}q")
                        elif piece_type == 'k':
                            row.append(f"{piece_color}k")
                        else:
                            row.append("empty")  # Fallback
                new_labels.append(row)
            
            # Handle board orientation - get the board in display format
            if self.is_flipped:
                new_labels.reverse()
                for row in new_labels:
                    row.reverse()
            
            if self.coords_switched:
                # If coordinates are switched, we need to reverse rows but not flip the board
                new_labels.reverse()
                for row in new_labels:
                    row.reverse()
            
            # Update our labels and squares
            if all(len(row) == 8 for row in new_labels):
                self.labels_2d = new_labels
                for r in range(8):
                    for c in range(8):
                        self.squares[r][c].piece_label = self.labels_2d[r][c]
                        self.squares[r][c].update()
                self.refresh_castling_checkboxes()
                self.save_state()
                QMessageBox.information(self, "FEN Loaded", "FEN position loaded successfully")
            else:
                QMessageBox.warning(self, "Invalid FEN", "FEN has incorrect number of squares")
        
        except Exception as e:
            QMessageBox.warning(self, "Invalid FEN", f"Could not parse FEN: {str(e)}")

    def show_engine_menu(self):
        """Display a popup menu with available UCI engines"""
        # Find all engines in the engine folder
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        engine_menu = QMenu(self)
        
        try:
            if os.path.exists(engine_dir):
                engines = []
                
                # Find executables on Windows (recursively search in subdirectories)
                if sys.platform.startswith("win"):
                    for root, dirs, files in os.walk(engine_dir):
                        for file in files:
                            if file.lower().endswith(".exe"):
                                # Get path relative to engine directory for display
                                rel_path = os.path.relpath(os.path.join(root, file), engine_dir)
                                engines.append(rel_path)
                # Find executables on macOS/Linux (recursively)
                else:
                    for root, dirs, files in os.walk(engine_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path) and os.access(file_path, os.X_OK):
                                rel_path = os.path.relpath(file_path, engine_dir)
                                engines.append(rel_path)
                
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
            else:
                # Engine directory doesn't exist
                no_dir_action = engine_menu.addAction("Engine directory not found")
                no_dir_action.setEnabled(False)
                
                # Add option to create directory and add engine
                engine_menu.addSeparator()
                create_dir_action = engine_menu.addAction("Create engine directory")
                create_dir_action.triggered.connect(self.create_engine_directory)
        except Exception as e:
            # Error occurred
            error_action = engine_menu.addAction(f"Error: {str(e)}")
            error_action.setEnabled(False)
        
        # Show the menu
        pos = self.engine_dropdown_btn.mapToGlobal(self.engine_dropdown_btn.rect().bottomLeft())
        engine_menu.exec_(pos)
    
    def select_engine(self, engine_name):
        """Set the selected engine"""
        # Store the engine name
        self.selected_engine = engine_name
        
        # Save engine selection to settings
        self.settings.setValue("last_used_engine", engine_name)
        
        # Update the analysis button text to show selected engine
        # For engines in subdirectories, just show the filename without extension
        if os.path.sep in engine_name:
            # For nested engines, just show the filename
            display_name = os.path.basename(engine_name)
            display_name = os.path.splitext(display_name)[0]
        else:
            # For top level engines
            display_name = os.path.splitext(engine_name)[0]
            
        self.analysis_btn.setText(f"Analysis ({display_name})")
        
        # Test if the engine can be initialized properly
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        engine_path = os.path.join(engine_dir, engine_name)
        
        if not self.test_engine_compatibility(engine_path):
            # If there's a problem, show a warning but still allow selection
            QMessageBox.warning(
                self, 
                "Engine Warning", 
                f"The engine '{display_name}' may not be compatible or may have issues.\n"
                "It will remain selected, but may fail during analysis."
            )
            
    def test_engine_compatibility(self, engine_path):
        """Test if an engine can be properly initialized"""
        try:
            # Attempt to start the engine with a short timeout
            engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=2.0)
            # If successful, close it properly
            engine.quit()
            return True
        except Exception as e:
            print(f"Engine compatibility test failed: {str(e)}")
            return False

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
        """
        Runs UCI engine at depth 20 and shows the top-3 PVs.
        """
        # Display "Analyzing..." in the analysis view to show activity
        self.analysis_view.setPlainText("Analyzing...")
        
        # find any engine in ./engine
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        try:
            engine_name = None
            missing_engine = None
            
            # First try to use the selected engine
            if self.selected_engine:
                engine_path = os.path.join(engine_dir, self.selected_engine)
                if os.path.exists(engine_path):
                    engine_name = self.selected_engine
                else:
                    # Remember the missing engine for a more informative message
                    missing_engine = self.selected_engine
                    # Reset the selected engine since it's missing
                    self.selected_engine = None
                    self.analysis_btn.setText("Analysis")
                    # Also clear from settings so we don't try to load it again
                    self.settings.remove("last_used_engine")
            
            # If no engine is currently selected, try to load last used engine from settings
            if not engine_name:
                last_engine = self.settings.value("last_used_engine", None)
                if last_engine:
                    last_engine_path = os.path.join(engine_dir, last_engine)
                    if os.path.exists(last_engine_path):
                        engine_name = last_engine
                        self.selected_engine = last_engine
                        # Update the UI to show the loaded engine
                        if os.path.sep in last_engine:
                            display_name = os.path.basename(last_engine)
                            display_name = os.path.splitext(display_name)[0]
                        else:
                            display_name = os.path.splitext(last_engine)[0]
                        self.analysis_btn.setText(f"Analysis ({display_name})")
                    else:
                        # Remember the missing engine for a more informative message
                        missing_engine = last_engine
                        # Clear from settings so we don't try to load it again
                        self.settings.remove("last_used_engine")
            
            # If still no engine, search recursively for any available engine
            if not engine_name:
                # Search recursively in all subdirectories
                for root, dirs, files in os.walk(engine_dir):
                    if engine_name:
                        break  # Stop once we found an engine
                        
                    if sys.platform.startswith("win"):
                        # Look for .exe files on Windows
                        for file in files:
                            if file.lower().endswith(".exe"):
                                # Get path relative to engine directory
                                engine_name = os.path.relpath(os.path.join(root, file), engine_dir)
                                break
                    else:
                        # Look for executable files on Unix
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path) and os.access(file_path, os.X_OK):
                                engine_name = os.path.relpath(file_path, engine_dir)
                                break
                
                if engine_name:
                    # Update the selected engine
                    self.select_engine(engine_name)
                else:
                    error_message = f"No UCI engine executable found in:\n  {engine_dir} or its subdirectories\n\n"
                    if missing_engine:
                        # Add information about the missing engine
                        display_name = os.path.basename(missing_engine) if os.path.sep in missing_engine else missing_engine
                        error_message = f"Previously selected engine '{display_name}' was not found.\n\n{error_message}"
                    
                    error_message += "Please add a chess engine (e.g. stockfish.exe) using the dropdown menu."
                    
                    QMessageBox.critical(self, "Engine not found", error_message)
                    self.analysis_view.setPlainText("No engine available")
                    return
                    
        except (StopIteration, FileNotFoundError):
            QMessageBox.critical(
                self, "Engine not found",
                f"No UCI engine executable found in:\n  {engine_dir} or its subdirectories\n\n"
                "Please add a chess engine (e.g. stockfish.exe) using the dropdown menu."
            )
            self.analysis_view.setPlainText("No engine available")
            return
        
        ANALYSIS_ENGINE_PATH = os.path.join(engine_dir, engine_name)
        
        # build a FEN that matches the user's coordinate view
        self.sync_squares_to_labels()
        board_copy = [row[:] for row in self.labels_2d]
        if self.rank_labels[0] == '1':
            board_copy.reverse()
        if self.file_labels[0] != 'a':
            for row in board_copy:
                row.reverse()

        fen = labels_to_fen(
            board_copy,
            self.get_side_to_move(),
            self.get_castling_rights(),
            self.get_ep_field()
        )

        # fire up the engine
        try:
            # Show a busy cursor during analysis
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Use a longer timeout for potentially slow engines
            engine = chess.engine.SimpleEngine.popen_uci(ANALYSIS_ENGINE_PATH, timeout=10.0)
            
            board = chess.Board(fen)

            # Set a maximum time limit to prevent hanging on problematic engines
            try:
                info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=20, time=10.0),  # Add time limit of 10 seconds
                    multipv=3
                )
                
                # build a nice text block
                lines = []
                for i, pv in enumerate(info, 1):
                    score = pv["score"].white()  # always from white's viewpoint
                    if score.is_mate():
                        score_str = f"# {score.mate()}"
                    else:
                        score_str = f"{score.score()/100:.2f}"
                    san_line = board.variation_san(pv["pv"])
                    lines.append(f"{i}.  {score_str}  |  {san_line}")

                self.analysis_view.setPlainText("\n".join(lines))
                
            except chess.engine.EngineError as e:
                QMessageBox.critical(self, "Engine analysis error", str(e))
                self.analysis_view.setPlainText(f"Engine error: {str(e)}\n\nTry a different engine.")
                
            # Always make sure to quit the engine
            try:
                engine.quit()
            except Exception:
                # If engine already crashed, we may not be able to quit properly
                pass
                
        except chess.engine.EngineTerminatedError as e:
            QMessageBox.critical(
                self, "Engine crash",
                f"The engine '{os.path.splitext(engine_name)[0]}' crashed unexpectedly.\n\n"
                f"Error: {str(e)}\n\n"
                "Try selecting a different engine."
            )
            self.analysis_view.setPlainText(f"Engine crashed. Try a different engine.")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Engine error",
                f"Error running engine: {str(e)}"
            )
            self.analysis_view.setPlainText(f"Error: {str(e)}")
            
        finally:
            # Always restore cursor
            QApplication.restoreOverrideCursor()
            
        self.analysis_view.moveCursor(self.analysis_view.textCursor().Start)

    def get_final_labels_2d(self):
        self.sync_squares_to_labels()
        return self.labels_2d
    
    def get_side_to_move(self):
        return 'w' if self.white_rb.isChecked() else 'b'
    def get_castling_rights(self) -> str:
        rights = ""
        if self.w_k_cb.isChecked(): rights += "K"
        if self.w_q_cb.isChecked(): rights += "Q"
        if self.b_k_cb.isChecked(): rights += "k"
        if self.b_q_cb.isChecked(): rights += "q"
        return rights or "-"           # FEN uses "-" when no rights remain
    
    def refresh_castling_checkboxes(self):
        """Enable box when K+R are on their start squares, else grey-out+untick."""
        # get the board in *standard orientation* (rank‑8 first, file‑a first)
        board = [row[:] for row in self.labels_2d]
        if self.rank_labels[0] == '1':           # ranks are upside‑down
            board.reverse()
        if self.file_labels[0] != 'a':           # files are flipped
            for r in board:
                r.reverse()

        # quick tests: is king + rook on their initial squares?
        WK = board[7][4] == "wk" and board[7][7] == "wr"     # e1 + h1
        WQ = board[7][4] == "wk" and board[7][0] == "wr"     # e1 + a1
        BK = board[0][4] == "bk" and board[0][7] == "br"     # e8 + h8
        BQ = board[0][4] == "bk" and board[0][0] == "br"     # e8 + a8

        for cb, ok in ((self.w_k_cb, WK), (self.w_q_cb, WQ),
                    (self.b_k_cb, BK), (self.b_q_cb, BQ)):
            cb.blockSignals(True)        # avoid spurious clicked()
            cb.setEnabled(ok)            # grey‑out if not possible
            if not ok:                   # impossible → untick
                cb.setChecked(False)
            else:                        # If enabled, automatically check it
                cb.setChecked(True)      
            cb.blockSignals(False)
        self.refresh_en_passant()

    def compute_ep_candidates(self):
        """
        Identify squares where en passant captures might be possible.
        """
        self.ep_possible.clear()
        side = self.get_side_to_move()  # 'w' or 'b'

        # board in standard orientation
        std = [row[:] for row in self.labels_2d]
        if self.rank_labels[0] == '1': std.reverse()
        if self.file_labels[0] != 'a': [row.reverse() for row in std]

        def alg(r, c): return "abcdefgh"[c] + str(8 - r)

        # In chess, en passant is only possible on the 3rd rank (for white) or 6th rank (for black)
        if side == 'w':  # White to move
            # Check for possible en passant targets on the 6th rank (3rd rank from white's perspective)
            # This is where a black pawn might have just moved two squares forward
            enpassant_rank = 2  # Target rank in the array (6th rank)
            pawn_rank = 3      # Where the black pawn would be
            
            for col in range(8):
                # Check if there's a black pawn on the 5th rank
                if std[pawn_rank][col] == "bp":
                    # Check if there are white pawns on either side that could capture
                    for dcol in [-1, 1]:
                        if 0 <= col + dcol < 8 and std[pawn_rank][col + dcol] == "wp":
                            # Found a potential en passant situation
                            # The en passant target is on the 6th rank, same file as the black pawn
                            trg_r, trg_c = enpassant_rank, col
                            # Convert to display coordinates
                            disp_r = 7 - trg_r if self.is_flipped else trg_r
                            disp_c = 7 - trg_c if self.is_flipped else trg_c
                            self.ep_possible[(disp_r, disp_c)] = alg(trg_r, trg_c)
                            break  # Only need to find one capturing pawn
                            
        else:  # Black to move
            # Check for possible en passant targets on the 3rd rank (6th rank from black's perspective)
            # This is where a white pawn might have just moved two squares forward
            enpassant_rank = 5  # Target rank in the array (3rd rank)
            pawn_rank = 4      # Where the white pawn would be
            
            for col in range(8):
                # Check if there's a white pawn on the 4th rank
                if std[pawn_rank][col] == "wp":
                    # Check if there are black pawns on either side that could capture
                    for dcol in [-1, 1]:
                        if 0 <= col + dcol < 8 and std[pawn_rank][col + dcol] == "bp":
                            # Found a potential en passant situation
                            # The en passant target is on the 3rd rank, same file as the white pawn
                            trg_r, trg_c = enpassant_rank, col
                            # Convert to display coordinates
                            disp_r = 7 - trg_r if self.is_flipped else trg_r
                            disp_c = 7 - trg_c if self.is_flipped else trg_c
                            self.ep_possible[(disp_r, disp_c)] = alg(trg_r, trg_c)
                            break  # Only need to find one capturing pawn

    def refresh_en_passant(self):
        """Re-evaluate EP candidates; disable if none or if in check."""
        self.compute_ep_candidates()
        ok = bool(self.ep_possible)

        std = [row[:] for row in self.labels_2d]
        if self.rank_labels[0] == '1':      # ranks flipped in UI?
            std.reverse()
        if self.file_labels[0] != 'a':      # files flipped?
            for row in std:
                row.reverse()

        # force ep field = '-' and check for check()
        fen = labels_to_fen(
            std,
            self.get_side_to_move(),
            self.get_castling_rights(),
            "-"   # ignore any selected ep square
        )
        board = chess.Board(fen)
        if board.is_check():
            ok = False

        self.ep_cb.blockSignals(True)
        self.ep_cb.setEnabled(ok)
        if not ok:
            self.ep_cb.setChecked(False)
            self.ep_selected = None
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
    
    def place_remembered_piece(self, row, col):
        """Place the remembered piece on the board at the specified position"""
        if self.remembered_piece and 0 <= row < 8 and 0 <= col < 8:
            # Update the square with the remembered piece
            self.squares[row][col].piece_label = self.remembered_piece
            self.squares[row][col].update()
            self.sync_squares_to_labels()
            self.save_state()  # Save state after placing a piece
            return True
        return False
        
    def clear_remembered_piece(self):
        """Clear the currently remembered piece"""
        self.remembered_piece = None
        if QApplication.overrideCursor():
            QApplication.restoreOverrideCursor()
            
    def mousePressEvent(self, event):
        """Clear remembered piece if clicking outside of board squares"""
        if self.remembered_piece:
            # We'll keep the piece remembered unless the click was on a non-board area
            # The individual BoardSquareWidgets will handle their own clicks
            if event.button() == Qt.LeftButton:
                # Check if we clicked on something that isn't a board square
                widget = self.childAt(event.pos())
                if not widget or not isinstance(widget, BoardSquareWidget):
                    self.clear_remembered_piece()
        super().mousePressEvent(event)