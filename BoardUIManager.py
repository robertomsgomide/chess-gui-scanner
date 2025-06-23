import os
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QCheckBox, QRadioButton, QButtonGroup, QWidget, QSizePolicy,
    QTextEdit, QScrollArea, QAction
)
from BoardSquareWidget import BoardSquareWidget
from labels import PIECE_LABELS

class BoardUIManager:
    """
    Manages the creation and layout of all UI widgets for the Board Editor.
    This class is responsible for constructing the UI but not for handling events
    or managing state transitions.
    """
    
    def __init__(self, parent_dialog, predicted_side_to_move='w'):
        """
        Initialize the UI manager and create all widgets.
        
        Args:
            parent_dialog: The parent QDialog (BoardEditor)
            predicted_side_to_move: Initial side to move ('w' or 'b')
        """
        self.parent = parent_dialog
        self.settings = QSettings("ChessAIScanner", "Settings")
        
        # Icons as QIcon objects
        base_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
        self.icons = {
            'window': QIcon(os.path.join(base_icon, "Chess_icon.png")),
            'white': QIcon(os.path.join(base_icon, "Chess_klt.png")),
            'black': QIcon(os.path.join(base_icon, "Chess_kdt.png")),
            'switch': QIcon(os.path.join(base_icon, "Chess_switch.png")),
            'flip': QIcon(os.path.join(base_icon, "Chess_flip.png")),
            'clip': QIcon(os.path.join(base_icon, "Chess_clip.png")),
            'nn': QIcon(os.path.join(base_icon, "Chess_nn.png")),
            'paste': QIcon(os.path.join(base_icon, "Chess_paste.png")),
            'bcastle': QIcon(os.path.join(base_icon, "Chess_bcastle.png")),
            'wcastle': QIcon(os.path.join(base_icon, "Chess_wcastle.png")),
            'engine': QIcon(os.path.join(base_icon, "Chess_engine.png")),
            'undo': QIcon(os.path.join(base_icon, "Chess_undo.png")),
            'redo': QIcon(os.path.join(base_icon, "Chess_redo.png")),
            'opening': QIcon(os.path.join(base_icon, "Chess_opening.png")),
            'done': QIcon(os.path.join(base_icon, "Chess_done.png")),
            'empty': QIcon()  # Empty icon for clearing button icons
        }
        
        # Coordinate labels
        self.file_labels = list("abcdefgh")
        self.rank_labels = list("87654321")
        
        # Create the main layout
        self._create_main_layout(predicted_side_to_move)
        
    def _create_main_layout(self, predicted_side_to_move):
        """Create the main layout structure"""
        # Main horizontal layout
        self.main_layout = QHBoxLayout()
        
        # Left side (board + controls)
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)
        
        # Create all UI sections
        self._create_top_controls(predicted_side_to_move)
        self._create_board_grid()
        self._create_palette()
        self._create_bottom_buttons()
        
        # Right side (analysis/move list)
        self._create_right_panel()
        
    def _create_top_controls(self, predicted_side_to_move):
        """Create top control row with side-to-move and castling options"""
        top_row = QHBoxLayout()
        self.left_layout.addLayout(top_row)
        
        # Side to move radio buttons
        self.white_rb = QRadioButton()
        self.white_rb.setIcon(self.icons['white'])
        self.black_rb = QRadioButton()
        self.black_rb.setIcon(self.icons['black'])
        self.white_rb.setChecked(predicted_side_to_move == 'w')
        self.black_rb.setChecked(predicted_side_to_move == 'b')
        
        self.side_group = QButtonGroup(self.parent)
        self.side_group.addButton(self.white_rb)
        self.side_group.addButton(self.black_rb)
        
        # En passant checkbox
        self.ep_cb = QCheckBox("en passant")
        self.ep_cb.setEnabled(False)
        top_row.addWidget(self.ep_cb)
        
        # Castling checkboxes
        self.w_k_cb = QCheckBox("\nO-O")
        self.w_k_cb.setIcon(self.icons['wcastle'])
        self.w_q_cb = QCheckBox("\nO-O-O")
        self.w_q_cb.setIcon(self.icons['wcastle'])
        self.b_k_cb = QCheckBox("\nO-O")
        self.b_k_cb.setIcon(self.icons['bcastle'])
        self.b_q_cb = QCheckBox("\nO-O-O")
        self.b_q_cb.setIcon(self.icons['bcastle'])
        
        for cb in (self.w_k_cb, self.w_q_cb, self.b_k_cb, self.b_q_cb):
            cb.setChecked(True)
        
        # Add all widgets to top row
        for widget in (self.white_rb, self.black_rb, self.w_k_cb, self.w_q_cb, 
                      self.b_k_cb, self.b_q_cb):
            top_row.addWidget(widget)
        top_row.addStretch()
        
    def _create_board_grid(self):
        """Create the 8x8 board grid with coordinate labels"""
        grid_container = QWidget()
        self.grid = QGridLayout(grid_container)
        self.left_layout.addWidget(grid_container)
        
        # Set fixed size and spacing for perfect alignment
        # Width: 60px rank labels + 8*60px board + 70px buttons = 610px
        # Height: 8*60px board + 60px file labels = 540px  
        grid_container.setFixedSize(9*60 + 70, 9*60)
        self.grid.setSpacing(0)
        self.grid.setContentsMargins(0, 0, 0, 0)
        grid_container.setStyleSheet("padding: 0px; margin: 0px; border: 0px; spacing: 0px;")
        
        # Disable automatic stretching for precise alignment
        self.grid.setVerticalSpacing(0)
        self.grid.setHorizontalSpacing(0)
        
        # Set size policies - keep board squares uniform but allow coordinate labels to be fixed
        # Board rows (1-8) should maintain equal size, coordinate rows (0,9) are fixed
        for i in range(1, 9):  # Board rows only
            self.grid.setRowStretch(i, 1)
        for i in range(1, 9):  # Board columns only  
            self.grid.setColumnStretch(i, 1)
        
        # Coordinate label rows and columns have no stretch (fixed size)
        self.grid.setRowStretch(0, 0)    # Rank labels row (actually not used)
        self.grid.setRowStretch(9, 0)    # File labels row
        self.grid.setColumnStretch(0, 0) # Rank labels column
        
        # Create board squares
        self.squares = []
        for r in range(8):
            # Rank label with fixed size to match squares
            rank_lbl = QLabel(self.rank_labels[r])
            rank_lbl.setAlignment(Qt.AlignCenter)
            rank_lbl.setFixedSize(60, 60)  # Match square size exactly
            rank_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.grid.addWidget(rank_lbl, r+1, 0)
            
            row_widgets = []
            for c in range(8):
                # Squares will be created by BoardEditor using the model
                row_widgets.append(None)
            self.squares.append(row_widgets)
        
        # File labels with fixed size to match squares
        for c in range(8):
            file_lbl = QLabel(self.file_labels[c])
            file_lbl.setAlignment(Qt.AlignCenter)
            file_lbl.setFixedSize(60, 60)  # Match square size exactly
            file_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.grid.addWidget(file_lbl, 9, c+1)
        
        # Blank corner with fixed size
        corner_lbl = QLabel("")
        corner_lbl.setFixedSize(60, 60)
        self.grid.addWidget(corner_lbl, 9, 0)
        
        # Undo/Redo buttons
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.undo_btn.setMinimumWidth(60)
        self.redo_btn.setMinimumWidth(60)
        self.undo_btn.setIcon(self.icons['undo'])
        self.redo_btn.setIcon(self.icons['redo'])
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        
        self.grid.addWidget(self.undo_btn, 1, 9)
        self.grid.addWidget(self.redo_btn, 2, 9)
        self.grid.setColumnStretch(9, 1)
        
    def _create_palette(self):
        """Create the piece palette"""
        self.palette_container = QWidget()
        palette_layout = QHBoxLayout(self.palette_container)
        palette_layout.setSpacing(0)
        palette_layout.setContentsMargins(0, 0, 0, 0)
        self.palette_container.setStyleSheet("padding: 0px; margin: 0px; border: 0px; spacing: 0px;")
        
        self.left_layout.addWidget(self.palette_container)
        
        # Palette squares will be created by BoardEditor
        self.palette_squares = []
        self.palette_layout = palette_layout
        
    def _create_bottom_buttons(self):
        """Create bottom button bar"""
        btn_bar = QHBoxLayout()
        self.left_layout.addLayout(btn_bar)
        
        # Create buttons
        self.clear_btn = QPushButton("Clear Board")
        self.flip_btn = QPushButton("Flip Board")
        self.flip_btn.setIcon(self.icons['flip'])
        self.switch_btn = QPushButton("Switch")
        self.switch_btn.setIcon(self.icons['switch'])
        self.reset_btn = QPushButton("Set To Opening")
        self.reset_btn.setIcon(self.icons['opening'])
        self.copy_btn = QPushButton("Copy FEN")
        self.copy_btn.setIcon(self.icons['clip'])
        self.paste_btn = QPushButton("Paste FEN")
        self.paste_btn.setIcon(self.icons['paste'])
        self.copy_pgn_btn = QPushButton("Copy PGN")
        self.copy_pgn_btn.setIcon(self.icons['clip'])
        self.learn_btn = QPushButton("Learn")
        self.learn_btn.setIcon(self.icons['nn'])
        
        # Add buttons to layout
        for btn in (self.clear_btn, self.flip_btn, self.switch_btn, self.reset_btn,
                   self.copy_btn, self.paste_btn, self.copy_pgn_btn, self.learn_btn):
            btn_bar.addWidget(btn)
        btn_bar.addStretch()
        
    def _create_right_panel(self):
        """Create right panel with analysis and move list"""
        self.right_layout = QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)
        
        # State toggle button
        self.state_toggle_btn = QPushButton("Finish Edit")
        self.state_toggle_btn.setIcon(self.icons['done'])
        self.right_layout.addWidget(self.state_toggle_btn)
        
        # Analysis controls
        self._create_analysis_controls()
        
        # Analysis display area
        self._create_analysis_display()
        
        # Move list panel
        self._create_move_list_panel()
        
        self.right_layout.addStretch()
        
    def _create_analysis_controls(self):
        """Create analysis button with dropdown"""
        analysis_container = QWidget()
        analysis_layout = QHBoxLayout(analysis_container)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(0)
        
        # Main analysis button
        self.analysis_btn = QPushButton("Analysis")
        self.analysis_btn.setIcon(self.icons['engine'])
        
        # Dropdown button
        self.engine_dropdown_btn = QPushButton("▼")
        self.engine_dropdown_btn.setMaximumWidth(20)
        self.engine_dropdown_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.engine_dropdown_btn.setFixedHeight(self.analysis_btn.sizeHint().height())
        self.engine_dropdown_btn.setStyleSheet("QPushButton { padding: 0px; }")
        
        analysis_layout.addWidget(self.analysis_btn, 1)
        analysis_layout.addWidget(self.engine_dropdown_btn, 0)
        
        self.right_layout.addWidget(analysis_container)
        
        # Reset analysis button
        self.reset_analysis_btn = QPushButton("Restore Analysis Board")
        self.reset_analysis_btn.setEnabled(False)
        self.right_layout.addWidget(self.reset_analysis_btn)
        
    def _create_analysis_display(self):
        """Create analysis lines display area"""
        self.analysis_view_container = QWidget()
        self.analysis_view_container.setFixedWidth(280)
        analysis_view_layout = QVBoxLayout(self.analysis_view_container)
        analysis_view_layout.setContentsMargins(5, 5, 5, 5)
        
        # Placeholder
        self.placeholder_label = QLabel("Engine lines will appear here")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        analysis_view_layout.addWidget(self.placeholder_label)
        
        # Analysis line widgets
        self.analysis_line_widgets = []
        for i in range(3):
            line_widget = QLabel()
            line_widget.setWordWrap(True)
            line_widget.setTextFormat(Qt.RichText)
            line_widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            line_widget.setMinimumHeight(60)
            line_widget.setStyleSheet("padding: 5px; border: 1px solid transparent;")
            line_widget.setFocusPolicy(Qt.StrongFocus)
            line_widget.hide()
            
            analysis_view_layout.addWidget(line_widget)
            self.analysis_line_widgets.append(line_widget)
        
        self.right_layout.addWidget(self.analysis_view_container, 1)
        
    def _create_move_list_panel(self):
        """Create move list panel for Play mode"""
        self.move_list_container = QWidget()
        move_list_layout = QVBoxLayout(self.move_list_container)
        move_list_layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        move_list_header = QLabel("Move List")
        move_list_header.setAlignment(Qt.AlignCenter)
        move_list_header.setStyleSheet("font-weight: bold;")
        move_list_layout.addWidget(move_list_header)
        
        # Move list text
        self.move_list_text = QTextEdit()
        self.move_list_text.setReadOnly(True)
        self.move_list_text.setFixedWidth(260)
        self.move_list_text.setMinimumHeight(150)
        move_list_layout.addWidget(self.move_list_text)
        
        # Play mode undo button
        self.play_undo_btn = QPushButton("Undo Last Move")
        self.play_undo_btn.setToolTip("Undo the last move played")
        move_list_layout.addWidget(self.play_undo_btn)
        
        self.right_layout.addWidget(self.move_list_container)
        self.move_list_container.hide()
        
    def update_coordinate_labels(self, file_labels, rank_labels):
        """Update the coordinate labels on the board"""
        self.file_labels = file_labels
        self.rank_labels = rank_labels
        
        # Update rank labels with proper styling
        for r in range(8):
            label = self.grid.itemAtPosition(r+1, 0).widget()
            label.setText(self.rank_labels[r])
            label.setFixedSize(60, 60)  # Ensure size stays consistent
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Update file labels with proper styling
        for c in range(8):
            label = self.grid.itemAtPosition(9, c+1).widget()
            label.setText(self.file_labels[c])
            label.setFixedSize(60, 60)  # Ensure size stays consistent
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 14px;")
            
    def set_board_square(self, row, col, square_widget):
        """Set a board square widget at the given position"""
        self.grid.addWidget(square_widget, row+1, col+1)
        self.squares[row][col] = square_widget
        
    def add_palette_square(self, square_widget):
        """Add a square widget to the palette"""
        self.palette_layout.addWidget(square_widget, 0, Qt.AlignLeft)
        self.palette_squares.append(square_widget)
        
    def finalize_palette(self):
        """Add stretch to palette after all squares are added"""
        self.palette_layout.addStretch() 