from labels import (get_piece_pixmap)
from PyQt5.QtCore import (Qt, QMimeData, QSize, QRect)
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import (QPainter, QColor, QPixmap, QDrag)


#########################################
# BoardSquareWidget
#########################################

LIGHT_COLOR = QColor(240,217,181)
DARK_COLOR  = QColor(181,136, 99)
HIGHLIGHT_COLOR = QColor(100, 255, 100, 120)  # Green highlight for hover
MEMORY_HIGHLIGHT_COLOR = QColor(100, 100, 255, 120)  # Blue highlight for selected piece
DRAG_HIGHLIGHT_COLOR = QColor(255, 255, 100, 120)  # Yellow highlight for drag hover

class BoardSquareWidget(QWidget):
    """
    A single board square or palette tile that:
    - draws a light/dark or gray background
    - shows the piece icon if not 'empty'
    - supports drag & drop
    - supports double-click for piece memory
    - highlights when pieces are dragged over
    """
    def __init__(self, row, col, piece_label="empty", parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.piece_label = piece_label
        self.setFixedSize(60,60)
        self.setContentsMargins(0, 0, 0, 0)  # Remove any internal margins
        self.setStyleSheet("padding: 0px; margin: 0px; border: 0px; border-width: 0px;")  # Ensure no CSS styling adds spacing
        self.setAcceptDrops(True)
        self.board_editor = parent  # might be None if used differently
        self.highlight = False  # For en-passant
        self.drag_highlight = False  # For drag operations
        self.hover_highlight = False  # For cursor hover with remembered piece
        self.is_palette = False  # Flag for palette pieces, set by BoardEditor
    
    def sizeHint(self):
        # Ensure we return exactly our fixed size with no extra margins
        return QSize(60, 60)
        
    def set_highlight(self, on: bool):
        self.highlight = on
        self.update()
        
    def set_drag_highlight(self, on: bool):
        self.drag_highlight = on
        self.update()
        
    def set_hover_highlight(self, on: bool):
        self.hover_highlight = on
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)  # Disable antialiasing for crisp edges
        
        # Draw the entire rect including borders to avoid gaps
        rect = self.rect()
        
        # Extend drawing area by 1 pixel on all sides to ensure squares touch
        extended_rect = QRect(rect.x()-1, rect.y()-1, rect.width()+2, rect.height()+2)
        
        # normal background
        if self.row >= 0:
            base = LIGHT_COLOR if (self.row + self.col) % 2 == 0 else DARK_COLOR
            p.fillRect(extended_rect, base)
        else:
            # Use the parent widget's background color for palette pieces
            if self.board_editor:
                p.fillRect(extended_rect, self.board_editor.palette().color(self.board_editor.backgroundRole()))
            else:
                p.fillRect(extended_rect, Qt.lightGray)

        # highlight overlay for en-passant
        if self.highlight:
            p.fillRect(extended_rect, HIGHLIGHT_COLOR)
        
        # highlight overlay for drag operations
        if (self.drag_highlight or self.hover_highlight) and self.row >= 0:
            p.fillRect(extended_rect, DRAG_HIGHLIGHT_COLOR)
        
        # Highlight if this is the currently remembered piece
        if hasattr(self.board_editor, 'remembered_piece') and self.board_editor.remembered_piece == self.piece_label and self.is_palette:
            p.fillRect(extended_rect, MEMORY_HIGHLIGHT_COLOR)

        # piece icon
        icon = get_piece_pixmap(self.piece_label)
        x = (self.width() - icon.width()) // 2
        y = (self.height() - icon.height()) // 2
        p.drawPixmap(x, y, icon)
        
    def enterEvent(self, event):
        """Handle mouse entering the widget - for cursor highlight"""
        if (hasattr(self.board_editor, 'remembered_piece') and 
            self.board_editor.remembered_piece and 
            self.row >= 0):  # Only highlight board squares, not palette
            self.set_hover_highlight(True)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget - for cursor highlight"""
        self.set_hover_highlight(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # First check for en passant selection
        if (self.board_editor and
            self.board_editor.ep_highlight_on and
            (self.row, self.col) in self.board_editor.ep_possible):
            self.board_editor.on_ep_square_clicked(self.row, self.col)
            return
            
        # Handle right-click for erasing a piece
        if event.button() == Qt.RightButton and self.row >= 0:
            if self.piece_label != "empty":
                self.piece_label = "empty"
                self.update()
                if self.board_editor:               # keep FEN in sync, etc.
                    self.board_editor.sync_squares_to_labels()
                    # Save state after erasing a piece
                    if hasattr(self.board_editor, 'save_state'):
                        self.board_editor.save_state()
            return  # stop; no drag on right click
            
        # If left-clicking on a board square and we have a remembered piece, place it
        if (event.button() == Qt.LeftButton and 
            self.row >= 0 and 
            hasattr(self.board_editor, 'remembered_piece') and 
            self.board_editor.remembered_piece):
            # Place the remembered piece
            self.piece_label = self.board_editor.remembered_piece
            self.update()
            if self.board_editor:
                self.board_editor.sync_squares_to_labels()
                # Save state after placing a remembered piece
                if hasattr(self.board_editor, 'save_state'):
                    self.board_editor.save_state()
            return

        # Otherwise do normal drag behavior
        if event.button() == Qt.LeftButton and (self.row < 0 or self.piece_label != "empty"):
            piece_being_dragged = self.piece_label
            if self.row >= 0:
                self.piece_label = "empty"
                self.update()
                if self.board_editor:
                    self.board_editor.sync_squares_to_labels()

            drag = QDrag(self)
            mime = QMimeData()
            mime.setData("application/x-chess-piece",
                        f"{self.row},{self.col},{piece_being_dragged}".encode())
            drag.setMimeData(mime)

            icon = get_piece_pixmap(piece_being_dragged)
            drag_pix = QPixmap(icon.size())         # pixmap is just the icon size
            drag_pix.fill(Qt.transparent)           # keep alpha = 0 everywhere
            p = QPainter(drag_pix)
            p.drawPixmap(0, 0, icon)                # no square colour, just the piece
            p.end()

            drag.setPixmap(drag_pix)
            drag.setHotSpot(drag_pix.rect().center())  # cursor "grabs" the piece centre
            result = drag.exec_(Qt.MoveAction)

            # if the drop was rejected / cancelled, restore the piece
            if result != Qt.MoveAction and self.row >= 0:
                self.piece_label = piece_being_dragged
                self.update()
                if self.board_editor:
                    self.board_editor.sync_squares_to_labels()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to remember a piece"""
        if event.button() == Qt.LeftButton and self.is_palette and hasattr(self.board_editor, 'set_remembered_piece'):
            # Toggle or set piece memory on double-click for palette pieces
            self.board_editor.set_remembered_piece(self.piece_label)
            self.update()  # Update to show selection highlight
            return
        super().mouseDoubleClickEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-chess-piece"):
            if self.row >= 0:  # Only highlight board squares, not palette
                self.set_drag_highlight(True)
            event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        self.set_drag_highlight(False)
        super().dragLeaveEvent(event)

    # This was previously aliased, but we need different behavior now
    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-chess-piece"):
            if self.row >= 0:  # Only highlight board squares, not palette
                self.set_drag_highlight(True)
            event.acceptProposedAction()

    # handle drop: clear origin, place here, sync board state
    def dropEvent(self, event):
        # Clear highlight when dropping
        self.set_drag_highlight(False)
        
        data = bytes(event.mimeData().data("application/x-chess-piece")).decode()
        src_r, src_c, lbl = data.split(',')
        src_r, src_c = int(src_r), int(src_c)

        # clear origin square if it was on the board
        if src_r >= 0:
            origin_sq = self.board_editor.squares[src_r][src_c]
            origin_sq.piece_label = "empty"
            origin_sq.update()

        # place piece on this square
        self.piece_label = lbl
        self.update()

        # keep BoardEditor's 2‑D state in sync for FEN, flips, etc.
        self.board_editor.sync_squares_to_labels()
        
        # Save state after dragging and dropping a piece
        if hasattr(self.board_editor, 'save_state'):
            self.board_editor.save_state()

        event.acceptProposedAction()