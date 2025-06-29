from labels import (get_piece_pixmap)
from PyQt5.QtCore import (Qt, QMimeData, QSize, QRect, pyqtSignal)
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
    
    Now uses signals instead of direct references to BoardEditor for better decoupling.
    """
    
    # Signals for communicating with parent
    squareClicked = pyqtSignal(int, int, int)  # row, col, button
    squareRightClicked = pyqtSignal(int, int)  # row, col
    squareDoubleClicked = pyqtSignal(int, int)  # row, col
    pieceDragStarted = pyqtSignal(int, int, str)  # row, col, piece_label
    pieceDropped = pyqtSignal(int, int, int, int, str)  # from_row, from_col, to_row, to_col, piece_label
    enPassantSquareClicked = pyqtSignal(int, int)  # row, col
    
    def __init__(self, row, col, piece_label="empty", parent=None, is_palette=False):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.piece_label = piece_label
        self.setFixedSize(60,60)
        self.setContentsMargins(0, 0, 0, 0)  # Remove any internal margins
        self.setStyleSheet("padding: 0px; margin: 0px; border: 0px; border-width: 0px;")  # Ensure no CSS styling adds spacing
        self.setAcceptDrops(True)
        self.highlight = False  # For en-passant
        self.drag_highlight = False  # For drag operations
        self.hover_highlight = False  # For cursor hover with remembered piece
        self.is_palette = is_palette  # Flag for palette pieces
        self.is_memory_highlighted = False  # For remembered piece highlight
        self.ep_highlight_on = False  # En passant highlight state
        self.ep_possible = {}  # En passant possible squares
        
        # For proper drag detection
        self.drag_start_position = None
    
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
        
    def set_memory_highlight(self, on: bool):
        self.is_memory_highlighted = on
        self.update()
        
    def set_ep_state(self, highlight_on: bool, possible_squares: dict):
        """Set en passant highlighting state"""
        self.ep_highlight_on = highlight_on
        self.ep_possible = possible_squares
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
            if self.parent():
                p.fillRect(extended_rect, self.parent().palette().color(self.parent().backgroundRole()))
            else:
                p.fillRect(extended_rect, Qt.lightGray)

        # highlight overlay for en-passant
        if self.highlight:
            p.fillRect(extended_rect, HIGHLIGHT_COLOR)
        
        # highlight overlay for drag operations
        if (self.drag_highlight or self.hover_highlight) and self.row >= 0:
            p.fillRect(extended_rect, DRAG_HIGHLIGHT_COLOR)
        
        # Highlight if this is the currently remembered piece
        if self.is_memory_highlighted and self.is_palette:
            p.fillRect(extended_rect, MEMORY_HIGHLIGHT_COLOR)

        # piece icon
        icon = get_piece_pixmap(self.piece_label)
        x = (self.width() - icon.width()) // 2
        y = (self.height() - icon.height()) // 2
        p.drawPixmap(x, y, icon)
        
    def enterEvent(self, event):
        """Handle mouse entering the widget - for cursor highlight"""
        # This will be handled by the parent now via signals
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget - for cursor highlight"""
        self.set_hover_highlight(False)
        # Clear drag start position when mouse leaves
        self.drag_start_position = None
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # First check for en passant selection
        if self.ep_highlight_on and (self.row, self.col) in self.ep_possible:
            self.enPassantSquareClicked.emit(self.row, self.col)
            return
        
        # Emit click signal with button information
        self.squareClicked.emit(self.row, self.col, event.button())
        
        # Handle right-click
        if event.button() == Qt.RightButton:
            self.squareRightClicked.emit(self.row, self.col)
            return  # stop; no drag on right click
        
        # For left-clicks on draggable pieces, just record the start position
        # Actual drag will start in mouseMoveEvent
        if event.button() == Qt.LeftButton and (self.row < 0 or self.piece_label != "empty"):
            self.drag_start_position = event.pos()
    
    def mouseMoveEvent(self, event):
        # Only start drag if we have a start position and mouse has moved enough
        if (self.drag_start_position is not None and 
            (event.buttons() & Qt.LeftButton) and
            (event.pos() - self.drag_start_position).manhattanLength() >= 3):  # 3 pixel threshold
            self._start_drag_operation()
            self.drag_start_position = None
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        # Clear drag start position on release
        self.drag_start_position = None
        super().mouseReleaseEvent(event)
    
    def _start_drag_operation(self):
        """Perform the actual drag operation"""
        # Signal that a drag is starting
        self.pieceDragStarted.emit(self.row, self.col, self.piece_label)
        
        # Temporarily clear the piece for visual feedback during drag
        piece_being_dragged = self.piece_label
        if self.row >= 0:
            self.piece_label = "empty"
            self.update()

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

    def mouseDoubleClickEvent(self, event):
        """Handle double-click"""
        if event.button() == Qt.LeftButton:
            # Clear any drag start position since this is a double-click
            self.drag_start_position = None
            self.squareDoubleClicked.emit(self.row, self.col)
        super().mouseDoubleClickEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-chess-piece"):
            if self.row >= 0:  # Only highlight board squares, not palette
                self.set_drag_highlight(True)
            event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        self.set_drag_highlight(False)
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-chess-piece"):
            if self.row >= 0:  # Only highlight board squares, not palette
                self.set_drag_highlight(True)
            event.acceptProposedAction()

    def dropEvent(self, event):
        # Clear highlight when dropping
        self.set_drag_highlight(False)
        
        data = bytes(event.mimeData().data("application/x-chess-piece")).decode()
        src_r, src_c, lbl = data.split(',')
        src_r, src_c = int(src_r), int(src_c)
        
        # Emit the drop signal
        self.pieceDropped.emit(src_r, src_c, self.row, self.col, lbl)
        
        # The parent will decide whether to accept or reject the drop
        event.acceptProposedAction()