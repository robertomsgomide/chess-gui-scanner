from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import (QPainter, QColor, QPen, QFont)
from PyQt5.QtCore import (Qt, QRect)


#########################################
# SnipOverlay
#########################################

class SnipOverlay(QDialog):
    """A fullscreen overlay for selecting screen regions."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.showFullScreen()
        self.setWindowOpacity(0.3)
        self.instruction = "Select an 8x8 region. ESC to cancel."
        self.font = QFont("Helvetica", 16, QFont.Bold)
        self.dragging = False
        self.start_x = 0
        self.start_y = 0
        self.rect_x = 0
        self.rect_y = 0
        self.rect_w = 0
        self.rect_h = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        fm = painter.fontMetrics()
        text_h = fm.height() + 20
        painter.fillRect(self.rect(), QColor(0,0,0,100))
        painter.setFont(self.font)
        painter.setPen(QColor(255,255,255))
        top_rect = QRect(0, 0, self.width(), text_h)
        painter.drawText(top_rect, Qt.AlignHCenter | Qt.AlignTop, self.instruction)
        if self.rect_w>0 and self.rect_h>0:
            pen = QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.rect_x, self.rect_y, self.rect_w, self.rect_h)

            col_w = self.rect_w / 8
            row_h = self.rect_h / 8
            for c in range(1,8):
                x = int(self.rect_x + c*col_w)
                painter.drawLine(x, self.rect_y, x, self.rect_y+self.rect_h)
            for r in range(1,8):
                y = int(self.rect_y + r*row_h)
                painter.drawLine(self.rect_x, y, self.rect_x+self.rect_w, y)

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.dragging=True
            self.start_x = event.x()
            self.start_y = event.y()
            self.rect_w=0
            self.rect_h=0

    def mouseMoveEvent(self, event):
        if self.dragging:
            x = self.start_x
            y = self.start_y
            w = event.x() - x
            h = event.y() - y
            if w<0:
                x+=w
                w=abs(w)
            if h<0:
                y+=h
                h=abs(h)
            # snap to multiples of 8
            w = (w//8)*8
            h = (h//8)*8
            self.rect_x=x
            self.rect_y=y
            self.rect_w=w
            self.rect_h=h
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.dragging=False
            self.close()

    @property
    def selectionRect(self):
        if self.rect_w<2 or self.rect_h<2:
            return None
        return QRect(self.rect_x, self.rect_y, self.rect_w, self.rect_h)

    def keyPressEvent(self, event):
        if event.key()==Qt.Key_Escape:
            self.rect_w=0
            self.rect_h=0
            self.close()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.close()
        else:
            super().keyPressEvent(event)