import os
import pyautogui
from BoardEditor import BoardEditor
from SnipOverlay import SnipOverlay
from labels import labels_to_fen
from BoardAnalyzer import BoardAnalyzer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QPixmap, QImage, QIcon)
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton,
                              QLabel, QVBoxLayout, QDialog, QApplication)


#########################################
# MainWindow
#########################################

class MainWindow(QMainWindow):
    def __init__(self, classifier):
        super().__init__()
        self.setWindowTitle("Chessboard Scanner")
        wincon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "Chess_icon.png")
        self.setWindowIcon(QIcon(wincon_path))
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.info_label = QLabel("Click 'Capture Board' to begin.")
        self.capture_btn = QPushButton("Capture Board")
        self.capture_btn.clicked.connect(self.on_capture)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(300,300)
        self.preview_label.setStyleSheet("border:1px solid gray;")

        self.edit_btn = QPushButton("Recognize/Edit")
        self.edit_btn.setEnabled(False)
        self.edit_btn.clicked.connect(self.on_edit)

        layout.addWidget(self.info_label)
        layout.addWidget(self.capture_btn)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.edit_btn)

        self.classifier = classifier
        self.analyzer = BoardAnalyzer()  # New board analyzer
        self.captured_pil = None
        self.resize(500,500)

    def on_capture(self):
        self.hide()
        QApplication.processEvents()
        overlay = SnipOverlay()
        overlay.exec_()
        rect = overlay.selectionRect
        if not rect:
            self.show()
            self.info_label.setText("No valid selection.")
            return
        shot = pyautogui.screenshot()
        left   = rect.left()
        top    = rect.top()
        right  = left + rect.width()
        bottom = top  + rect.height()
        cropped = shot.crop((left, top, right, bottom))
        self.captured_pil = cropped
        self.show()
        self.info_label.setText("Board captured.")
        self.show_preview(cropped)
        self.edit_btn.setEnabled(True)

    def show_preview(self, pil_img):
        pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw","RGBA")
        qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.preview_label.width(), self.preview_label.height(),
                            Qt.KeepAspectRatio)
        self.preview_label.setPixmap(scaled)

    def on_edit(self):
        if self.captured_pil is None:
            return

        # slice the captured PIL into 8 squares
        labels_2d, squares_imgs_2d = self.do_recognition(self.captured_pil)
        
        # Analyze the position to predict orientation and side to move
        predicted_is_flipped = self.analyzer.predict_orientation(labels_2d)
        predicted_side_to_move = self.analyzer.predict_side_to_move(labels_2d)
        
        # Create editor with predictions
        editor = BoardEditor(labels_2d, predicted_is_flipped, predicted_side_to_move)
        
        if editor.exec_() == QDialog.Accepted:
            # the user pressed "Learn"
            final_labels = editor.get_final_labels_2d()
            side = editor.get_side_to_move()
            fen = labels_to_fen(final_labels, side)
            self.info_label.setText("Board corrected. FEN: " + fen)

            # train on the user's corrected squares:
            self.classifier.train_on_data(squares_imgs_2d, final_labels)
            
            # Save the orientation and side to move for future predictions
            final_is_flipped = editor.is_flipped and editor.coords_switched  # Logical AND - true flipped state
            self.analyzer.save_training_data(final_labels, final_is_flipped, side)

        else:
            self.info_label.setText("Editing canceled.")

    def do_recognition(self, pil_img):
        """
        CNN: chop the 8x8 region into squares, pass each to classifier.
        Return (labels_2d, squares_2d_of_pil).
        """
        w, h = pil_img.size
        sq_w = w//8
        sq_h = h//8
        labels_2d = []
        squares_2d = []
        for r in range(8):
            row_lbls = []
            row_imgs = []
            for c in range(8):
                x1 = c*sq_w
                y1 = r*sq_h
                x2 = x1+sq_w
                y2 = y1+sq_h
                sq_img = pil_img.crop((x1,y1,x2,y2))
                lbl = self.classifier.predict_label(sq_img)
                row_lbls.append(lbl)
                row_imgs.append(sq_img)
            labels_2d.append(row_lbls)
            squares_2d.append(row_imgs)
        return labels_2d, squares_2d