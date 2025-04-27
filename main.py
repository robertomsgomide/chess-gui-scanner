import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QSettings
from labels import *
from CNNClassifier import *
from BoardSquareWidget import *
from BoardEditor import *
from SnipOverlay import *
from MainWindow import *
from BoardAnalyzer import BoardAnalyzer


def set_fusion_dark_theme(app):
    """Apply a modern dark fusion theme to the application"""
    app.setStyle("Fusion")
    
    # Dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)
    
    # Additional stylesheet for fine-tuning
    app.setStyleSheet("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white; 
        }
        QPushButton { 
            background-color: #353535; 
            border: 1px solid #5c5c5c;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover { 
            background-color: #454545; 
        }
        QPushButton:pressed { 
            background-color: #252525; 
        }
        QMessageBox { 
            background-color: #353535; 
        }
    """)

def set_fusion_light_theme(app):
    """Apply a modern light fusion theme to the application"""
    app.setStyle("Fusion")
    
    # Light palette with blue accents
    light_palette = QPalette()
    light_palette.setColor(QPalette.Window, QColor(240, 240, 240))
    light_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.AlternateBase, QColor(233, 231, 227))
    light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.Text, QColor(0, 0, 0))
    light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    light_palette.setColor(QPalette.Link, QColor(0, 100, 200))
    light_palette.setColor(QPalette.Highlight, QColor(38, 110, 183))
    light_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(light_palette)
    
    # Additional stylesheet for fine-tuning
    app.setStyleSheet("""
        QToolTip { 
            color: #000000; 
            background-color: #ffffff; 
            border: 1px solid #bdbdbd; 
        }
        QPushButton { 
            background-color: #f0f0f0; 
            border: 1px solid #bdbdbd;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover { 
            background-color: #e0e0e0; 
        }
        QPushButton:pressed { 
            background-color: #d0d0d0; 
        }
    """)

def set_chess_wooden_theme(app):
    """Apply a chess-themed style with wooden colors"""
    app.setStyle("Fusion")
    
    # Warm wooden palette
    wooden_palette = QPalette()
    wooden_palette.setColor(QPalette.Window, QColor(240, 220, 180))           # Light wood
    wooden_palette.setColor(QPalette.WindowText, QColor(70, 35, 10))          # Dark brown text
    wooden_palette.setColor(QPalette.Base, QColor(255, 250, 240))             # Light cream
    wooden_palette.setColor(QPalette.AlternateBase, QColor(230, 210, 175))    # Alternate wood
    wooden_palette.setColor(QPalette.ToolTipBase, QColor(253, 248, 228))      # Light cream
    wooden_palette.setColor(QPalette.ToolTipText, QColor(70, 35, 10))         # Dark brown
    wooden_palette.setColor(QPalette.Text, QColor(70, 35, 10))                # Dark brown text
    wooden_palette.setColor(QPalette.Button, QColor(210, 180, 140))           # Medium wood
    wooden_palette.setColor(QPalette.ButtonText, QColor(70, 35, 10))          # Dark brown
    wooden_palette.setColor(QPalette.BrightText, QColor(180, 0, 0))           # Deep red
    wooden_palette.setColor(QPalette.Link, QColor(150, 80, 0))                # Brown links
    wooden_palette.setColor(QPalette.Highlight, QColor(190, 100, 30))         # Highlight wood
    wooden_palette.setColor(QPalette.HighlightedText, QColor(255, 250, 240))  # Light cream
    
    app.setPalette(wooden_palette)
    
    # Wooden-themed stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0deb4;
        }
        QToolTip { 
            color: #462306; 
            background-color: #fdf8e4; 
            border: 1px solid #d2b48c; 
        }
        QPushButton { 
            background-color: #d2b48c; 
            border: 1px solid #b38b50;
            padding: 5px;
            border-radius: 3px;
            color: #462306;
            font-weight: bold;
        }
        QPushButton:hover { 
            background-color: #c2a478; 
        }
        QPushButton:pressed { 
            background-color: #b28b58; 
        }
        QLabel {
            color: #462306;
        }
        QDialog {
            background-color: #f0deb4;
        }
    """)

def show_terms_dialog():
    terms = """
    <b>Terms of Use</b><br>
    By using chess-scanner, you agree to:
    <ul>
    <li>Not use this software to cheat in live chess games or tournaments</li>
    <li>Use this tool only for educational purposes and post-game analysis</li>
    <li>Respect fair play policies of chess platforms and organizations</li>
    </ul>
    Do you agree to these terms?
    """
    
    dialog = QMessageBox()
    dialog.setWindowTitle("Terms of Use")
    dialog.setText(terms)
    dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    dialog.setDefaultButton(QMessageBox.Yes)
    
    # Set window icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "Chess_icon.png")
    dialog.setWindowIcon(QIcon(icon_path))
    
    result = dialog.exec_()
    if result == QMessageBox.No:
        sys.exit()
    
    # Save acceptance to user settings
    settings = QSettings("ChessAIScanner", "Settings")
    settings.setValue("terms_accepted", True)

def main():
    app = QApplication(sys.argv)
    initialize_icons()
    
    settings = QSettings("ChessAIScanner", "Settings")
    
    # Check terms acceptance
    if not settings.value("terms_accepted", False, type=bool):
        show_terms_dialog()
    
    # Choose one of these themes (comment out the others)
    # set_fusion_dark_theme(app)    # Modern dark theme
    set_fusion_light_theme(app)   # Modern light theme
    # set_chess_wooden_theme(app)   # Chess-themed wooden colors
    
    classifier = CNNClassifier()
    
    # Create a new main window with the loaded or new analyzer
    w = MainWindow(classifier)
    
    # Load analyzer state if it exists
    try:
        w.analyzer = BoardAnalyzer.load_from_disk()
    except Exception as e:
        print(f"Error loading analyzer: {e}")
    
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
