import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtCore import QSettings
from labels import initialize_icons, set_dark_mode
from CNNClassifier import CNNClassifier
from MainWindow import MainWindow


def set_fusion_dark_theme(app):
    """Apply a modern dark fusion theme matching VS Code Dark+ style"""
    app.setStyle("Fusion")
    
    # Set dark mode flag for icon system
    set_dark_mode(True)
    
    # High contrast dark palette for better readability
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))                # Main background
    dark_palette.setColor(QPalette.WindowText, QColor(224, 224, 224))         # High contrast text (#E0E0E0)
    dark_palette.setColor(QPalette.Base, QColor(38, 38, 38))                  # Input backgrounds (#262626)
    dark_palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))         # Alternate rows
    dark_palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))           # Tooltip background
    dark_palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))        # Tooltip text
    dark_palette.setColor(QPalette.Text, QColor(224, 224, 224))               # Input text (#E0E0E0)
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))                # Button background
    dark_palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))         # Button text (#E0E0E0)
    dark_palette.setColor(QPalette.BrightText, QColor(255, 100, 100))         # Error text
    dark_palette.setColor(QPalette.Link, QColor(100, 150, 230))               # Links
    dark_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))            # Selection (VS Code blue)
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))    # Selected text
    
    # Disabled state colors
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(94, 94, 94))   # #5E5E5E
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(94, 94, 94))   # #5E5E5E
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(94, 94, 94))         # #5E5E5E
    dark_palette.setColor(QPalette.Disabled, QPalette.Button, QColor(35, 35, 35))
    
    app.setPalette(dark_palette)
    
    # Comprehensive dark theme stylesheet with improved contrast
    app.setStyleSheet("""
        /* Global focus removal - kill all blue rings */
        * {
            outline: none;
        }
        *:focus {
            outline: none;
        }
        /* Global font settings */
        * {
            font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", Arial, sans-serif;
            font-weight: 400;
        }
        
        /* Headings use medium weight */
        QLabel[accessibleName="heading"], QGroupBox::title {
            font-weight: 500;
        }
        
        /* Main window and dialogs */
        QMainWindow, QDialog {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        
        /* Tooltips */
        QToolTip { 
            color: #e0e0e0; 
            background-color: #2d2d2d; 
            border: 1px solid #454545;
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 12px;
        }
        
        /* Buttons with emergency patch colors */
        QPushButton { 
            background-color: #404040; 
            border: 1px solid #4a4a4a;
            color: #f0f0f0;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
        }
        QPushButton:hover { 
            background-color: #454545; 
            border-color: #4a4a4a;
        }
        QPushButton:pressed { 
            background-color: #383838; 
        }
        QPushButton:disabled {
            background-color: #232323;
            color: #5e5e5e;
            border-color: #3a3a3a;
        }
        QPushButton:focus {
            outline: none;
        }
        
        /* Text inputs and text areas */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #404040;
            border: 1px solid #4a4a4a;
            color: #e0e0e0;
            selection-background-color: #0078d4;
            selection-color: #ffffff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #0078d4;
            outline: none;
        }
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
            background-color: #1a1a1a;
            color: #5e5e5e;
            border-color: #3a3a3a;
        }
        
        /* Labels */
        QLabel {
            color: #e0e0e0;
            font-size: 13px;
        }
        QLabel:disabled {
            color: #5e5e5e;
        }
        
        /* Chess board container - raised card for better contrast */
        QFrame[accessibleName="boardContainer"], QWidget[accessibleName="boardContainer"] {
            background-color: #404040;
            border: 1px solid #4a4a4a;
            border-radius: 8px;
        }
        
        /* Reduce icon panel padding */
        QFrame[accessibleName="iconPanel"], QWidget[accessibleName="iconPanel"] {
            padding: 4px;
            margin: 2px;
            background-color: #505050;
            border: 1px solid #606060;
            border-radius: 6px;
        }
        
        /* Checkboxes and radio buttons */
        QCheckBox, QRadioButton {
            color: #e0e0e0;
            spacing: 8px;
            font-size: 13px;
        }
        QCheckBox:disabled, QRadioButton:disabled {
            color: #5e5e5e;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 18px;
            height: 18px;
            background-color: #404040;
            border: 2px solid #4a4a4a;
        }
        QCheckBox::indicator {
            border-radius: 4px;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d4;
            border-color: #0078d4;
        }
        QCheckBox::indicator:focus {
            outline: none;
        }
        QRadioButton::indicator {
            border-radius: 9px;
        }
        QRadioButton::indicator:checked {
            background-color: #0078d4;
            border-color: #0078d4;
        }
        QRadioButton::indicator:focus {
            outline: none;
        }
        
        /* Scrollbars */
        QScrollBar:vertical {
            background-color: #2d2d2d;
            width: 12px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background-color: #454545;
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #565656;
        }
        QScrollBar:horizontal {
            background-color: #2d2d2d;
            height: 12px;
            border: none;
        }
        QScrollBar::handle:horizontal {
            background-color: #454545;
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #565656;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
        
        /* Menu bars and menus */
        QMenuBar {
            background-color: #404040;
            color: #e0e0e0;
            border-bottom: 1px solid #4a4a4a;
            padding: 2px;
            font-weight: 500;
        }
        QMenuBar::item {
            padding: 6px 12px;
            border-radius: 6px;
        }
        QMenuBar::item:selected {
            background-color: #383838;
        }
        QMenu {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #4a4a4a;
            border-radius: 8px;
            padding: 4px;
        }
        QMenu::item {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
        }
        QMenu::item:selected {
            background-color: #0078d4;
        }
        
        /* Message boxes */
        QMessageBox {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        
        /* Group boxes with better spacing */
        QGroupBox {
            font-weight: 500;
            font-size: 14px;
            color: #e0e0e0;
            border: 1px solid #4a4a4a;
            border-radius: 8px;
            margin: 8px 0px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0px 8px;
            color: #e0e0e0;
        }
        
        /* Table and list widgets for better contrast */
        QTableWidget, QListWidget, QTreeWidget {
            background-color: #404040;
            alternate-background-color: #454545;
            color: #e0e0e0;
            border: 1px solid #4a4a4a;
            border-radius: 6px;
        }
        QTableWidget::item, QListWidget::item, QTreeWidget::item {
            padding: 4px;
            border: none;
        }
        QTableWidget::item:selected, QListWidget::item:selected, QTreeWidget::item:selected {
            background-color: #0078d4;
            color: #ffffff;
        }
        
        /* Headers */
        QHeaderView::section {
            background-color: #404040;
            color: #e0e0e0;
            border: 1px solid #4a4a4a;
            padding: 6px;
            font-weight: 500;
        }
    """)

def set_fusion_light_theme(app):
    """Apply a modern flat light theme inspired by Material 3 and macOS Sonoma"""
    app.setStyle("Fusion")
    
    # Set light mode flag for icon system
    set_dark_mode(False)
    
    # Clean light palette with high contrast
    light_palette = QPalette()
    light_palette.setColor(QPalette.Window, QColor(250, 250, 250))            # Clean background #FAFAFA
    light_palette.setColor(QPalette.WindowText, QColor(28, 28, 28))           # High contrast text
    light_palette.setColor(QPalette.Base, QColor(255, 255, 255))              # Input backgrounds
    light_palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))     # Alternate rows
    light_palette.setColor(QPalette.ToolTipBase, QColor(248, 248, 248))       # Tooltip background
    light_palette.setColor(QPalette.ToolTipText, QColor(28, 28, 28))          # Tooltip text
    light_palette.setColor(QPalette.Text, QColor(28, 28, 28))                 # Input text
    light_palette.setColor(QPalette.Button, QColor(255, 255, 255))            # Button background (flat white)
    light_palette.setColor(QPalette.ButtonText, QColor(28, 28, 28))           # Button text
    light_palette.setColor(QPalette.BrightText, QColor(200, 50, 50))          # Error text
    light_palette.setColor(QPalette.Link, QColor(0, 120, 215))                # Links (primary accent)
    light_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))           # Selection accent
    light_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))   # Selected text
    
    # Disabled state colors
    light_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(140, 140, 140))
    light_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(140, 140, 140))
    light_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(140, 140, 140))
    light_palette.setColor(QPalette.Disabled, QPalette.Button, QColor(248, 248, 248))
    
    app.setPalette(light_palette)
    
    # Modern flat light theme stylesheet with Material Design influences
    app.setStyleSheet("""
        /* Global focus removal - kill all blue rings */
        * {
            outline: none;
        }
        *:focus {
            outline: none;
        }
        /* Global font settings */
        * {
            font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", Arial, sans-serif;
            font-weight: 400;
        }
        
        /* Headings use medium weight */
        QLabel[accessibleName="heading"], QGroupBox::title {
            font-weight: 500;
        }
        
        /* Main window and dialogs */
        QMainWindow, QDialog {
            background-color: #fafafa;
            color: #1c1c1c;
        }
        
        /* Tooltips */
        QToolTip { 
            color: #1c1c1c; 
            background-color: #ffffff; 
            border: 1px solid #e0e0e0;
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 12px;
        }
        
        /* Buttons - completely flat design with Material-style elevation */
        QPushButton { 
            background-color: #ffffff; 
            border: 1px solid #dddddd;
            color: #1c1c1c;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
        }
        QPushButton:hover { 
            background-color: #f8f8f8; 
            border-color: #bdbdbd;
        }
        QPushButton:pressed { 
            background-color: #f0f0f0; 
        }
        QPushButton:disabled {
            background-color: #f8f8f8;
            color: #8c8c8c;
            border-color: #e8e8e8;
        }
        QPushButton:focus {
            outline: none;
        }
        
        /* Primary accent buttons */
        QPushButton[accessibleName="primary"] {
            background-color: #0078d7;
            color: #ffffff;
            border-color: #0078d7;
        }
        QPushButton[accessibleName="primary"]:hover {
            background-color: #106ebe;
            border-color: #106ebe;
        }
        
        /* Text inputs and text areas */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #ffffff;
            border: 1px solid #dddddd;
            color: #1c1c1c;
            selection-background-color: #0078d7;
            selection-color: #ffffff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #0078d7;
            outline: none;
        }
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
            background-color: #f8f8f8;
            color: #8c8c8c;
            border-color: #e8e8e8;
        }
        
        /* Labels */
        QLabel {
            color: #1c1c1c;
            font-size: 13px;
        }
        QLabel:disabled {
            color: #8c8c8c;
        }
        
        /* Reduce icon panel padding */
        QFrame[accessibleName="iconPanel"], QWidget[accessibleName="iconPanel"] {
            padding: 4px;
            margin: 2px;
        }
        
        /* Chess board container - add card elevation */
        QFrame[accessibleName="boardContainer"], QWidget[accessibleName="boardContainer"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        
        /* Checkboxes and radio buttons */
        QCheckBox, QRadioButton {
            color: #1c1c1c;
            spacing: 8px;
            font-size: 13px;
        }
        QCheckBox:disabled, QRadioButton:disabled {
            color: #8c8c8c;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 18px;
            height: 18px;
            background-color: #ffffff;
            border: 2px solid #dddddd;
        }
        QCheckBox::indicator {
            border-radius: 4px;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d7;
            border-color: #0078d7;
        }
        QCheckBox::indicator:focus {
            outline: none;
        }
        QRadioButton::indicator {
            border-radius: 9px;
        }
        QRadioButton::indicator:checked {
            background-color: #0078d7;
            border-color: #0078d7;
        }
        QRadioButton::indicator:focus {
            outline: none;
        }
        
        /* Scrollbars - minimal flat design */
        QScrollBar:vertical {
            background-color: transparent;
            width: 12px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background-color: #cccccc;
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #999999;
        }
        QScrollBar:horizontal {
            background-color: transparent;
            height: 12px;
            border: none;
        }
        QScrollBar::handle:horizontal {
            background-color: #cccccc;
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #999999;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
        
        /* Menu bars and menus */
        QMenuBar {
            background-color: #ffffff;
            color: #1c1c1c;
            border-bottom: 1px solid #e0e0e0;
            padding: 2px;
            font-weight: 500;
        }
        QMenuBar::item {
            padding: 6px 12px;
            border-radius: 6px;
        }
        QMenuBar::item:selected {
            background-color: #f0f0f0;
        }
        QMenu {
            background-color: #ffffff;
            color: #1c1c1c;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 4px;
        }
        QMenu::item {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
        }
        QMenu::item:selected {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        /* Message boxes */
        QMessageBox {
            background-color: #ffffff;
            color: #1c1c1c;
        }
        
        /* Group boxes with better spacing */
        QGroupBox {
            font-weight: 500;
            font-size: 14px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin: 8px 0px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0px 8px;
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
    
    # Choose one of these themes (comment out the other)
    set_fusion_dark_theme(app)    # Modern dark theme (VS Code Dark+ style)
    #set_fusion_light_theme(app)     # Modern light theme (Material 3 / macOS Sonoma style)
    
    classifier = CNNClassifier()
    
    # Create a new main window with the loaded or new analyzer
    w = MainWindow(classifier)
    
    # Analyzer is already initialized in MainWindow constructor
    
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
