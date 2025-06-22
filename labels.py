import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "icons")

# Global variable to track if we're in dark mode
_is_dark_mode = False

#########################################
# Chess Data
#########################################

PIECE_LABELS = [
    "empty",
    "wp","wn","wb","wr","wq","wk",
    "bp","bn","bb","br","bq","bk"
]

PIECE_ICON_PATHS = {
    "wp": os.path.join(ICON_DIR, "Chess_plt.png"),
    "wn": os.path.join(ICON_DIR, "Chess_nlt.png"),
    "wb": os.path.join(ICON_DIR, "Chess_blt.png"),
    "wr": os.path.join(ICON_DIR, "Chess_rlt.png"),
    "wq": os.path.join(ICON_DIR, "Chess_qlt.png"),
    "wk": os.path.join(ICON_DIR, "Chess_klt.png"),

    "bp": os.path.join(ICON_DIR, "Chess_pdt.png"),
    "bn": os.path.join(ICON_DIR, "Chess_ndt.png"),
    "bb": os.path.join(ICON_DIR, "Chess_bdt.png"),
    "br": os.path.join(ICON_DIR, "Chess_rdt.png"),
    "bq": os.path.join(ICON_DIR, "Chess_qdt.png"),
    "bk": os.path.join(ICON_DIR, "Chess_kdt.png"),
}

PIECE_ICONS = {}
PIECE_ICONS_DARK = {}  # Use original PNGs exactly as stored

def set_dark_mode(dark_mode: bool):
    """Set whether we're in dark mode - called from theme initialization"""
    global _is_dark_mode
    _is_dark_mode = dark_mode

def is_dark_mode() -> bool:
    """Check if we're currently in dark mode"""
    return _is_dark_mode

def initialize_icons():
    """Load all piece icons once at startup."""
    for label, path in PIECE_ICON_PATHS.items():
        pm = QPixmap(path)
        PIECE_ICONS[label] = pm
        
        # Use original PNGs exactly as stored for both light and dark modes
        PIECE_ICONS_DARK[label] = pm

def get_piece_pixmap(label: str) -> QPixmap:
    """Return the QPixmap for 'label', or blank if 'empty'."""
    if label == "empty":
        blank = QPixmap(60, 60)
        blank.fill(Qt.transparent)
        return blank
        
    # Choose the appropriate icon set based on current theme
    icon_set = PIECE_ICONS_DARK if _is_dark_mode else PIECE_ICONS
    pix = icon_set.get(label)
    
    if not pix:
        fallback = QPixmap(60, 60)
        fallback.fill(Qt.red)
        return fallback
    return pix

def labels_to_fen(labels_2d, side_to_move='w', castling='-', ep_target='-'):
    """
    Convert the 8x8 board of labels into a FEN string,
    specifying side_to_move as 'w' or 'b'.
    """
    piece_map = {
        'wp': 'P', 'wn': 'N', 'wb': 'B', 'wr': 'R', 'wq': 'Q', 'wk': 'K',
        'bp': 'p', 'bn': 'n', 'bb': 'b', 'br': 'r', 'bq': 'q', 'bk': 'k'
    }
    fen_rows = []
    for row in labels_2d:
        empties = 0
        out = ""
        for lbl in row:
            if lbl == "empty":
                empties += 1
            else:
                if empties:
                    out += str(empties)
                    empties = 0
                out += piece_map.get(lbl, '?')
        if empties:
            out += str(empties)
        fen_rows.append(out)

    return "/".join(fen_rows) + f" {side_to_move} {castling or '-'} {ep_target} 0 1"