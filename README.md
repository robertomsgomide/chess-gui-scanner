# chess-gui-scanner


Capture any chessboard from your screen and analyse it instantly :)

![Demo](https://github.com/user-attachments/assets/a0373e94-41f5-4d65-a2ac-5773d0d7ab07)

## Highlights

• Screen overlay with 8×8 grid for pixel-perfect selection  
• Lightweight CNN recognizes every square (~0.9 M parameters)  
• Continuous learning – every manual correction can be stored as training data  
• Feature-rich editor with drag-and-drop, undo/redo, and position validation  
• Edit and Play modes with move recording and PGN export  
• UCI engine integration (Stockfish, lc0, etc.) with navigable analysis lines   
• Works on Windows, Linux and macOS

---

## Installation

1. Install **Python ≥ 3.9**.
2. Clone the repository and (optionally) create a virtual environment:

```bash
git clone https://github.com/robertomsgomide/chess-gui-scanner.git
cd chess-gui-scanner

python -m venv .venv        # optional but recommended
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install the Python requirements:

```bash
pip install -r requirements.txt
```

4. (Optional) place your favorite UCI engine executable (e.g. `stockfish.exe`) in the `engine/` folder.

---

## Usage

```bash
python main.py
```

1. Press **Capture Board** and draw a rectangle around the chessboard on screen, or click **Auto Detect Board** for automatic detection.
2. Verify / correct the automatically recognized position in Edit mode.
3. Click **Finish Edit** to enter Play mode for making moves, or use **Analysis** to examine the position with an engine.


## Project layout

```
chess-gui-scanner/
├─ main.py                    # Entry point
├─ MainWindow.py              # Main application window
├─ BoardEditor.py             # Position editor and play interface
├─ ChessBoardModel.py         # Core board state management
├─ StateController.py         # Edit/Play mode management
├─ AnalysisManager.py         # Engine analysis and navigation
├─ HistoryManager.py          # Undo/redo functionality
├─ PgnManager.py              # Move recording and PGN export
├─ CNNClassifier.py           # PyTorch piece classifier
├─ BoardAnalyzer.py           # Heuristic orientation predictor
├─ SnipOverlay.py             # Full-screen capture overlay
├─ AutoDetector.py            # Automatic board detection
├─ BoardSquareWidget.py       # Individual square widgets
├─ restore_training_data.py   # Helper script to manage all chess training data
├─ labels.py                  # Piece definitions and utilities
├─ engine/                    # Place UCI engines here
└─ icons/                     # Piece bitmaps
```

---

## Tech stack

• PyQt5 — cross-platform desktop GUI  
• PyTorch & TorchVision — CNN and image pre-processing  
• Pillow — handling raw screen captures  
• python-chess — FEN generation & engine communication  
• numpy — feature extraction  
• pyautogui — cross-platform screen-capture helper  
• OpenCV — automatic board detection and image processing

---

## License & Usage

This project is licensed under the MIT License (see `LICENSE`).

By using Chess Scanner you also agree to the [Terms of Use](TERMS_OF_USE.md), which *strictly forbid* employing the software for real-time engine assistance during rated, competitive, or otherwise fair-play chess games.
