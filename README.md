# Chess Scanner

Capture any chessboard from your screen and analyse it instantly — fully offline.

![Demo screenshot](docs/demo.gif)

## Highlights

• Screen overlay with 8×8 grid for pixel-perfect selection  
• Lightweight CNN recognises every square (~0.9 M parameters)  
• Smart heuristics predict board orientation *and* side-to-move  
• Feature-rich editor (drag-and-drop, sticky pieces, auto-castling & en-passant)  
• Continuous learning – every manual correction can be stored as training data  
• UCI engine integration (Stockfish, Berserk, etc.) for instant evaluation  
• Works on Windows, Linux and macOS – **no internet connection required**

---

## Installation

1. Install **Python ≥ 3.9**.
2. Clone the repository and (optionally) create a virtual environment:

```bash
git clone https://github.com/<your-username>/chess-scanner.git
cd chess-scanner

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

4. (Optional) place your favourite UCI engine executable (e.g. `stockfish.exe`) in the `engine/` folder.

---

## Usage

```bash
python main.py
```

1. Press **Capture Board** and draw a rectangle around the chessboard on screen.  
2. Verify / correct the automatically recognised position.  
3. Click **Analyse** to let the engine examine the position.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| Esc | Cancel capture / close dialog |
| R   | Re-capture board |
| Ctrl&nbsp;+&nbsp;S | Copy FEN to clipboard |
| Space | Start / stop engine |

---

## Project layout

```
chess-scanner/
├─ BoardEditor.py        # GUI for editing / analysing positions
├─ BoardAnalyzer.py      # Heuristic + k-NN orientation predictor
├─ CNNClassifier.py      # PyTorch piece classifier
├─ SnipOverlay.py        # Full-screen capture overlay
├─ main.py               # Entry point
├─ engine/               # Place UCI engines here
└─ icons/                # Piece bitmaps
```

---

## Tech stack

• PyQt5 — cross-platform desktop GUI  
• PyTorch & TorchVision — CNN and image pre-processing  
• Pillow — handling raw screen captures  
• python-chess — FEN generation & engine communication  
• numpy — feature extraction  
• pyautogui — cross-platform screen-capture helper  

---

## Contributing

Pull requests are welcome! For larger changes, please open an issue first to discuss what you would like to change.

---

## License & Usage

This project is licensed under the MIT License (see `LICENSE`).

By using Chess Scanner you also agree to the [Terms of Use](TERMS_OF_USE.md), which *strictly forbid* employing the software for real-time engine assistance during rated, competitive, or otherwise fair-play chess games.
