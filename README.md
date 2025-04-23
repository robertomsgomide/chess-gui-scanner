# chess-scanner

A PyQt5-based desktop application that **captures chessboards from your screen, automatically identifies pieces using a neural network, predicts board orientation and active player, and lets you edit positions with a convenient drag-and-drop interface**.

---

## Key Features

| Feature | Details |
|---------|---------|
| **Screen Capture** | Full-screen overlay with 8×8 grid for precise board selection |
| **Piece Recognition** | Convolutional neural network (~0.85M parameters) identifies pieces from screenshots |
| **Position Analysis** | Uses both heuristics and machine learning to predict board orientation and side to move |
| **Advanced Board Editor** | Intuitive interface with drag-and-drop, piece memory, castling rights and en passant detection |
| **Self-Improving AI** | Learns from your corrections to improve future recognition accuracy |
| **UCI Engine Support** | Connect to any compatible chess engine (e.g., Stockfish) for instant position analysis |
| **Fully Offline** | No internet connection required - everything runs locally |

---

## Quick Start

### 1. Clone and set up environment
```bash
git clone https://github.com/<your-username>/chess-ai-scanner.git
cd chess-ai-scanner
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add a chess engine (optional)
```bash
# Place any UCI engine executable in the 'engine' folder
```

### 4. Run the application
```bash
python main.py
```

### Requirements
- Python ≥ 3.9
- PyQt5
- python-chess
- PyTorch & torchvision
- Pillow
- pyautogui
- numpy

---

## How It Works

1. **Capture** - Use the screen overlay to select any chessboard visible on your screen
2. **Recognize** - The CNN classifies each square and predicts the board's orientation
3. **Edit** - Make corrections with intuitive tools:
   - Drag pieces from the palette or right-click to clear squares
   - Double-click pieces for "sticky" placement mode
   - Automatic castling rights detection based on king and rook positions
   - En passant target highlighting and selection
4. **Analyze** - Run a UCI engine to evaluate the position with multiple lines of analysis
5. **Learn** - The model improves with each correction, becoming more accurate over time

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c6594f4b-6fff-49b0-a3f3-50baaf257380" alt="Main window with captured board"></td>
    <td><img src="https://github.com/user-attachments/assets/4710b8e6-4b21-446f-897f-7a9f44fafc0f" alt="Board editor with engine analysis"></td>
  </tr>
  <tr>
    <td align="center"><em>Main window with board preview</em></td>
    <td align="center"><em>Board editor with engine analysis panel</em></td>
  </tr>
</table>

---

## Technical Details

The application consists of:

- **BoardAnalyzer**: Predicts board orientation and side to move using k-nearest neighbors and heuristics
- **CNNClassifier**: Neural network for piece recognition with continuous learning capability
- **SnipOverlay**: Full-screen capture interface with 8×8 grid alignment
- **BoardEditor**: Feature-rich position editor with drag-and-drop interface and UCI engine integration

The neural network model (`SimpleCNN`) uses:
- 3 convolutional blocks with batch normalization
- Adaptive pooling for size flexibility
- Dropout layers for regularization
- Incremental learning that preserves previously gained knowledge

All training data is saved locally, allowing the model to improve with each use session.

---

## License

- **Code License**: This project's code is licensed under the [MIT License](LICENSE).
- **Usage Terms**: By using this application, you agree to our [Terms of Use](TERMS_OF_USE.md), which prohibit using this tool for cheating in chess games.
- Chess piece icons CC BY-SA 3.0 from Wikimedia Commons.
