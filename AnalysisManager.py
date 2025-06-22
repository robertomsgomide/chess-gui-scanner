import os
import sys
import chess
import chess.engine
from typing import List, Optional, Tuple, Callable
from ChessBoardModel import ChessBoardModel
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import QMessageBox, QApplication
#########################################
# AnalysisManager.py
#########################################

class AnalysisManager:
    """
    Manages chess engine analysis and navigation through analysis lines.
    """
    
    def __init__(self):
        self.settings = QSettings("ChessAIScanner", "Settings")
        self.selected_engine: Optional[str] = None
        self.analysis_lines: List[str] = []
        self.analysis_positions: List[List[ChessBoardModel]] = [[], [], []]
        self.selected_line_index = -1
        self.current_move_indices = [0, 0, 0]
        self.has_navigated = [False, False, False]
        self.original_position: Optional[ChessBoardModel] = None
        
        # Callbacks
        self.update_display_callback: Optional[Callable] = None
        self.update_board_callback: Optional[Callable[[ChessBoardModel], None]] = None
        
        # Load last used engine
        self._load_last_engine()
    
    def set_update_display_callback(self, callback: Callable):
        """Set callback to update analysis display"""
        self.update_display_callback = callback
    
    def set_update_board_callback(self, callback: Callable[[ChessBoardModel], None]):
        """Set callback to update board position"""
        self.update_board_callback = callback
    
    def _load_last_engine(self):
        """Load the last used engine from settings"""
        last_engine = self.settings.value("last_used_engine", None)
        if last_engine:
            engine_dir = os.path.join(os.path.dirname(__file__), "engine")
            engine_path = os.path.join(engine_dir, last_engine)
            if os.path.exists(engine_path):
                self.selected_engine = last_engine
    
    def get_available_engines(self) -> List[str]:
        """Get list of available UCI engines"""
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        engines = []
        
        try:
            if os.path.exists(engine_dir):
                # Find executables (recursively search in subdirectories)
                if sys.platform.startswith("win"):
                    for root, dirs, files in os.walk(engine_dir):
                        for file in files:
                            if file.lower().endswith(".exe"):
                                rel_path = os.path.relpath(os.path.join(root, file), engine_dir)
                                engines.append(rel_path)
                else:
                    for root, dirs, files in os.walk(engine_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path) and os.access(file_path, os.X_OK):
                                rel_path = os.path.relpath(file_path, engine_dir)
                                engines.append(rel_path)
        except Exception as e:
            print(f"Error finding engines: {e}")
        
        return engines
    
    def select_engine(self, engine_name: str) -> bool:
        """
        Select an engine for analysis.
        
        Args:
            engine_name: Name of the engine file
            
        Returns:
            True if engine was successfully selected
        """
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        engine_path = os.path.join(engine_dir, engine_name)
        
        if not os.path.exists(engine_path):
            return False
        
        self.selected_engine = engine_name
        self.settings.setValue("last_used_engine", engine_name)
        return True
    
    def get_selected_engine_name(self) -> Optional[str]:
        """Get display name of selected engine"""
        if not self.selected_engine:
            return None
        
        if os.path.sep in self.selected_engine:
            display_name = os.path.basename(self.selected_engine)
            display_name = os.path.splitext(display_name)[0]
        else:
            display_name = os.path.splitext(self.selected_engine)[0]
        
        return display_name
    

    
    def analyze_position(self, board_model: ChessBoardModel, depth: int = 20, time_limit: float = 10.0) -> bool:
        """
        Analyze the given position with the selected engine.
        
        Args:
            board_model: Board position to analyze
            depth: Analysis depth
            time_limit: Time limit in seconds
            
        Returns:
            True if analysis completed successfully
        """
        if not self.selected_engine:
            # Try to find any available engine
            engines = self.get_available_engines()
            if not engines:
                QMessageBox.critical(
                    None, "Engine not found",
                    f"No UCI engine executable found in engine directory.\n\n"
                    "Please add a chess engine (e.g. stockfish.exe) using the dropdown menu."
                )
                return False
            
            # Select first available engine
            self.select_engine(engines[0])
        
        engine_dir = os.path.join(os.path.dirname(__file__), "engine")
        engine_path = os.path.join(engine_dir, self.selected_engine)
        
        if not os.path.exists(engine_path):
            QMessageBox.critical(
                None, "Engine not found",
                f"Selected engine '{self.selected_engine}' not found.\n"
                "Please select a different engine."
            )
            return False
        
        # Save original position
        self.original_position = board_model.copy()
        
        # Clear previous analysis
        self.analysis_lines = []
        self.analysis_positions = [[], [], []]
        self.selected_line_index = -1
        self.current_move_indices = [0, 0, 0]
        self.has_navigated = [False, False, False]
        
        try:
            # Show busy cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Start engine
            engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=10.0)
            
            # Create chess board from model
            chess_board = chess.Board(board_model.get_fen())
            
            # Analyze with multipv=3
            info = engine.analyse(
                chess_board,
                chess.engine.Limit(depth=depth, time=time_limit),
                multipv=3
            )
            
            # Process results
            lines = []
            for i, pv in enumerate(info, 1):
                score = pv["score"].white()
                if score.is_mate():
                    score_str = f"# {score.mate()}"
                else:
                    score_str = f"{score.score()/100:.2f}"
                
                san_line = chess_board.variation_san(pv["pv"])
                lines.append(f"{i}.  {score_str}  |  {san_line}")
                
                # Generate positions for each move in this line
                current_positions = [board_model.copy()]  # Start with initial position
                position_board = chess.Board(board_model.get_fen())
                
                for move in pv["pv"]:
                    position_board.push(move)
                    # Create ChessBoardModel from the new position  
                    # Start with an empty board (proper 8x8 grid)
                    empty_labels = [['empty'] * 8 for _ in range(8)]
                    new_model = ChessBoardModel(empty_labels)
                    new_model.is_display_flipped = board_model.is_display_flipped
                    new_model.set_from_fen(position_board.fen())
                    current_positions.append(new_model)
                
                # Store positions for this line
                self.analysis_positions[i-1] = current_positions
            
            self.analysis_lines = lines
            self.selected_line_index = 0 if lines else -1
            
            # Quit engine
            engine.quit()
            
            # Update display
            if self.update_display_callback:
                self.update_display_callback()
            
            return True
            
        except chess.engine.EngineTerminatedError as e:
            QMessageBox.critical(
                None, "Engine crash",
                f"The engine '{os.path.splitext(self.selected_engine)[0]}' crashed unexpectedly.\n\n"
                f"Error: {str(e)}\n\n"
                "Try selecting a different engine."
            )
            return False
            
        except Exception as e:
            QMessageBox.critical(
                None, "Engine error",
                f"Error running engine: {str(e)}"
            )
            return False
            
        finally:
            QApplication.restoreOverrideCursor()
    
    def navigate_line(self, direction: str) -> bool:
        """
        Navigate through analysis line.
        
        Args:
            direction: 'next', 'previous', 'start', 'end', 'line_up', 'line_down'
            
        Returns:
            True if navigation was successful
        """
        if self.selected_line_index < 0 or self.selected_line_index >= len(self.analysis_lines):
            return False
        
        line_idx = self.selected_line_index
        
        if direction == 'next':
            line_positions = self.analysis_positions[line_idx]
            if line_positions and self.current_move_indices[line_idx] < len(line_positions) - 1:
                self.current_move_indices[line_idx] += 1
                self.has_navigated[line_idx] = True
                self._update_board_position()
                return True
                
        elif direction == 'previous':
            if self.current_move_indices[line_idx] > 0:
                self.current_move_indices[line_idx] -= 1
                self.has_navigated[line_idx] = True
                self._update_board_position()
                return True
                
        elif direction == 'start':
            self.current_move_indices[line_idx] = 0
            self.has_navigated[line_idx] = True
            self._update_board_position()
            return True
            
        elif direction == 'end':
            line_positions = self.analysis_positions[line_idx]
            if line_positions:
                self.current_move_indices[line_idx] = len(line_positions) - 1
                self.has_navigated[line_idx] = True
                self._update_board_position()
                return True
                
        elif direction == 'line_up':
            if self.selected_line_index > 0:
                self.selected_line_index -= 1
                self._update_board_position()
                return True
                
        elif direction == 'line_down':
            if self.selected_line_index < len(self.analysis_lines) - 1:
                self.selected_line_index += 1
                self._update_board_position()
                return True
        
        return False
    
    def select_line(self, line_index: int) -> bool:
        """
        Select a specific analysis line.
        
        Args:
            line_index: Index of line to select
            
        Returns:
            True if line was selected
        """
        if 0 <= line_index < len(self.analysis_lines):
            self.selected_line_index = line_index
            self._update_board_position()
            return True
        return False
    
    def _update_board_position(self):
        """Update board position based on current navigation state"""
        if (self.selected_line_index >= 0 and 
            self.selected_line_index < len(self.analysis_positions) and
            self.update_board_callback):
            
            line_idx = self.selected_line_index
            move_idx = self.current_move_indices[line_idx]
            
            if (self.analysis_positions[line_idx] and 
                move_idx < len(self.analysis_positions[line_idx])):
                
                position = self.analysis_positions[line_idx][move_idx]
                self.update_board_callback(position)
        
        # Update display
        if self.update_display_callback:
            self.update_display_callback()
    
    def get_analysis_display_data(self) -> Tuple[List[str], int, List[int], List[bool]]:
        """
        Get data for updating analysis display.
        
        Returns:
            Tuple of (lines, selected_line_index, current_move_indices, has_navigated)
        """
        return (self.analysis_lines, self.selected_line_index, 
                self.current_move_indices, self.has_navigated)
    
    def restore_original_position(self) -> bool:
        """
        Restore the original position before analysis.
        
        Returns:
            True if original position was restored
        """
        if self.original_position and self.update_board_callback:
            self.update_board_callback(self.original_position.copy())
            return True
        return False
    
    def has_analysis(self) -> bool:
        """Check if there are analysis results available"""
        return len(self.analysis_lines) > 0
    
    def clear_analysis(self):
        """Clear all analysis data"""
        self.analysis_lines = []
        self.analysis_positions = [[], [], []]
        self.selected_line_index = -1
        self.current_move_indices = [0, 0, 0]
        self.has_navigated = [False, False, False]
        self.original_position = None 