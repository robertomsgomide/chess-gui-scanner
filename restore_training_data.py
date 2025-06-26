"""
restore_training_data.py

Unified helper script to manage all chess training data. This allows:
- Managing CNN classifier training data (piece recognition)
- Managing board orientation training data  
- Viewing statistics and removing recent corrections
- Clearing data and retraining models

This is useful when incorrect data was saved during training.

Usage:
    python restore_training_data.py
"""

import os
import sys
import sqlite3
import shutil
from datetime import datetime
from typing import Optional, Tuple, List
import json

from BoardAnalyzer import BoardAnalyzer


def backup_model_files():
    """Backup current model files before making changes"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"model_backup_{timestamp}"
    
    files_to_backup = [
        "chess_classifier.pt",
        "chess_classifier_best.pt", 
        "chess_classifier_metrics.json"
    ]
    
    backed_up_files = []
    
    for filename in files_to_backup:
        if os.path.exists(filename):
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            backup_path = os.path.join(backup_dir, filename)
            shutil.copy2(filename, backup_path)
            backed_up_files.append(filename)
    
    if backed_up_files:
        print(f"Backed up model files to: {backup_dir}")
        print(f"Files: {', '.join(backed_up_files)}")
        return backup_dir
    else:
        print("No model files found to backup")
        return None


def get_database_stats(db_path: str) -> dict:
    """Get current database statistics"""
    if not os.path.exists(db_path):
        return {"error": "Database not found"}
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        stats = {"tables": tables}
        
        # CNN classifier data
        if 'samples' in tables:
            cursor.execute("SELECT COUNT(*) FROM samples WHERE split = 'train'")
            stats["train_samples"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM samples WHERE split = 'val'")
            stats["val_samples"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT position_hash) FROM samples WHERE position_hash IS NOT NULL")
            stats["unique_positions"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(timestamp) FROM samples")
            stats["latest_cnn_timestamp"] = cursor.fetchone()[0]
        
        # Board analyzer data
        if 'orientation_training' in tables:
            cursor.execute("SELECT COUNT(*) FROM orientation_training")
            stats["orientation_total"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM orientation_training WHERE is_flipped = 0")
            stats["orientation_normal"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM orientation_training WHERE is_flipped = 1") 
            stats["orientation_flipped"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(timestamp) FROM orientation_training")
            stats["latest_orientation_timestamp"] = cursor.fetchone()[0]
        
        return stats


def print_cnn_stats(stats: dict):
    """Print CNN classifier statistics"""
    print("CNN Classifier Data:")
    if 'train_samples' in stats:
        print(f"   Training samples: {stats['train_samples']}")
        print(f"   Validation samples: {stats['val_samples']}")
        print(f"   Total samples: {stats['train_samples'] + stats['val_samples']}")
        print(f"   Unique positions: {stats['unique_positions']}")
        if stats['latest_cnn_timestamp']:
            print(f"   Latest entry: {stats['latest_cnn_timestamp']}")
    else:
        print("   No CNN training data found")


def print_orientation_stats(analyzer: BoardAnalyzer):
    """Print orientation training statistics"""
    stats = analyzer.get_orientation_stats()
    
    print("Board Orientation Data:")
    print(f"   Total examples: {stats['total']}")
    print(f"   Normal orientation: {stats['normal_count']}")
    print(f"   Flipped orientation: {stats['flipped_count']}")
    
    if stats['recent_entries']:
        print("   Recent entries:")
        for i, (timestamp, is_flipped) in enumerate(stats['recent_entries'][:3]):
            orientation = "Flipped" if is_flipped else "Normal"
            print(f"     {i+1}. {orientation} - {timestamp}")


def find_most_recent_cnn_position(db_path: str) -> Optional[Tuple[str, str, int]]:
    """Find the most recently added CNN position and its correction count"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT position_hash, MAX(timestamp) as latest_time
            FROM samples 
            WHERE position_hash IS NOT NULL 
            GROUP BY position_hash 
            ORDER BY latest_time DESC 
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if not result:
            return None
        
        position_hash, timestamp = result
        
        cursor.execute("SELECT COUNT(*) FROM samples WHERE position_hash = ?", (position_hash,))
        correction_count = cursor.fetchone()[0]
        
        return position_hash, timestamp, correction_count


def get_cnn_position_details(db_path: str, position_hash: str) -> dict:
    """Get detailed information about a specific CNN position"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT split, label_name, COUNT(*) as count
            FROM samples 
            WHERE position_hash = ? 
            GROUP BY split, label_name
            ORDER BY split, label_name
        """)
        
        samples = cursor.fetchall()
        
        cursor.execute("""
            SELECT MIN(timestamp) as first_time, MAX(timestamp) as last_time
            FROM samples 
            WHERE position_hash = ?
        """, (position_hash,))
        
        first_time, last_time = cursor.fetchone()
        
        return {
            "samples": samples,
            "first_timestamp": first_time,
            "last_timestamp": last_time
        }


def remove_cnn_position_data(db_path: str, position_hash: str) -> int:
    """Remove all CNN samples for a specific position"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM samples WHERE position_hash = ?", (position_hash,))
        count_before = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM samples WHERE position_hash = ?", (position_hash,))
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM samples WHERE position_hash = ?", (position_hash,))
        count_after = cursor.fetchone()[0]
        
        return count_before - count_after


def retrain_cnn_model():
    """Retrain the CNN model after removing data"""
    print("\nWould you like to retrain the CNN model with the cleaned data?")
    response = input("   Retrain CNN model? (y/n): ").lower().strip()
    
    if response == 'y':
        try:
            print("Retraining CNN model...")
            from CNNClassifier import CNNClassifier
            
            classifier = CNNClassifier()
            train_count = classifier.db.get_sample_count('train')
            
            if train_count < 8:
                print(f"Not enough training data ({train_count} samples). Model not retrained.")
                return False
            
            classifier._execute_training()
            print("CNN model retrained successfully")
            return True
            
        except Exception as e:
            print(f"Error during CNN retraining: {e}")
            return False
    else:
        print("CNN model not retrained.")
        return False


def manage_cnn_data(db_path: str):
    """Manage CNN classifier training data"""
    print("\n" + "="*50)
    print("CNN CLASSIFIER DATA MANAGEMENT")
    print("="*50)
    
    # Find most recent position
    recent_position = find_most_recent_cnn_position(db_path)
    
    if not recent_position:
        print("No CNN position data found to remove.")
        return
    
    position_hash, timestamp, correction_count = recent_position
    
    print(f"Most Recent CNN Position:")
    print(f"   Position Hash: {position_hash[:12]}...")
    print(f"   Timestamp: {timestamp}")
    print(f"   Corrections: {correction_count} samples")
    
    # Get detailed breakdown
    details = get_cnn_position_details(db_path, position_hash)
    print(f"\nCorrection Breakdown:")
    for split, label, count in details['samples']:
        print(f"   {split.title()}: {label} × {count}")
    
    # Confirm removal
    print(f"\nThis will remove {correction_count} CNN training samples.")
    response = input("Proceed with removal? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Operation cancelled.")
        return
    
    # Remove the data
    print(f"\nRemoving CNN corrections for position {position_hash[:12]}...")
    removed_count = remove_cnn_position_data(db_path, position_hash)
    
    if removed_count > 0:
        print(f"Successfully removed {removed_count} CNN samples")
        retrain_cnn_model()
    else:
        print("No CNN data was removed.")


def manage_orientation_data():
    """Manage board orientation training data"""
    print("\n" + "="*50)
    print("BOARD ORIENTATION DATA MANAGEMENT")
    print("="*50)
    
    analyzer = BoardAnalyzer()
    
    if analyzer.get_training_data_count() == 0:
        print("No orientation training data found.")
        return
    
    print("Options:")
    print("1. Remove most recent orientation example")
    print("2. Remove multiple recent examples")
    print("3. Clear all orientation data")
    print("4. Back to main menu")
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                removed = analyzer.remove_recent_orientation_data(1)
                if removed > 0:
                    print("Removed 1 orientation example")
                    print_orientation_stats(analyzer)
                else:
                    print("No orientation data was removed")
                    
            elif choice == '2':
                count = input("How many recent examples to remove? ").strip()
                try:
                    count = int(count)
                    if count <= 0:
                        print("Please enter a positive number")
                        continue
                        
                    removed = analyzer.remove_recent_orientation_data(count)
                    print(f"✅ Removed {removed} orientation examples")
                    print_orientation_stats(analyzer)
                    
                except ValueError:
                    print("Invalid number")
                    
            elif choice == '3':
                print("\nThis will remove ALL orientation training data!")
                confirm = input("Are you sure? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    analyzer.clear_orientation_data()
                    print("Cleared all orientation training data")
                    print_orientation_stats(analyzer)
                else:
                    print("Operation cancelled")
                    
            elif choice == '4':
                return
                
            else:
                print("Invalid option. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return


def main():
    """Main function for unified training data management"""
    print("Chess Training Data Management Tool")
    print("=" * 50)
    
    db_path = "chess_training_data.db"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("   Make sure you're running this from the correct directory.")
        return 1
    
    # Get and display current stats
    print("Current Database Statistics:")
    stats = get_database_stats(db_path)
    if "error" in stats:
        print(f"{stats['error']}")
        return 1
    
    print_cnn_stats(stats)
    
    analyzer = BoardAnalyzer()
    print_orientation_stats(analyzer)
    
    # Check if there's any data
    has_cnn_data = stats.get('train_samples', 0) + stats.get('val_samples', 0) > 0
    has_orientation_data = analyzer.get_training_data_count() > 0
    
    if not has_cnn_data and not has_orientation_data:
        print("\nNo training data found. Nothing to manage.")
        return 0
    
    # Main menu
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Manage CNN classifier data (piece recognition)")
        print("2. Manage board orientation data")
        print("3. Clear ALL training data")
        print("4. Refresh statistics")
        print("5. Exit")
        
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                if has_cnn_data:
                    manage_cnn_data(db_path)
                else:
                    print("No CNN training data found.")
                    
            elif choice == '2':
                if has_orientation_data:
                    manage_orientation_data()
                else:
                    print("No orientation training data found.")
                    
            elif choice == '3':
                print("\nThis will remove ALL training data from the database!")
                print("   This includes both CNN classifier and orientation data.")
                confirm = input("Are you absolutely sure? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    backup_dir = backup_model_files()
                    
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM samples")
                        cursor.execute("DELETE FROM orientation_training")
                        cursor.execute("DELETE FROM pending_positions")
                        cursor.execute("DELETE FROM pending_corrections")
                        conn.commit()
                    
                    print("Cleared all training data")
                    if backup_dir:
                        print(f"   Model backup saved in: {backup_dir}")
                    
                    has_cnn_data = False
                    has_orientation_data = False
                else:
                    print("Operation cancelled")
                    
            elif choice == '4':
                # Refresh stats
                stats = get_database_stats(db_path)
                print("\nUpdated Database Statistics:")
                print_cnn_stats(stats)
                print_orientation_stats(analyzer)
                
                has_cnn_data = stats.get('train_samples', 0) + stats.get('val_samples', 0) > 0
                has_orientation_data = analyzer.get_training_data_count() > 0
                
            elif choice == '5':
                print("\nGoodbye!")
                return 0
                
            else:
                print("Invalid option. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return 1
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1) 