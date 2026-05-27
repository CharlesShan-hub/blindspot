
import sys
from PySide6.QtWidgets import QApplication

try:
    # Attempt to import the modified module
    from scripts import source_view_gui2
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except SyntaxError as e:
    print(f"Syntax error: {e}")
