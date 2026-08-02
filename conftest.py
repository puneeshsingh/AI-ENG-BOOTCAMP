import sys
from pathlib import Path

# Ensure `import main` resolves regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent))
