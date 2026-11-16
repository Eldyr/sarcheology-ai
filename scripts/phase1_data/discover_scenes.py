
import sys
from pathlib import Path

# Locate the project root and source package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.ingestion.catalog import run_discovery


if __name__ == "__main__":
    run_discovery()
