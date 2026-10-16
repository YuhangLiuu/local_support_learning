"""Entry for Linux instance environment recovery (use the instance base Python)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduction.environment_bootstrap import main
if __name__=='__main__':main()
