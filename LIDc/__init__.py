
import sys
from pathlib import Path
p = str((Path(__file__).parent / '../build/src').resolve().absolute())
sys.path.append(p)

from .liblid import *

