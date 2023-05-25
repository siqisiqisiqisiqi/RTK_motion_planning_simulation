import pathlib
import sys

path = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(1, path)
from config import *
print(DT)

from utils.QuarticPolynomial import QuarticPolynomial
