from metap import MetaP
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

input_f = os.path.join(FILE_DIR, "patt_matcher_mp.py")
output_f = os.path.join(FILE_DIR, "patt_matcher.py")
mp = MetaP(filename=input_f)
mp.compile()
# mp.log_returns(include_fname=True)
mp.dump(filename=output_f)