from metap import MetaP
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

input_f = os.path.join(FILE_DIR, "patt_matcher_mp.py")
output_f = os.path.join(FILE_DIR, "patt_matcher.py")
mp = MetaP(filename=input_f)
mp.compile()
# mp.log_calls(range=[566])
# mp.log_returns(range=[(501, 560)])
mp.dump(filename=output_f)