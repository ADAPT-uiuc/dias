from metap import MetaP

mp = MetaP(filename="patt_matcher_mp.py")
# mp.log_returns(include_fname=True)
mp.compile()
mp.dump(filename="patt_matcher.py")