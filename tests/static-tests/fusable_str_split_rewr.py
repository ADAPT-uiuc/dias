if type(df['original_price']) != pd.Series:
  _REWR_res = df['original_price'].str.split('$')
else:
  _REWR_targ_0 = []
  _REWR_ls = df['original_price'].tolist()
  for _REWR_s in _REWR_ls:
    _REWR_spl = _REWR_s.split('$')
    _REWR_targ_0.append(_REWR_spl[1] if len(_REWR_spl) > 1 else None)
  _REWR_res = _REWR_targ_0
df['original_price'] = _REWR_res

