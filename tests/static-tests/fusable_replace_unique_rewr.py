if type(df['rating']) == pd.Series and isinstance(ratings_ages, dict):
  _REWR_res = []
  _REWR_uniq = set()
  for _REWR_s in df['rating'].tolist():
    try:
      _REWR_s = ratings_ages[_REWR_s]
    except KeyError:
      pass
    _REWR_res.append(_REWR_s)
    _REWR_uniq.add(_REWR_s)
  df['target_ages'] = _REWR_res
  _REWR_temp_res = _REWR_uniq
else:
  df['target_ages'] = df['rating'].replace(ratings_ages)
  _REWR_temp_res = df['target_ages'].unique()
_REWR_temp_res

