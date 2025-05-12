def _REWR_index_contains(index, s):
  if index.dtype == np.int64:
    try:
      i = int(s)
      return len(index.loc[i]) > 0
    except:
      return False
  else:
    return index.astype(str).str.contains(s).any()


for col in df.columns:
  if (df[col].astype(str).str.contains('%').any() or _REWR_index_contains(
      df[col].index, '%') if type(df) == pd.DataFrame else '%' in df[col].
      to_string()) or (df[col].astype(str).str.contains(',').any() or
      _REWR_index_contains(df[col].index, ',') if type(df) == pd.DataFrame else
      ',' in df[col].to_string()):
    pass

