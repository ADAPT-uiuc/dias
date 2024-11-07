import pandas as pd

def drop_to_pop_orig(df, col):
  return df.drop(col, axis=1, inplace=True)

def drop_to_pop(df, col):
  if not isinstance(df, pd.DataFrame):
    return drop_to_pop_orig(df, col)
  
  col_str = None
  if isinstance(col, str):
    col_str = col
  elif isinstance(col, list) and len(col) == 1 and isinstance(col[0], str):
    col_str = col[0]
  else:
    return drop_to_pop_orig(df, col)
  
  df.pop(col_str)
  return None