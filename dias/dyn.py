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

def subseq_orig(df, pred, col):
  return df[pred][col]

def subseq(df, pred, col):
  if not isinstance(df, pd.DataFrame):
    return subseq_orig(df, pred, col)
  
  if not isinstance(pred, pd.Series):
    return subseq_orig(df, pred, col)

  if not pred.dtype == 'bool':
    return subseq_orig(df, pred, col)
  
  return df.loc[pred, col]


def replace_to_map_orig(ser, map_):
  return ser.replace(map_)

def replace_to_map(ser, map_):
  if not isinstance(ser, pd.Series):
    return replace_to_map_orig(ser, map_)
  
  if not isinstance(map_, map):
    return replace_to_map_orig(ser, map_)

  default = {x: x for x in ser.drop_duplicates().values}
  default.update(map_)
  return ser.map(default)