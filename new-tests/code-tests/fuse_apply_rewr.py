def fused_apply(ser):
  if type(ser) == pd.Series:
    res = []
    ls = ser.tolist()
    for it in ls:
      left = literal_eval(it)
      right = [i['name'] for i in left] if isinstance(left, list) else []
      res.append(right)
    return pd.Series(res)
  else:
    return ser.apply(literal_eval).apply(lambda x: [i['name'] for i in left
        ] if isinstance(left, list) else [])


md['genres'] = fused_apply(md['genres'].fillna('[]')).head()

