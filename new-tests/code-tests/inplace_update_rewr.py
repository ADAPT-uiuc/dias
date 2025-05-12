_DIAS_ser = df['A']
if type(_DIAS_ser) != pd.Series:
  df['A'] = df['A'].fillna(5)
else:
  df['A'].fillna(5, inplace=True)

