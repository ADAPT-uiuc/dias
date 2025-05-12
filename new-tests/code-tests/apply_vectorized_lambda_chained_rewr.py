def _REWR_apply_vec(df):
  if type(df) == pd.DataFrame:
    return pd.Series(np.select([(df['Math_PassStatus'] == 'F') | (df[
        'Reading_PassStatus'] == 'F') | (df['Writing_PassStatus'] == 'F')],
        ['F'], default='P'))
  else:
    return df.apply(lambda x: 'F' if x['Math_PassStatus'] == 'F' or x[
        'Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else
        'P', axis=1)


our = _REWR_apply_vec(df).str.replace('F', 'B')

