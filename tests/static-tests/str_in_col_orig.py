for col in df.columns:
  if '%' in df[col].to_string() or ',' in df[col].to_string():
    pass