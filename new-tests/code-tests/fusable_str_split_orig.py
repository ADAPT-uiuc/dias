df['original_price'] = df['original_price'].str.split('$')
df['original_price'] = df['original_price'].str[1]