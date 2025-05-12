our = dias.rewriter.substr_search_apply(ser=pd.read_csv('titanic.csv')[
    'Name'], needle='G', orig=lambda _DIAS_x: _DIAS_x.apply(lambda s: 'G' in s)
    )

