dias.rewriter.replace_remove_list(popul_df['name'], ['OHare'], "O'Hare",
    'OHare', "O'Hare", lambda _DIAS_x1, _DIAS_x2: popul_df['name'].replace(
    _DIAS_x1, _DIAS_x2, inplace=True))
