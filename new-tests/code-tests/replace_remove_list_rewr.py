dias.rewriter.replace_remove_list(responses_df_2018, ['Kaggle Learn'],
    'Kaggle Learn Courses', 'Kaggle Learn', 'Kaggle Learn Courses', lambda
    _DIAS_x1, _DIAS_x2: responses_df_2018.replace(_DIAS_x1, _DIAS_x2,
    inplace=True))