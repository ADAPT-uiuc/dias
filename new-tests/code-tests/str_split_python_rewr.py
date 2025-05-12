if type(course['course_code']) != pd.Series:
  course[['our_course_unit', 'our_course_num']] = course['course_code'
      ].str.split(' ', expand=True)
else:
  _REWR_targ_0 = []
  _REWR_targ_1 = []
  _REWR_ls = course['course_code'].tolist()
  for _REWR_s in _REWR_ls:
    _REWR_spl = _REWR_s.split(' ')
    _REWR_targ_0.append(_REWR_spl[0])
    _REWR_targ_1.append(_REWR_spl[1] if len(_REWR_spl) > 1 else None)
  course['our_course_unit'] = _REWR_targ_0
  course['our_course_num'] = _REWR_targ_1

