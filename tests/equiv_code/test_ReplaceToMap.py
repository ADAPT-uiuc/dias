import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest


def test_simple():
  cell = "df['Sex'].replace({'male': 0, 'female': 1})"

  rewr_corr = "dias.dyn.replace_to_map(ser=df['Sex'], map_={'male': 0, 'female': 1})"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()

def test_simple2():
  cell = "df['Sex'].replace(mymap())"

  rewr_corr = "dias.dyn.replace_to_map(ser=df['Sex'], map_=mymap())"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_no_match():
  cell = "df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)"

  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == cell.strip()