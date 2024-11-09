import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest


def test_simple():
  cell = "df[s]['col']"

  rewr_corr = "dias.dyn.subseq(df=df, pred=s, col='col')"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()

def test_simple2():
  cell = "df[df['a'] == 1]['col']"

  rewr_corr = "dias.dyn.subseq(df=df, pred=df['a'] == 1, col='col')"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()

def test_complex_caller():
  cell = "foo().bar[df['a'] == 1]['col']"

  rewr_corr = "dias.dyn.subseq(df=foo().bar, pred=df['a'] == 1, col='col')"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()


def test_mult_matches():
  cell = """
df[x1]['col1']
print('test')
df[x2]['col2']
"""
  
  rewr_corr = """
dias.dyn.subseq(df=df, pred=x1, col='col1')
print('test')
dias.dyn.subseq(df=df, pred=x2, col='col2')
"""
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()


def test_no_match():
  cell = "df[df['a'] == 1][x]"
  
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == cell.strip()


def test_no_match2():
  cell = "df[df['a'] == 1][['col1', 'col2']]"
  
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == cell.strip()