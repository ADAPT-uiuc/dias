import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest


def test_simple():
  cell = "df.drop('col', axis=1, inplace=True)"
  
  rewr_corr = "dias.dyn.drop_to_pop(df=df, col='col')"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()

def test_simple2():
  cell = "df.drop(['col'], axis=1, inplace=True)"
  
  rewr_corr = "dias.dyn.drop_to_pop(df=df, col=['col'])"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()

def test_complex():
  cell = """
Y = df['Sex']
df.drop(['Sex'], axis=1, inplace=True)
X = df
"""
  
  rewr_corr = """
Y = df['Sex']
dias.dyn.drop_to_pop(df=df, col=['Sex'])
X = df
"""

  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()


def test_mult_matches():
  cell = """
df.drop('col1', axis=1, inplace=True)
print('test')
df.drop('col2', axis=1, inplace=True)
"""
  
  rewr_corr = """
dias.dyn.drop_to_pop(df=df, col='col1')
print('test')
dias.dyn.drop_to_pop(df=df, col='col2')
"""
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()

def test_complex_caller():
  cell = "foo().bar().drop('col', axis=1, inplace=True)"
  
  rewr_corr = "dias.dyn.drop_to_pop(df=foo().bar(), col='col')"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()


def test_no_match():
  cell = "df.drop('col', axis=0, inplace=True)"
  
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == cell.strip()


def test_no_match2():
  cell = "df.drop('col', axis=1)"
  
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == cell.strip()