import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest


def test_simple():
  cell = "pd.pivot_table(df, index = 'Survived', values = 'Pclass', aggfunc = [np.mean, np.max, np.size])"

  rewr_corr = """
dias.dyn.pivot_to_gby(df=df, index='Survived', values='Pclass', aggs=[np.
    mean, np.max, np.size])
"""
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]
  
  assert rewr_dias.strip() == rewr_corr.strip()

def test_def_aggfunc():
  cell = "pd.pivot_table(df, index = 'Survived', values = 'Pclass')"

  rewr_corr = """
dias.dyn.pivot_to_gby(df=df, index='Survived', values='Pclass', aggs='mean')
"""
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]
  
  assert rewr_dias.strip() == rewr_corr.strip()
  


def test_no_match():
  cell = """
x = pd.pivot_table(df, index='Survived', aggfunc=['min', 'sum'])
"""
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == cell.strip()