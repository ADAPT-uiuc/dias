import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest


def test_df_simple():
  cell = "df.sort_values(by='col').head(n=2)"

  rewr_corr = "dias.dyn.sort_head_df(df=df, by='col', asc=True, n=2)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_df_by():
  cell = "df.sort_values(by=['col', 'col2']).head(n=2)"

  rewr_corr = "dias.dyn.sort_head_df(df=df, by=['col', 'col2'], asc=True, n=2)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_df_by_asc():
  cell = "df.sort_values(by='col', ascending=False).head(n=2)"

  rewr_corr = "dias.dyn.sort_head_df(df=df, by='col', asc=False, n=2)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_df_head1():
  cell = "df.sort_values(by='col', ascending=False).head(2)"

  rewr_corr = "dias.dyn.sort_head_df(df=df, by='col', asc=False, n=2)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_df_head2():
  cell = "df.sort_values(by='col', ascending=False).head()"

  rewr_corr = "dias.dyn.sort_head_df(df=df, by='col', asc=False, n=5)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()





def test_ser_simple():
  cell = "ser.sort_values().head(n=2)"

  rewr_corr = "dias.dyn.sort_head_ser(ser=ser, asc=True, n=2)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_ser_asc():
  cell = "ser.sort_values(ascending=False).head(n=2)"

  rewr_corr = "dias.dyn.sort_head_ser(ser=ser, asc=False, n=2)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()

def test_ser_head():
  cell = "ser.sort_values().head()"

  rewr_corr = "dias.dyn.sort_head_ser(ser=ser, asc=True, n=5)"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()
  
def test_ser_complex():
  cell = "bar().sort_values(ascending=False).head(foo())"

  rewr_corr = "dias.dyn.sort_head_ser(ser=bar(), asc=False, n=foo())"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  assert rewr_dias.strip() == rewr_corr.strip()