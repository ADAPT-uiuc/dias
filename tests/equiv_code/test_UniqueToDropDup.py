import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest


def test_simple():
  cell = "df['Name'].unique()"

  rewr_corr = "dias.dyn.unique_to_drop_dup(ser=df['Name'])"
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # To be consistent, do: actual == expected
  assert rewr_dias.strip() == rewr_corr.strip()