import pytest
import common


def test_simple():
  cell = "x = df[df['Pclass'] == 1]['Sex']"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr





def test_complex_caller():
  cell = """
class ComplexCaller:
  def __init__(self, bar):
    self.bar = bar

def foo():
  global df
  return ComplexCaller(bar=df)

x = foo().bar[df['Pclass'] == 1]['Sex']
"""

  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr





def test_no_match():
  cell = """
cols = ['Sex', 'Survived']
x = df[df['Pclass'] == 1][cols]
"""


  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr