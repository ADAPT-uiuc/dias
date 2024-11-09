import pytest
import common


def test_simple():
  cell = "x = df['Sex'].replace({'male': 0, 'female': 1})"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr

def test_default():
  setup_state = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/students.csv')
"""

  cell = """
x = df['parental level of education'].replace({'high school': 'one', 'some college': 'two', "master's degree": 'three', "bachelor's degree": "four"})
"""
  mod_orig, mod_rewr = common.boiler(cell, setup_state=setup_state)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr
