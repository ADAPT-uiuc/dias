import common
import pytest

def test_simple():
  cell = "x = df['Name'].unique()"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert (actual == expected).all()

  del mod_orig
  del mod_rewr


def test_simple2():
  setup_state = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/essays.csv')
"""

  cell = "x = df['text_id'].unique()"
  mod_orig, mod_rewr = common.boiler(cell, setup_state=setup_state)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert (actual == expected).all()

  del mod_orig
  del mod_rewr
