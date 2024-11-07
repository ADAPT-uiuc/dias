import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest
import types
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(FILE_DIR, "../datasets")

def exec_code(code, mod):
  # Note: Exec's update the module's dict
  exec(compile(code, mod.__name__, 'exec'), mod.__dict__)

SETUP_STATE = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{DATASETS_PATH}/titanic.csv')
"""

def boiler(cell):
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # We use different modules and so different states
  mod_orig = types.ModuleType('Orig')
  exec_code(SETUP_STATE, mod_orig)
  exec_code(cell, mod_orig)
  
  mod_rewr = types.ModuleType('Rewr')
  exec_code(SETUP_STATE, mod_rewr)
  exec_code(rewr_dias, mod_rewr)

  return mod_orig, mod_rewr


def test_simple():
  cell = "df.drop(['Sex'], axis=1, inplace=True)"
  mod_orig, mod_rewr = boiler(cell)

  actual = mod_orig.__dict__['df']
  expected = mod_rewr.__dict__['df']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr




def test_complex():
  cell = """
Y = df['Sex']
df.drop(['Sex'], axis=1, inplace=True)
X = df
"""

  mod_orig, mod_rewr = boiler(cell)

  actual_X = mod_orig.__dict__['X']
  expected_X = mod_rewr.__dict__['X']

  actual_Y = mod_orig.__dict__['Y']
  expected_Y = mod_rewr.__dict__['Y']

  assert actual_X.equals(expected_X)
  assert actual_Y.equals(expected_Y)

  del mod_orig
  del mod_rewr





def test_mult_matches():
  cell = """
df.drop('Sex', axis=1, inplace=True)
print('test')
df.drop('Pclass', axis=1, inplace=True)
"""

  mod_orig, mod_rewr = boiler(cell)

  actual = mod_orig.__dict__['df']
  expected = mod_rewr.__dict__['df']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr