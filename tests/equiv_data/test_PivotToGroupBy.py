import common
import pytest

def test_simple():
  cell = "x = pd.pivot_table(df, index = 'Survived', values = 'Pclass', aggfunc = [np.mean, np.max, np.size])"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr


def test_simple2():
  cell = "x = pd.pivot_table(df, index = 'Survived', values = 'Pclass', aggfunc = [len, np.std])"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr


def test_simple3():
  setup_state = f"""
import pandas as pd
import numpy as np
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/students.csv')
"""

  cell = "x = pd.pivot_table(df, index = 'race/ethnicity', values = 'math score', aggfunc = ['min', np.sum])"
  mod_orig, mod_rewr = common.boiler(cell, setup_state=setup_state)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr


def test_simple4():
  cell = "x = pd.pivot_table(df, index = 'Survived', values = 'Pclass', aggfunc = np.std)"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr
  

def test_mult_values():
  cell = "x = pd.pivot_table(df, index = 'Survived', values = ['Pclass', 'Age'], aggfunc = ['min', 'sum'])"
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr


def test_def_values():
  cell = """
df.drop(['Sex', 'Name'], axis=1, inplace=True)
x = pd.pivot_table(df, index = 'Survived', aggfunc = ['min', 'sum'])
"""
  mod_orig, mod_rewr = common.boiler(cell)

  actual = mod_orig.__dict__['x']
  expected = mod_rewr.__dict__['x']

  assert actual.equals(expected)

  del mod_orig
  del mod_rewr