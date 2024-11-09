import pytest

import common

MIN_SPEEDUP = 10

def test_titanic():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/titanic.csv')
df = pd.concat([df]*80_000, ignore_index=True)
"""

  cell = "df.drop(['Sex'], axis=1, inplace=True)"
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > MIN_SPEEDUP


def test_adidas():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/adidas.csv')
df = pd.concat([df]*20_000, ignore_index=True)
"""

  cell = """
Y = df['breadcrumbs']
df.drop(['breadcrumbs'], axis=1, inplace=True)
X = df
"""
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > MIN_SPEEDUP


def test_students():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/students.csv')
df = pd.concat([df]*20_000, ignore_index=True)
"""

  cell = """
df.drop(['lunch'], axis = 1, inplace=True)
"""
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > MIN_SPEEDUP