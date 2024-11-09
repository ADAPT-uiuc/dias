import pytest

import common

def test_titanic():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/titanic.csv')
df = pd.concat([df]*80_000, ignore_index=True)
"""

  cell = "orig = df[df['Pclass'] == 1]['Age']"
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > 2


def test_essays():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/essays.csv')
df = pd.concat([df]*30_000, ignore_index=True)
"""

  cell = "df[df['cohesion'] == 3.5]['syntax']"
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > 2


def test_adidas():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/adidas.csv')
df = pd.concat([df]*30_000, ignore_index=True)
"""

  cell = "df[df['selling_price'] > 40]['reviews_count']"
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > 2


def test_aggfunc():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{common.DATASETS_PATH}/titanic.csv')
df = pd.concat([df]*50_000, ignore_index=True)
"""

  cell = "df[df['Pclass'] == 1]['Age'].mean()"
  orig_ns, rewr_ns = common.boiler(cell, setup)

  assert (orig_ns / rewr_ns) > 2