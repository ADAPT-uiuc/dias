import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import pytest
import types
import time

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(FILE_DIR, "../datasets")

def exec_code(code, mod):
  # Note: Exec's update the module's dict
  exec(compile(code, mod.__name__, 'exec'), mod.__dict__)

MIN_SPEEDUP = 10


def boiler(cell, setup):
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]

  # We use different modules and so different states
  mod_orig = types.ModuleType('Orig')
  exec_code(setup, mod_orig)
  orig_start = time.perf_counter_ns()
  exec_code(cell, mod_orig)
  orig_end = time.perf_counter_ns()
  orig_ns = orig_end - orig_start

  mod_rewr = types.ModuleType('Rewr')
  exec_code(setup, mod_rewr)
  rewr_start = time.perf_counter_ns()
  exec_code(rewr_dias, mod_rewr)
  rewr_end = time.perf_counter_ns()
  rewr_ns = rewr_end - rewr_start

  del mod_orig
  del mod_rewr

  return orig_ns, rewr_ns

def test_titanic():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{DATASETS_PATH}/titanic.csv')
df = pd.concat([df]*80_000, ignore_index=True)
"""

  cell = "df.drop(['Sex'], axis=1, inplace=True)"
  orig_ns, rewr_ns = boiler(cell, setup)

  assert (orig_ns / rewr_ns) > MIN_SPEEDUP


def test_adidas():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{DATASETS_PATH}/adidas.csv')
df = pd.concat([df]*20_000, ignore_index=True)
"""

  cell = """
Y = df['breadcrumbs']
df.drop(['breadcrumbs'], axis=1, inplace=True)
X = df
"""
  orig_ns, rewr_ns = boiler(cell, setup)

  assert (orig_ns / rewr_ns) > MIN_SPEEDUP


def test_students():
  setup = f"""
import pandas as pd
import dias.dyn

df = pd.read_csv('{DATASETS_PATH}/students.csv')
df = pd.concat([df]*20_000, ignore_index=True)
"""

  cell = """
df.drop(['lunch'], axis = 1, inplace=True)
"""
  orig_ns, rewr_ns = boiler(cell, setup)

  assert (orig_ns / rewr_ns) > MIN_SPEEDUP