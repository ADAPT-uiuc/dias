import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import types

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

def boiler(cell, setup_state=None):
  rewr_dias = dias.rewriter.rewrite_ast_from_source(cell)[0]
  print('rewr_dias:', rewr_dias)
  
  if setup_state is None:
    setup_state = SETUP_STATE

  # We use different modules and so different states
  mod_orig = types.ModuleType('Orig')
  exec_code(setup_state, mod_orig)
  exec_code(cell, mod_orig)
  
  mod_rewr = types.ModuleType('Rewr')
  exec_code(setup_state, mod_rewr)
  exec_code(rewr_dias, mod_rewr)

  return mod_orig, mod_rewr