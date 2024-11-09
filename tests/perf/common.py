import os
os.environ["_IREWR_USE_AS_LIB"] = "True"
import dias.rewriter
import time
import types

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(FILE_DIR, "../datasets")

def exec_code(code, mod):
  # Note: Exec's update the module's dict
  exec(compile(code, mod.__name__, 'exec'), mod.__dict__)

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