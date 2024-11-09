from IPython import get_ipython, InteractiveShell
from IPython.display import display, Markdown
import ast
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Literal, Callable
import types
import inspect
import json
import time
import pandas as pd
import numpy as np
import functools
import copy
from enum import Enum
import warnings

from pandas._typing import AggFuncType, Axis

### NON-DEFAULT PACKAGES: The user has to have these installed
import astor

import dias.patt_matcher as patt_matcher
import dias.nb_utils as nb_utils

############ CONFIGURATIONS ############

_IREWR_JSON_STATS=False
if "_IREWR_JSON_STATS" in os.environ and os.environ["_IREWR_JSON_STATS"] == "True":
  _IREWR_JSON_STATS=True

disabled_patts = set()
if "_IREWR_DISABLED_PATTS" in os.environ:
  disabled_patts = ast.literal_eval(os.environ["_IREWR_DISABLED_PATTS"])

############ UTILS ############

_REWR_id_counter = 0
def get_unique_id():
  global _REWR_id_counter
  res = "_REWR_tmp_" + str(_REWR_id_counter)
  _REWR_id_counter = _REWR_id_counter + 1
  return res

# Source: https://stackoverflow.com/a/39662359
def is_notebook_env() -> bool:
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    else:
      return False  # Other type (?)
  except NameError:
    return False      # Probably standard Python interpreter

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

DBG_PRINT = True
def dbg_print(*args, **kwargs):
  if DBG_PRINT:
    print(*args, **kwargs)

############ AST CONSTRUCTORS ############

def AST_name(name: str) -> ast.Name:
  return ast.Name(id=name)

def AST_assign(lhs: ast.AST, rhs: ast.expr):
  return ast.Assign(targets=[lhs], value=rhs)

def AST_var_alone(var: ast.Name):
  return ast.Expr(value=var)

def AST_sub_const(name: str, const: Any) -> ast.Subscript:
  return ast.Subscript(value=AST_name(name), slice=ast.Constant(value=const))

def AST_sub_const2(name: ast.Name, const: Any) -> ast.Subscript:
  return ast.Subscript(value=name, slice=ast.Constant(value=const))


def AST_call(func: ast.AST,
             args: Optional[List[ast.arg]]=None,
             keywords: Optional[Dict[str, ast.AST]]=None) -> ast.Call:
  if args is None:
    args = []
  
  if keywords is None:
    keywords = {}

  keywords_list = \
    [ast.keyword(arg=key, value=value) for key, value in keywords.items() if value is not None]
  
  return \
  ast.Call(
    func=func,
    args=args,
    keywords=keywords_list
  )

def AST_attr_call(called_on: ast.AST, name: str,
                  args: Optional[List[ast.arg]]=None,
                  keywords: Optional[Dict[str, ast.AST]]=None) -> ast.Call:
  return AST_call(
    func=ast.Attribute(
      value=called_on,
      attr=name
    ),
    args=args,
    keywords=keywords
  )

def AST_attr_chain(chain: str) -> ast.Attribute:
  return ast.parse(chain, mode='eval')

def AST_cmp(lhs: ast.AST, rhs: ast.AST, op: ast.AST) -> ast.Compare:
  return \
  ast.Compare(
    left=lhs,
    comparators=[rhs],
    ops=[op]
  )

def AST_if_else(cond: ast.AST, then_block: List[ast.AST], else_block: List[ast.AST]) -> ast.If:
  return \
  ast.If(
    body=then_block,
    orelse=else_block,
    test=cond
  )

def AST_iter_container(it: ast.Name, cont: ast.Name, body: List[ast.AST]) -> ast.For:
  return \
  ast.For(
    body=body,
    iter=cont,
    orelse=[],
    target=it
  )

def AST_bool_and(lhs: ast.AST, rhs: ast.AST) -> ast.BoolOp:
  return \
  ast.BoolOp(
    op=ast.And(),
    values=[lhs, rhs]
  )

def AST_bool_or(lhs: ast.AST, rhs: ast.AST) -> ast.BoolOp:
  return \
  ast.BoolOp(
    op=ast.Or(),
    values=[lhs, rhs]
  )

def AST_try_except(try_block: List[ast.stmt],
                   except_block: List[ast.stmt]) -> ast.Try:
  return \
  ast.Try(
    body=try_block,
    finalbody=[],
    handlers=[ast.ExceptHandler(body=except_block, type=AST_name("KeyError"))],
    orelse=[]
  )

def AST_function_def(name: str, args: List[str], body: List[ast.stmt]) -> ast.FunctionDef:
  arguments = ast.arguments(
    args=[ast.arg(arg=x) for x in args],
    defaults=[],
    kw_defaults=[],
    kwarg=None,
    kwonlyargs=[],
    posonlyargs=[],
    vararg=None
  )

  return \
  ast.FunctionDef(
    args=arguments,
    body=body,
    name=name,
    decorator_list=[],
    returns=None,
    type_comment=None
  )

def rewrite_ast(cell_ast: ast.Module) -> Tuple[str, Dict]:
  assert isinstance(cell_ast, ast.Module)
  matched_patts = patt_matcher.patt_match(cell_ast)

  # Stats of what pattern was applied.
  stats = dict()
  for patt in matched_patts:
    if isinstance(patt, patt_matcher.DropToPop):
      call = AST_attr_call(
        called_on=AST_attr_chain('dias.dyn'),
        name="drop_to_pop",
        keywords={'df': patt.df, 'col': patt.col}
      )
      patt.call_encl.set_enclosed_obj(call)
    elif isinstance(patt, patt_matcher.SubSeq):
      call = AST_attr_call(
        called_on=AST_attr_chain('dias.dyn'),
        name="subseq",
        keywords={'df': patt.df, 'pred': patt.pred, 'col': patt.col}
      )
      patt.sub_encl.set_enclosed_obj(call)
    elif isinstance(patt, (patt_matcher.IsTrivialDFCall,
                           patt_matcher.IsTrivialDFAttr,
                           patt_matcher.TrivialName,
                           patt_matcher.TrivialCall)):
      pass
    else:
      print(patt)
      assert False
  ### END FOR ###

  new_source = astor.to_source(cell_ast)

  return new_source, stats


# In call_rewrite(), we modify the code such that it calls rewrite(). Inside
# rewrite() we run code using ip.run_cell(). This function, however, calls the
# cell transformers, and thus call_rewrite() again. This leads to infinite
# recursion. We want to apply call_rewrite() only if the execution does _not_
# originate from Dias itself. So, we use a global to track whether we reached
# call_rewrite() from Dias.
_inside_dias = False

def rewrite_ast_from_source(cell):
  cell_ast = ast.parse(cell)
  return rewrite_ast(cell_ast)

def rewrite(verbose: str, cell: str,
            # If true, this function returns stats, otherwise
            # it returns None. 
            ret_stats: bool = False):
  global _inside_dias
  _inside_dias = True

  # TODO: Is this a good idea or we should ask it every time we want to use it?
  ipython = get_ipython()

  new_source, hit_stats = rewrite_ast_from_source(cell)
  start = time.perf_counter_ns()
  ipython.run_cell(new_source)
  end = time.perf_counter_ns()
  time_spent_in_exec = end-start
  if verbose.strip() == "verbose":
    if not len(hit_stats):
      if not is_notebook_env():
        print("### Dias did not rewrite code")
      else:
        display(Markdown("### Dias did not rewrite code"))
    else:
      if not is_notebook_env():
        print("### Dias rewrote code:")
        print(new_source)
      else:
        # This looks kind of ugly. But there are no great solutions. Probably the best alternative is pygments,
        # but the defaults are not amazing enough to be worth adding an extra dependency.
        display(Markdown(f"""
### Dias rewrote code:
<br />

```python
{new_source}
```
"""))

  # Create the JSON.
  stats = None
  if _IREWR_JSON_STATS or ret_stats:
    stats = dict()
    # NOTE: The original and modified codes might be the same but because the modified code was round-tripped
    # using astor, the strings might not be the same. If you want to test for equality, don't
    # just compare the two strings. Rather, round-trip the original source too and then test.
    stats['raw'] = cell
    stats['modified'] = new_source
    stats['patts-hit'] = hit_stats
    # In ns.
    stats['rewritten-exec-time'] = nb_utils.ns_to_ms(time_spent_in_exec)

  # TODO: _IREWR_JSON_STATS should go away now that we can return the stats
  # from this function and we don't have to print, the parse and all this
  # hackery. But this requires changing the testing infra too.
  if _IREWR_JSON_STATS:
    eprint("[IREWRITE JSON]")
    dumped_stats = json.dumps(stats, indent=2)
    eprint(dumped_stats)
    eprint("[IREWRITE END JSON]")

  _inside_dias = False
  return (stats if ret_stats else None)

def call_rewrite(lines: List[str]):
  cell = ''.join(lines)
  # IMPORTANT: This second check can be very slow but we need a lot of code.
  if _inside_dias or "dias.rewriter.rewrite" in cell:
    return lines

  # Skip magic functions (i.e., find the line that doesn't start with %%)
  save_idx = -1
  for idx, ln in enumerate(lines):
    if not ln.strip().startswith('%%'):
      save_idx = idx
      break
  assert save_idx != -1
  assert save_idx < len(lines)
  # Check if the line right after the magics has the DIAS_VERBOSE or DIAS_DISABLE
  verbose = ""
  disable = False
  first_non_magic = lines[save_idx]
  if first_non_magic.startswith("#"):
    if "DIAS_VERBOSE" in first_non_magic:
      verbose = "verbose"
    # TODO: Remove this because in the case of apply() patterns it won't
    # be disabled because these are overwritten globally.
    elif "DIAS_DISABLE" in first_non_magic:
      disable = True

  if disable:
    # Just return the original immediately
    return lines
  magics = lines[:save_idx]
  cell = ''.join(lines[save_idx:])

  # Escape it because it may contain triple quotes.
  cell = cell.replace('"""', '\\"\\"\\"')
  # Make the code call rewrite()
  new_cell = f"""
dias.rewriter.rewrite("{verbose}",
\"\"\"
{cell}\"\"\")
"""

  # Leave the magics untouched.
  res = magics + [new_cell]
  return res

ip = get_ipython()
use_as_lib = "_IREWR_USE_AS_LIB" in os.environ and os.environ["_IREWR_USE_AS_LIB"] == "True"
if not use_as_lib:
  if ip is None:
    text = "IPython instance has not been detected. Dias won't rewrite cells \
  by default and can only be used as a library."
    warnings.warn(text)
  else:
    ip.input_transformers_cleanup.append(call_rewrite)
