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

############ REWRITERS ############

def remove_axis_1(df, mod_ast, func_ast, arg0_name, the_one_series):
  # Rewrite all the subscripts in the function (it's in-place)  
  rewrite_subscripts(func_ast, arg0_name, the_one_series)

  # We change the name of the function, basically generating a new function,
  # because we don't want to overwrite the old one.
  new_func_name = get_unique_id()
  func_ast.name = new_func_name
  new_func_obj = compile(mod_ast, filename="<ast>", mode="exec")
  # It's important to provide the correct namespace here
  ip = get_ipython()
  exec(new_func_obj, ip.user_ns)
  # We must not recurse. We're calling the apply() of a Series so we should
  # not recurse.
  assert df[the_one_series].apply != _DIAS_apply
  return df[the_one_series].apply(ip.user_ns[new_func_name])

## apply() overwrite ##

class _DIAS_ApplyPat(Enum):
  Vectorized = 1,
  RemoveAxis1 = 2,
  HasOnlyMath = 3

# _DIAS_apply() can't notify outsiders, including the rewriter itself, because
# the rewrite doesn't happen through the rewrite() (i.e., externally). These
# variables are a hack around that. If we need to see whether one of the apply()
# patterns hit, or how much overhead the apply() rewrites added in an arbitrary
# piece of code, we reset these variables, run the code, and check them again.
_DIAS_apply_pat : _DIAS_ApplyPat | None = None
_DIAS_apply_overhead_ns = 0

_DIAS_save_pandas_apply = pd.DataFrame.apply
def _DIAS_apply(self, func: AggFuncType, axis: Axis = 0, raw: bool = False, 
                result_type: Literal["expand", "reduce", "broadcast"] | None = None,
                args=(), **kwargs):
  global _DIAS_apply_pat, _DIAS_apply_overhead_ns
  assert isinstance(self, pd.DataFrame)

  overhead_start = time.perf_counter_ns()

  default_call = functools.partial(_DIAS_save_pandas_apply, self, func, axis, raw, result_type, args, **kwargs)

  def end_overhead():
    global _DIAS_apply_overhead_ns
    overhead_end = time.perf_counter_ns()
    _DIAS_apply_overhead_ns = overhead_end - overhead_start

  def default():
    end_overhead()
    return default_call()

  
  # If any of the args after `axis` is not the default, bail.
  if raw != False or result_type != None:
    return default()
  
  # Only axis=1.
  # TODO: We may be able to relax this for the vectorization pattern.
  if axis != 1:
    return default()
  
  func_source = inspect.getsource(func)
  mod_ast = ast.parse(func_source)
  func_ast = mod_ast.body[0]
  # If it's not a FunctionDef, we bail out because Python is weird.
  # In particular, we don't handle Lambdas here because it would
  # be too hair and unreliable. See this: http://xion.io/post/code/python-get-lambda-code.html
  # These are handled with RemoveAxis1Lambda pattern.
  if not isinstance(func_ast, ast.FunctionDef):
    return default()

  # Either we have one argument or if we have more, they
  # should have default values.
  args_ok = (len(func_ast.args.args) == 1 or 
             len(func_ast.args.defaults) == (len(func_ast.args.args) - 1))
  if not args_ok:
    return default()
  arg0_name = func_ast.args.args[0].arg

  ### ApplyVectorized ###

  if patt_matcher.can_be_vectorized_func(func_ast):
    _DIAS_apply_pat = _DIAS_ApplyPat.Vectorized
    # We need to introduce an ast.Name, the `called_on`. Say we transform this:
    #   def foo(row):
    #     if row['A'] == 1:
    #       return True
    #     return False
    #   df.apply(foo, axis=1)
    # to:
    #
    #   conditions = [df['A'] == 1]
    #   choices = [True]
    #   np.select(conditions, choices, default=False)

    # To do that, we need to replace instances of `row` with `df`. However, we don't
    # the name of the object on which apply() was called. It might not even have a name
    # e.g., it might have been an expression. We'll just introduce an ast.Name that
    # points to `self`.
    ip = get_ipython()
    called_on_id: str = get_unique_id()
    ip.user_ns[called_on_id] = self
    called_on = AST_name(called_on_id)
    def rewrite_if(if_: ast.If, arg0_name, called_on, cond_value_pairs, else_val):
      then_ret = if_.body[0]
      assert isinstance(then_ret, ast.Return)
      else_ = if_.orelse[0]
      if isinstance(else_, ast.Return):
        else_val.append(vec__rewrite_ret_value(else_.value, arg0_name, called_on))
      else:
        assert isinstance(else_, ast.If)
        rewrite_if(else_, arg0_name, called_on, cond_value_pairs, else_val)

      # NOTE - IMPORTANT: Cond-value pairs are pushed in reverse order!
      cond_value_pairs.append(
        (
          vec__rewrite_cond(if_.test, arg0_name, called_on), 
          vec__rewrite_ret_value(then_ret.value, arg0_name, called_on)
        )
      )
    # END OF rewrite_if()
    cond_value_pairs = []
    # This will have a single element but basically we're emulating
    # a pointer.
    else_val = []
    rewrite_if(func_ast.body[0], arg0_name, called_on, cond_value_pairs, else_val)
    assert len(else_val) == 1
    
    np_select_res_id = get_unique_id()
    np_select_res = AST_name(np_select_res_id)
    # We should have been able to go from the Call AST to compile() directly, but
    # Python gave me a hard time, so I did the easy way.
    np_select_call = vec__build_np_select(cond_value_pairs, else_val)
    np_select_asgn = AST_assign(np_select_res, np_select_call)
    
    end_overhead()

    ip.run_cell(astor.to_source(np_select_asgn))
    return ip.user_ns[np_select_res_id]


  ### ApplyRemoveAxis1 ###

  can_remove, the_one_series = patt_matcher.can_remove_axis_1(func_ast, arg0_name)
  if can_remove:
    _DIAS_apply_pat = _DIAS_ApplyPat.RemoveAxis1
    end_overhead()
    return remove_axis_1(self, mod_ast, func_ast, arg0_name, the_one_series)
  
  ### ApplyHasOnlyMath ###

  buf_list = []
  buf_set = set()
  all_arg_names = [arg.arg for arg in func_ast.args.args]
  if patt_matcher.has_only_math(func_ast, all_arg_names, subs=buf_list, external_names=buf_set):
    _DIAS_apply_pat = _DIAS_ApplyPat.HasOnlyMath
    subs = buf_list
    external_names = buf_set
    default_args = func_ast.args.defaults
    ip = get_ipython()
    if has_only_math__preconds(self, default_args, subs, external_names, ip):
      end_overhead()
      return func(self)

  end_overhead()
  return default()


# IMPORTANT: How do we measure the rewriter's time and all that?
# IMPORTANT: If we import it twice, this will be a problem
assert pd.DataFrame.apply != _DIAS_apply
# Overwriting is not trivial. Thanks to:
# https://github.com/lux-org/lux/blob/550a2eca90b26c944ebe8600df7a51907bc851be/lux/core/__init__.py#L27
pd.DataFrame.apply = pd.core.frame.DataFrame.apply = _DIAS_apply

def sort_head(called_on, by: str | None, n: int, asc: bool, orig: Callable):
  func = "nsmallest" if asc else "nlargest"

  req_ty = pd.DataFrame
  opt_func_obj = functools.partial(getattr(req_ty, func), n=n, columns=by)
  if by is None:
    req_ty = pd.Series
    opt_func_obj = functools.partial(getattr(req_ty, func), n=n)
  
  if type(called_on) == req_ty:
    return opt_func_obj(self=called_on)
  else:
    assert isinstance(orig, types.LambdaType)
    return orig(called_on)

def substr_search_apply(ser, needle: str, orig: Callable):
  if type(ser) == pd.Series:
    ls = ser.tolist()
    res = [(needle in s) for s in ls]
    return pd.Series(res, index=ser.index)
  else:
    return orig(ser)
  
def concat_list_to_series(library, e1, e2):
  if library == pd and type(e1) == pd.Series and type(e2) == pd.Series:
    return pd.concat([e1, e2], ignore_index=True)
  return pd.Series(e1.tolist() + e2.tolist())

def fuse_isin(df, col_name, s1, s2):
  if type(df) == pd.DataFrame:
    return df[col_name].isin(s1 + s2)
  return df[col_name].isin(s1) | df[col_name].isin(s2)

def replace_remove_list(ser, orig_to_replace: str | List[str], orig_replace_with: str | List[str], to_replace: str, replace_with: str, orig_lambda: Callable):
  if type(ser) == pd.DataFrame or type(ser) == pd.Series:
    return orig_lambda(to_replace, replace_with)
  return orig_lambda(orig_to_replace, orig_replace_with)

def len_unique(series):
  if type(series) == pd.Series:
    return series.nunique(dropna=False)
  return len(series.unique())

def rewrite_enclosed_sub(enclosed_sub, arg0_name, the_one_series):
  sub = enclosed_sub.get_obj()
  compat_sub = patt_matcher.is_compatible_sub(sub)
  if compat_sub is None:
    return
  if (compat_sub.get_Series() == the_one_series and
      compat_sub.get_df() == arg0_name):
    name_node = AST_name(arg0_name)
    # We need to set these because this is passed to compile()
    # and for some reason it needs them.
    name_node.lineno = sub.lineno
    name_node.col_offset = sub.col_offset
    name_node.ctx = sub.ctx
    enclosed_sub.set_enclosed_obj(name_node)
# END OF FUNCTION #

def rewrite_subscripts(func_body, arg0_name, the_one_series):
  # Modify all the subscripts in the function. To do that, we need access
  # to the parent of the subscript. However, we can't do that so we have
  # to search for enclosed objects. 
  for n in ast.walk(func_body):
    subs = patt_matcher.search_enclosed(n, ast.Subscript)
    for enclosed_sub in subs:
      rewrite_enclosed_sub(enclosed_sub, arg0_name, the_one_series)

def has_only_math__numeric_ty(ty) -> bool:
  numeric_types = {int, float, np.int64, np.float64}
  # NOTE: For some reason we have to do a for loop.
  # I couldn't make it work by putting all acceptable
  # types in a set and checking if `dt` is in the set. 
  for num_dt in numeric_types:
    if ty == num_dt:
      return True
  return False

def has_only_math__preconds(the_df: pd.DataFrame,
                            default_args: List[ast.expr], subs: List[str],
                            external_names: List[str],
                            ipython: InteractiveShell) -> bool:
  # TODO: Do arguments get the value of when the function is defined
  # or when it is called? If it's the former, the arguments might
  # have changed in between.
  # Check that all arguments, except the first, are of numeric type.
  # All the arguments except the first should have default values.
  # Check those. These can be expressions. We only accept them
  # if they are Names or Constants.
  for arg in default_args:
    if isinstance(arg, ast.Name):
      if not has_only_math__numeric_ty(type(ipython.user_ns[arg.id])):
        return False
    elif isinstance(arg, ast.Constant):
      if not has_only_math__numeric_ty(type(arg.value)):
        return False
    else:
      return False
  # Check that all external names used are of numeric type.
  for ext in external_names:
    ext_obj = ipython.user_ns[ext]
    if not has_only_math__numeric_ty(type(ext_obj)):
      return False
  # END OF LOOP #

  # Check that all the Series accessed are numeric
  dtypes = the_df.dtypes
  for sub in subs:
    dt = None
    try:
      dt = dtypes[sub]
    except:
      # Something went bad... The sub is not in the DF?
      return False
    if not has_only_math__numeric_ty(dt):
      return False

  ### END OF LOOP ###
  return True
  

# Generate `ntargets` names and assign [] to all of them.
# Return the names and the assignments.
def gen_init_lists(ntargets: int) -> Tuple[List[ast.Name], List[ast.Assign]]:
  target_names = []
  init_lists = []
  for idx in range(ntargets):
    # We could use more descriptive variable names but the
    # name of a Series can be literally anything (questionmarks etc.)
    # so we're better off with our names.
    targ = AST_name("_REWR_" + "targ_" + str(idx))
    target_names.append(targ)
    init_lists.append(
      AST_assign(targ, ast.List(elts=[])))
  # END OF LOOP #
  return target_names, init_lists

# Generate <assign_to> = <on>.split(<split args>)
def gen_split(assign_to: ast.Name, on: ast.AST, split_args: ast.arguments) -> Tuple[ast.Name, ast.Assign]:
  return \
    AST_assign(
      assign_to,
      AST_attr_call(called_on=on,
                    name="split", args=split_args)
    )

def gen_none_checked_idx(obj_to_idx: ast.AST, len_obj: ast.Call, idx: int) -> ast.IfExp:
  none_check = \
    AST_cmp(lhs=len_obj, rhs=ast.Constant(value=idx), op=ast.Gt())
  if_exp = \
    ast.IfExp(
      body=AST_sub_const2(obj_to_idx, idx),
      orelse=ast.Constant(value=None),
      test=none_check)
  return if_exp

def gen_append(append_to: ast.Name, value: ast.AST):
  append = \
    ast.Expr(
      value=AST_attr_call(append_to, "append", 
                          args=[value]))
  return append

def str_split__tolist_and_split(sub_to_split: ast.Subscript,
                                split_args: ast.arguments,
                                str_it: ast.Name):
  ### .tolist() ###
  ls_name = AST_name("_REWR_ls")
  # _REWR_ls = <obj>.tolist()
  ls = AST_assign(ls_name, AST_attr_call(called_on=sub_to_split, name="tolist"))

  spl_name = AST_name("_REWR_spl")
  spl = gen_split(spl_name, str_it, split_args)

  return ls_name, ls, spl_name, spl

def wrap_in_if(obj: ast.AST,
               ty: str,
               then: List[ast.stmt],
               else_: List[ast.stmt]):
  return AST_if_else(
    cond=
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[obj]
        ),
        rhs=AST_attr_chain(ty),
        op=ast.NotEq()
      ),
    then_block=then,
    else_block=else_
  )

def vec__rewrite_name(name: ast.Name, arg0_name: str, called_on: ast.Name):
  new_name = None
  if name.id == arg0_name:
    new_name = called_on
  else:
    # Can avoid making a copy here because if anybody
    # changes this, they will change where the `id` points,
    # not the string itself.
    new_name = name
  return new_name

def vec__rewrite_ret_value(value: ast.expr, arg0_name: str, called_on: ast.Name):
  if isinstance(value, ast.Name):
    return vec__rewrite_name(value, arg0_name, called_on)
  for n in ast.walk(value):
    enclosed_names = patt_matcher.search_enclosed(n, ast.Name)
    for enclosed_name in enclosed_names:
      name = enclosed_name.get_obj()
      assert isinstance(name, ast.Name)
      enclosed_name.set_enclosed_obj(vec__rewrite_name(name, arg0_name, called_on))
  return value

def vec__rewrite_cond(c, arg0_name, called_on):
  if isinstance(c, ast.BoolOp):
    op_to_pandas_op = {ast.And: ast.BitAnd(), ast.Or: ast.BitOr()}
    op = op_to_pandas_op[type(c.op)]
    assert len(c.values) >= 2
    binop = ast.BinOp(left=vec__rewrite_cond(c.values[0], arg0_name, called_on), 
                      op=op,
                      right=vec__rewrite_cond(c.values[1], arg0_name, called_on))
    for val in c.values[2:]:
      new_cond = vec__rewrite_cond(val, arg0_name, called_on)
      binop = ast.BinOp(left=binop, op=op, right=new_cond)
    return binop
  elif isinstance(c, ast.Compare):
    op = c.ops[0]
    comp = c.comparators[0]
    new_left = vec__rewrite_cond(c.left, arg0_name, called_on)
    new_right = vec__rewrite_cond(comp, arg0_name, called_on)
    if not isinstance(op, ast.In):
      return ast.Compare(left=new_left, ops=c.ops, comparators=[new_right])
    # Handle `in` specially
    # NOTE: We know that since we check that the body of function is
    # an if-chain with only `return`s, then whatever is on the RHS 
    # cannot be defined inside the function. That would be a
    # problem because we would have to move it outside.
    call = AST_attr_call(called_on=new_left, name='isin', args=[new_right])
    return call
  elif isinstance(c, ast.Attribute):
    new_value = vec__rewrite_cond(c.value, arg0_name, called_on)
    # I don't think we need to visit `.attr` because it should be a string.
    return ast.Attribute(value=new_value, attr=c.attr)
  elif isinstance(c, ast.Subscript):
    new_value = vec__rewrite_cond(c.value, arg0_name, called_on)
    # I don't think we need to visit `.slice` because it should
    # be a Constant
    return ast.Subscript(value=new_value, slice=c.slice)
  elif isinstance(c, ast.Name):
    return vec__rewrite_name(c, arg0_name, called_on)
  elif isinstance(c, ast.Constant):
    return c
  else:
    print(type(c))
    assert 0

def vec__rewrite_ifexp(ifexp: ast.IfExp, arg0_name, called_on, cond_value_pairs, else_val):
  # NOTE - IMPORTANT: Cond-value pairs are pushed in reverse order!
  cond_value_pairs.append(
    (
      vec__rewrite_cond(ifexp.test, arg0_name, called_on), 
      vec__rewrite_ret_value(ifexp.body, arg0_name, called_on)
    )
  )
  else_val.append(vec__rewrite_ret_value(ifexp.orelse, arg0_name, called_on))

def vec__build_np_select(cond_value_pairs, else_val):
  conditions = ast.List(elts=[cond for (cond, _) in reversed(cond_value_pairs)])
  values = ast.List(elts=[val for (_, val) in reversed(cond_value_pairs)])
  np_select = \
    AST_attr_call(
      called_on=AST_name(name='np'), name='select',
      args=[conditions, values],
      keywords={'default': else_val[0]})
  return np_select

def extract_line_info(n1, n2, nodes_to_extract_info_for):
  if n1 in nodes_to_extract_info_for:
    assert hasattr(n2, "lineno")
    assert hasattr(n2, "end_lineno")
    n1.lineno = n2.lineno
    n1.end_lineno = n2.end_lineno
  # END IF #

def co_traverse_and_extract_line_info(node1, node2, nodes_to_extract_info_for):
  if type(node1) != type(node2):
    print(astor.dump_tree(node1, indentation='  '))
    print('******')
    print(astor.dump_tree(node2, indentation='  '))
    assert False 
  extract_line_info(node1, node2, nodes_to_extract_info_for)

  for field in node1._fields:
    if field in ["ctx", "type_ignores"]:
      continue
    val1 = getattr(node1, field)
    val2 = getattr(node2, field)

    if isinstance(val1, list):
      for item1, item2 in zip(val1, val2):
        co_traverse_and_extract_line_info(item1, item2, nodes_to_extract_info_for)
      ### END FOR ###
    elif isinstance(val1, ast.AST):
      co_traverse_and_extract_line_info(val1, val2, nodes_to_extract_info_for)
    else:
      # Leaf values (numbers, strings, etc.)
      pass
    # END IF #
  ### END FOR ###

# I don't think we can declare a type for `ipython`
def rewrite_ast(cell_ast: ast.Module, line_info=False) -> Tuple[str, Dict]:
  assert isinstance(cell_ast, ast.Module)
  body = cell_ast.body
  matched_patts = patt_matcher.patt_match(body)
  
  dias_rewriter_prefix = ast.Attribute(value=ast.Name(id='dias'), attr='rewriter')
  pd_Series_prefix = ast.Attribute(value=ast.Name(id='pd'), attr='Series')
  pd_DataFrame_prefix = ast.Attribute(value=ast.Name(id='pd'), attr='DataFrame')

  # TODO: We may be able to simplify the following not that sliced execution is gone.

  ##############
  # Currently, the pattern-matcher returns a list of patterns, and for each
  # pattern, the indexes (in `body`) of every stmt that takes part in this pattern.
  # The ordering of patterns is not particular order. This means that our rewriting based
  # on the patterns cannot be context-sensitive. For example, if your rewrite depends
  # on how the code before some stmt (contributing to the pattern) is, then we can't support
  # it. To see this, suppose we have the following body:
  #   stmt1   
  #   stmt2
  #   stmt3
  #
  # Say that a pattern p1 matches stmt1 and stmt3 and another pattern p2 matches
  # stmt2. Suppose that you see process p1 first. If you want to rewrite stmt3,
  # but this rewriting depends on how the code before stmt3 (e.g., stmt2) is, then you can't
  # support this rewrite because while processing p2, you might rewrite stmt2 and
  # p2 has not been processed yet.
  #
  # Similarly, if p2 is first, and your rewriting of stmt2 depends on how the code
  # before it looks like (e.g., stmt1), then you can't support this rewrite
  # because p1 has not been processed yet and while processing p1, we might
  # rewrite stmt1.
  #
  # So, no particular order helps us. Just to put it out there, there are at least
  # 3 different options from what we're doing now:
  #
  # 1. Sort based on the earliest stmt in the pattern. So, we would return [p1, p2], 
  #    because stmt1 (the earliest stmt in p1) is before stmt2.
  # 2. Sort based on the latest stmt. So, we would return [p2, p1]
  # 3. Don't match pattern in "between" other patterns. So, since we have a pattern
  #    that has both stmt1 and stmt3, then we don't match pattern that has stmt2,
  #    even though stmt2 is not in the bigger pattern. Basically, our patterns
  #    don't have just specific statements, but rather one range of statements.
  #
  # Each of those might be more useful than what we're doing now, but I also think
  # that any of them will constrain us, so I have deferred the decision till we have
  # more patterns.
  #
  # Benefits of the current way:
  # - You could parallelize it. The rewritings are basically independent. You just merge
  #   the stmt_lists in the end as it happens already.
  #
  # Cons of the current way:
  # - It is based on the assumption that when processing a pattern, you either only modify
  #   one of the matched statements (this is based on the guarantee of the pattern-matcher
  #   that the patterns are non-overlapping) or you only add statements. You don't delete
  #   or modify other statements. That would work because you may modify one of the statements
  #   of another pattern that has been not processed yet. Basically, this con is another
  #   way of highlighting the fact that the rewrites have to be context-sensitive.

  ##############
  # The patterns can't operate trivially over `body` (i.e., add things to it in the middle etc.)
  # because that would invalidate the `stmt_idxs` of other patterns. We need a cleverer way. We will
  # have the following assumption:
  # - Processing each pattern, we insert either before or after any of the statements contributing to the
  #   pattern.
  # Then, I had this idea where after we process a pattern, we return a tuple of an index and a list of stmts.
  # This would define the statements to be inserted and where. Once we process all patterns, we insert
  # all those in `body` (note that the indexes would be non-overlapping because you can always set the
  # index as being "on a statement that corresponds to the pattern" and so since the patterns are
  # non-overlapping, these indexes would not overlap). But @Damitha had a better idea: Convert `body` to
  # a list of lists (initially singletons). Then, add stmts to these individual lists. Finally, seriealize
  # this list of lists. This is more performant than my idea because lists in Python are represented as
  # arrays and so inserting in the middle (what we would need to do at the end) is way slower than
  # appending to the individual lists and finally searializing.
  ##############

  list_of_lists = [[stmt] for stmt in body]

  # Stats of what pattern was applied.
  stats = []
  for patt, stmt_idxs in matched_patts:
    if patt is not None and type(patt).__name__ in disabled_patts:
      print(type(patt).__name__)
      continue

    if isinstance(patt, patt_matcher.HasSubstrSearchApply):
      # Similar to SortHead
      
      orig_call = patt.attr_call.call.get_obj()
      orig_lineno = orig_call.lineno
      orig_end_lineno = orig_call.end_lineno

      lam_arg = "_DIAS_x"
      apply_call = patt.attr_call
      orig_called_on = apply_call.get_called_on()
      apply_call.set_called_on(AST_name(lam_arg))
      orig_called_on_replaced = apply_call.call.get_obj()
      lambda_args = ast.arguments(
        args=[ast.arg(arg=lam_arg)],
        defaults=[],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None
      )
      orig_wrapped_in_lambda = \
        ast.Lambda(args=lambda_args, 
                   body=orig_called_on_replaced)
      
      new_call = \
        AST_attr_call(
          called_on=dias_rewriter_prefix,
          name="substr_search_apply",
          keywords={'ser': orig_called_on, 'needle': patt.get_needle(),
                    'orig': orig_wrapped_in_lambda}
        )
      
      apply_call.call.set_enclosed_obj(new_call)
      
      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_call]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.IsInplaceUpdate):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      orig_stmt = stmt_list[0]

      series_call: patt_matcher.SeriesCall = patt.series_call
      orig_call = series_call.attr_call.call.get_obj()
      orig_lineno = orig_call.lineno
      orig_end_lineno = orig_call.end_lineno

      # Let's limit it to top-level. This makes our life easy
      # because we can bind the series access to a variable.
      # The reason we need to that even for CompatSub is that if
      # it's not a pd.Series, maybe the subscript changes internal state
      # or who knows what, so the subscript should too be evaluated once.
      if patt.assign != orig_stmt:
        continue

      func = series_call.attr_call.get_func()
      # TODO: For now, only do it for `fillna` because I haven't tested
      # the rest.
      if func != "fillna":
        continue
    

      # Assign the subscript result in a temp.
      tmp = AST_name("_DIAS_ser")
      asgn = AST_assign(tmp, series_call.get_sub().get_sub_ast())
      

      # Construct the inplace call.
      call: ast.Call = series_call.attr_call.call.get_obj()
      call_copy = copy.deepcopy(call)
      call_copy.keywords.append(ast.keyword(arg='inplace', value=ast.Constant(value=True, kind=None)))
      expr_call = ast.Expr(value=call_copy)
      
      precond_check = wrap_in_if(tmp, pd_Series_prefix, [orig_stmt], [expr_call])

      stmt_list[0] = asgn
      stmt_list.append(precond_check)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [asgn, precond_check]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.HasToListConcatToSeries):
      # Pass for now
      
      orig_call = patt.enclosed_call.get_obj()
      orig_lineno = orig_call.lineno
      orig_end_lineno = orig_call.end_lineno

      new_call = AST_attr_call(
          called_on=dias_rewriter_prefix,
          name="concat_list_to_series",
          keywords={
              'library': patt.enclosed_call.get_obj().func.value,
              'e1': patt.left_ser_call.get_sub().get_sub_ast(),
              'e2': patt.right_ser_call.get_sub().get_sub_ast()
          }
      )
      patt.enclosed_call.set_enclosed_obj(new_call)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_call]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.SortHead):
      # Replace the call to sort_values() with nsmallest/nlargest.

      # We need to check that the called-on object is a pd.Series (or pd.DataFrame; it can be both
      # but that's not the important thing here). The called-on object is what appears before .sort_values()
      # Say that the original is this (the called-on is foo().reads_x()):
      #   x = ...
      #   test(changes_x(), foo().reads_x().sort_values().head())
      # We can execute the following instead:
      #   x = ...
      #   test(changes_x(), foo().reads_x().nsmallest())
      # iff foo().reads_x() is a pd.Series. The problem is how/when do we do the check?
      # The naive way would be to wrap an if around checking the type of called-on:
      #   if type(foo().reads_x()) == pd.Series:
      #     test(changes_x(), foo().reads_x().nsmallest())
      #   else:
      #     <original>
      #
      # The problem with this is that called-on is evaluated twice, which doesn't
      # happen in the original program. This can be both a performance problem but also a correctness
      # problem if the called-on evaluation accesses/changes some global state. In particular here,
      # called-on reads `x`, which is changed in the call to changes_x(). So, the first and second
      # evaluations of called-on will be different.
      #
      # What we want is to _bind_ the result to a variable and then reuse that:
      #   called_on = foo().reads_x()
      #   if type(called_on) == pd.Series:
      #     test(changes_x(), called_on.nsmallest())
      #   else:
      #     <original>
      #
      # This still has a problem however. In the original code, the called-on evaluation should
      # read the updated value of `x`, i.e., after the call to changes_x(), but here, it will read
      # a stale value. What we need here is a _local_ binding, exactly at the place where the original
      # sub-expression (i.e., the call chain foo().reads_x().sort_values().head()) appeared. Basically,
      # we need a let binding like in e.g., OCaml:
      #   test(changes_x(), 
      #     let called_on = foo().reads_x() in called_on.nsmallest() 
      #       if type(called_on) == pd.Series 
      #       else ...)
      #
      # However, I don't know of a way to do that in Python. A workaround is to create a function, evalute
      # the called-on and pass it as an argument. This is a local binding, as the result of called-on is
      # bound to the argument (and what would appear after "in" is the body of the function)
      #   def sort_head(called_on, ...):
      #     if type(called_on) == pd.Series:
      #       return called_on.nsmallest()
      #     else:
      #       return called_on.sort_values().head()
      #   test(changes_x(), sort_head(foo.reads_x()))
      #
      # This is basically the solution we will employ, but we have some practical problems.
      # The original version is not always the same. It can be:
      # - <whatever>.sort_values().head()
      # or
      # - <whatever>.sort_values(columns='...').head()
      # or
      # - <whatever>.sort_values().head(n=7)
      # ... etc.
      #
      # Which means, we cannot create _one_ such function which has the original code hardcoded (the same
      # holds for the rewritten, for the same reasons). Ideally, we would like to make this function
      # parametric to the original code. But, in general a function can't get "code" as an argument. Code
      # can be 3 things:
      # (1) A string
      # (2) An AST
      # (3) A compiled object
      # 
      # Now, remember, the rewriter ouptputs a string (which will contain the call to sort_head()). (1) is ok
      # but it's generally unreliable. (2) is not ok because we need a stringified version of the AST, but
      # I don't know how to do that. And (3) just doesn't work because again, the rewriter outputs strings,
      # not objects.
      #
      # The solution is to wrap the original code in a lambda, and pass that. We would theoretically need
      # to do the same thing for the rewritten version, but it's easier and more comprehensible to
      # "perform the rewriting" (not really) in sort_head(), passing only the parameters to produce
      # the version to be called correctly (if you look at sort_head(), this will make sense).
      #
      # Note that a lambda will capture the local environment. So, for example, this:
      #   y = 3
      #   def bar(lam):
      #     return lam()
      #
      #   def foo():
      #     y = 2
      #     return bar(lambda: y)
      #   print(foo())
      #   print(y)
      #
      # will print 2, 3 (the global is not changed by foo() and the lambda captures the local environment of
      # the foo() when called from bar())
      
      orig_call = patt.head.call.get_obj()
      orig_lineno = orig_call.lineno
      orig_end_lineno = orig_call.end_lineno

      n = patt.get_head_n()
      by = patt.get_sort_by()

      if by is None:
        by = ast.Constant(value=None)

      # We _need_ to pick a name that won't clobber other locals and globals.
      lam_arg = "_DIAS_x"
      # Get the called-on part.
      orig_called_on = patt.sort_values.get_called_on()
      # Replace it with the argument of the lambda. Remember, the lambda
      # eventually will be sth like: lambda x: x.sort_values().head().
      patt.sort_values.set_called_on(AST_name(lam_arg))
      orig_called_on_replaced = patt.head.call.get_obj()
      lambda_args = ast.arguments(
        args=[ast.arg(arg=lam_arg)],
        defaults=[],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None
      )
      orig_wrapped_in_lambda = \
        ast.Lambda(args=lambda_args, 
                   body=orig_called_on_replaced)

      new_call = \
        AST_attr_call(
          called_on=dias_rewriter_prefix,
          name="sort_head",
          keywords={'called_on': orig_called_on, 'by': by, 
                    'n': n, 'asc': ast.Constant(value=patt.is_sort_ascending()), 
                    'orig': orig_wrapped_in_lambda}
        )
      
      patt.head.call.set_enclosed_obj(new_call)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_call]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.ReplaceRemoveList):
      """This rewrites `e.replace([x1], [x2], **kwargs)` as:
      dias.rewriter.replace_remove_list(e, [x1], [x2], x1, x2, 
        lambda _DIAS_x1, _DIAS_x2: e.replace(_DIAS_x1, _DIAS_x2, **kwargs))
      """
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]

      df_or_ser = patt.attr_call.get_called_on()

      call_raw = patt.attr_call.call.get_obj()      
      orig_to_replace = call_raw.args[0]
      orig_replace_with = call_raw.args[1]
      orig_lineno = call_raw.lineno
      orig_end_lineno = call_raw.end_lineno


      call_raw.args = [AST_name(name="_DIAS_x1"), AST_name(name="_DIAS_x2")]

      lambda_args_args = [
          ast.arg(arg="_DIAS_x1"),
          ast.arg(arg="_DIAS_x2")
      ]      
      lambda_args = ast.arguments(
        args=lambda_args_args,
        defaults=[],
        kw_defaults=[],
        kwarg=None,
        kwonlyargs=[],
        posonlyargs=[],
        vararg=None
      )

      orig_wrapped_in_lambda = ast.Lambda(args=lambda_args, body=call_raw)

      new_replace_call = AST_attr_call(
        called_on=dias_rewriter_prefix,
        name="replace_remove_list",
        args=[
          df_or_ser,
          orig_to_replace,
          orig_replace_with,
          patt.to_replace,
          patt.replace_with,
          orig_wrapped_in_lambda
        ]
      )

      if patt.inplace and patt.attr_call.call.get_encloser() == stmt_list[0]:
        # The parent of the call must be an assignment.
        # This assignment must be top-level because then we
        # can replace it in the stmt_list[]. Otherwise, we don't
        # know the parent of the assignment and if an assignment
        # is not top-level, we probably don't want to touch it
        # anyway.
        
        # Add the inplace=True argument. This implicitly 
        # updates orig_wrapped_in_lambda, which updates new_replace_call.
        call_raw.keywords.append(
          ast.keyword(arg="inplace", value=ast.Constant(value=True)))
        new_replace_call = ast.Expr(value=new_replace_call)
        stmt_list[0] = new_replace_call
      else:
        patt.attr_call.call.set_enclosed_obj(new_replace_call)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_replace_call]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.MultipleStrInCol):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      stmt = stmt_list[0]

      for str_in_col in patt.str_in_cols:
        # TODO(Mircea): Instead of creating an AST which will be deparsed to
        # string anyway, put a placeholder node, then deparse it, then replace
        # this with the string you want.

        # We don't even have to insert this. We can just define it once at the top
        # of the file and it's available to the notebook because they're using
        # the same Python namespace. But I'm leaving it because it helps the user
        # see what's happening.
        
        orig_cmp = str_in_col.cmp_encl.get_obj()
        orig_lineno = orig_cmp.lineno
        orig_end_lineno = orig_cmp.end_lineno


        # Get string versions
        the_str = str_in_col.the_str.value
        assert isinstance(str_in_col.the_sub.value, ast.Name)
        df = str_in_col.the_sub.value.id
        col = None
        if isinstance(str_in_col.the_sub.slice, ast.Name):
          col = str_in_col.the_sub.slice.id
        else:
          assert isinstance(str_in_col.the_sub.slice, ast.Constant)
          col = str_in_col.the_sub.slice.value
        the_sub = f"{df}[{col}]"
        orig = f"'{the_str}' in {the_sub}.to_string()"

        # You need to be careful when handling strings like that because you might
        # miss parentheses.

        # We can specialize this for an index that is an int. Try to convert the string to int
        # and if you fail, 
        contains_expr = f"astype(str).str.contains('{the_str}').any()"
        new_expr = f"({the_sub}.{contains_expr} or _REWR_index_contains({the_sub}.index, '{the_str}')) if type({df}) == pd.DataFrame else ({orig})"
        new_cmp = ast.parse(new_expr, mode='eval').body
        str_in_col.cmp_encl.set_enclosed_obj(new_cmp)
        
        patt_info = {"patt": type(patt).__name__,
                    "orig-lineno": orig_lineno,
                    "orig-end-lineno": orig_end_lineno,
                    "delegates": [new_cmp]}
        stats.append(patt_info)
      ### END OF LOOP ###

      index_contains_func_str = \
"""def _REWR_index_contains(index, s):
  if index.dtype == np.int64:
    try:
      i = int(s)
      return len(index.loc[i]) > 0
    except:
      return False
  else:
    return index.astype(str).str.contains(s).any()
"""

      save_stmt = stmt_list[0]
      stmt_list[0] = ast.parse(index_contains_func_str).body[0]
      stmt_list.append(save_stmt)
    elif isinstance(patt, patt_matcher.FuseIsIn):
      orig_binop = patt.binop_encl.get_obj()
      orig_lineno = orig_binop.lineno
      orig_end_lineno = orig_binop.end_lineno
      
      new_call = AST_attr_call(
          called_on=dias_rewriter_prefix,
          name="fuse_isin",
          keywords={
              'df': AST_name(patt.the_ser.get_sub().get_df()),
              'col_name': ast.Constant(value=patt.the_ser.get_sub().get_Series()),
              's1': patt.left_name,
              's2': patt.right_name
          }
      )
      patt.binop_encl.set_enclosed_obj(new_call)
      
      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_call]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.StrSplitPython):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      stmt = stmt_list[0]

      # TODO: Possibly set maxsplits (in split()) to idx_from_split_list+1
      # to save allocations and time.

      # Otherwise, we wouldn't have a gotten StrSplitPython directly.
      assert patt.expand_true == True
      
      orig_lineno = stmt.lineno
      orig_end_lineno = stmt.end_lineno

      sub_to_split = patt.rhs_obj.get_sub_ast()

      ### Initialize lists ###
      target_names, init_lists = gen_init_lists(len(patt.lhs_targets))


      ### Generate loop ###

      # Name for the loop iterator.
      str_it = AST_name("_REWR_s")
      ls_name, ls, spl_name, spl = \
        str_split__tolist_and_split(sub_to_split,
                                    patt.split_args,
                                    str_it)
      for_body = [spl]

      # Target 0 is special because we have at least one element in the result
      # of split().
      assert len(target_names) >= 1

      targ0_append = gen_append(target_names[0], AST_sub_const2(spl_name, 0))
      for_body.append(targ0_append)
      # For the other targets, we need to check we have a split
      spl_len = AST_call(func=AST_name("len"), args=[spl_name])
      for targ_idx in range(1, len(target_names)):
        value_to_append = gen_none_checked_idx(spl_name, spl_len, targ_idx)
        append = gen_append(target_names[targ_idx], value=value_to_append)
        for_body.append(append)

      for_loop = AST_iter_container(it=str_it, cont=ls_name, body=for_body)

      ### Generate assigns to Series ###

      assign_series = []
      # Assign back
      for idx in range(len(patt.lhs_targets)):
        assign = \
          AST_assign(
            lhs=AST_sub_const2(patt.lhs_root_name, patt.lhs_targets[idx].value),
            rhs=target_names[idx]
          )
        assign_series.append(assign)

      rewritten_body = init_lists + [ls] + [for_loop] + assign_series

      ### Wrap in if-else block ###

      new_stmt = \
        wrap_in_if(sub_to_split, pd_Series_prefix, then=[stmt_list[0]], else_=rewritten_body)
      stmt_list[0] = new_stmt

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_stmt]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.FusableStrSplit):
      assert len(stmt_idxs) == 2
      split_stmt_idx = stmt_idxs[0]
      split_stmt_list = list_of_lists[split_stmt_idx]
      
      orig_lineno = split_stmt_list[0].lineno
      orig_end_lineno = patt.expr_to_replace.get_obj().end_lineno

      sub_to_split = patt.source_split.rhs_obj.get_sub_ast()
      idx_from_split_list = patt.index

      ### Initialize lists ###
      # We need one list.
      target_names, init_lists = gen_init_lists(1)

      ### Generate loop ###

      # Name for the loop iterator.
      str_it = AST_name("_REWR_s")
      ls_name, ls, spl_name, spl = \
        str_split__tolist_and_split(sub_to_split,
                                    patt.source_split.split_args,
                                    str_it)
      
      value_to_append = None
      if idx_from_split_list == 0:
        value_to_append = AST_sub_const2(spl_name, idx_from_split_list)
      else:
        spl_len = AST_call(func=AST_name("len"), args=[spl_name])
        value_to_append = gen_none_checked_idx(spl_name, spl_len, idx_from_split_list)

      append = gen_append(target_names[0], value_to_append)
      for_body = [spl, append]

      for_loop = AST_iter_container(it=str_it, cont=ls_name, body=for_body)

      # Get a temporary. We will assign the result there of either
      # the original or the rewritten version and then replace the
      # original .split() expr with this
      res_name = AST_name("_REWR_res")
      assign_rewr_res = AST_assign(res_name, target_names[0])
      rewritten_body = init_lists + [ls] + [for_loop] + [assign_rewr_res]

      assign_orig_res = AST_assign(res_name, patt.source_split.whole_rhs_encl.get_obj())

      ### Wrap in if-else block ###

      new_stmt = \
        wrap_in_if(sub_to_split,
                   pd_Series_prefix,
                   then=[assign_orig_res],
                   else_=rewritten_body)
      # Replace the split with the new_stmt
      split_stmt_list[0] = new_stmt
      # Replace the index expression with the final result
      patt.expr_to_replace.set_enclosed_obj(res_name)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_stmt]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.FusableReplaceUnique):
      assert len(stmt_idxs) == 2
      replace_stmt_idx = stmt_idxs[0]
      replace_stmt_list = list_of_lists[replace_stmt_idx]
      unique_stmt_idx = stmt_idxs[1]
      unique_stmt_list = list_of_lists[unique_stmt_idx]
      
      orig_lineno = replace_stmt_list[0].lineno
      orig_end_lineno = unique_stmt_list[0].end_lineno

      res_list_name = AST_name("_REWR_res")
      res_list_assgn = AST_assign(res_list_name, ast.List(elts=[]))

      unique_set = AST_name("_REWR_uniq")
      unique_set_assign = \
        AST_assign(unique_set,
                   AST_call(func=AST_name("set")))

      # This should be a dict.
      replace_arg = patt.replace_arg

      # Loop iterator
      str_it = AST_name("_REWR_s")

      lookup_dict = \
        AST_assign(str_it, 
          ast.Subscript(value=replace_arg, 
                        slice=str_it))
      res_append = ast.Expr(
        value=AST_attr_call(called_on=res_list_name,
                            name="append", args=[str_it])
      )
      unique_add = ast.Expr(
        value=AST_attr_call(called_on=unique_set,
                            name="add", args=[str_it])
      )

      # Do it with a `try` instead of `if s in dict: s = dict[s]`
      # because it's faster (probably we do only one lookup)
      try_wrap = AST_try_except(try_block=[lookup_dict], 
                                except_block=[ast.Pass()])

      replace_on_sub = patt.replace_on_sub
      # Make it a list because it's faster
      listify = AST_attr_call(replace_on_sub.get_sub_ast(), "tolist")

      for_loop = \
        AST_iter_container(it=str_it,
                           cont=listify,
                           body=[try_wrap, res_append, unique_add])

      # Assign the list res to the LHS sub of the replace. This sub
      # may be used later
      repl_lhs_sub_assign = AST_assign(patt.replace_lhs_sub.get_sub_ast(), res_list_name)

      # Get a temporary. We will assign the result there of either
      # the original or the rewritten version and then replace the
      # original .unique() expr with this
      temp_res_name = AST_name("_REWR_temp_res")
      assign_rewr_res = AST_assign(temp_res_name, unique_set)
      rewritten_body = [res_list_assgn, unique_set_assign, for_loop,
                        repl_lhs_sub_assign, assign_rewr_res]

      assign_orig_res = AST_assign(temp_res_name, patt.expr_to_replace.get_obj())

      # The sub we call .replace() on should be a Series.
      precond_is_series = \
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[patt.replace_on_sub.get_sub_ast()]
        ),
        rhs=pd_Series_prefix,
        op=ast.Eq()
      )

      # The argument to replace() should be a dict
      precond_is_dict = AST_call(
          func=AST_name("isinstance"),
          args=[patt.replace_arg, AST_name("dict")]
        )
      preconditions = \
        AST_bool_and(precond_is_series, precond_is_dict)

      # Wrap in an if
      new_stmt = \
        AST_if_else(
          cond=preconditions,
          then_block=rewritten_body,
          else_block=[replace_stmt_list[0], assign_orig_res]
        )
      
      # Add a final expression with the temp res
      outside_exp = ast.Expr(value=temp_res_name)

      replace_stmt_list[0] = new_stmt
      unique_stmt_list[0] = outside_exp
      
      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_stmt, outside_exp]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.ApplyVectorizedLambda):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      
      orig_call = patt.apply_call.call.get_obj()
      orig_lineno = orig_call.lineno
      orig_end_lineno = orig_call.end_lineno

      # These two will have a single element but we're emulating a pointer.
      cond_value_pairs = []
      else_val = []
      vec__rewrite_ifexp(patt.lam.body, patt.arg0_name,
                         patt.called_on_name, cond_value_pairs, else_val)
      assert len(cond_value_pairs) == 1
      assert len(else_val) == 1

      # Create a function. We'll add our code there, along with precondition
      # checks and we'll replace the apply() with a call to this function.
      # We could probably check the preconditions with an IfExp, but the
      # code looks too ugly.

      the_series = AST_name("df")

      precond_is_DataFrame = \
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[the_series]
        ),
        rhs=pd_DataFrame_prefix,
        op=ast.Eq()
      )

      np_select = vec__build_np_select(cond_value_pairs, else_val)
      # Wrap it in pd.Series(). The conversion to Series will happen
      # implicitly if we have sth like this:
      #   df['x'] = df.apply(...)
      # but in the following we'll have a problem:
      #   ... = df.apply(...).str.replace()
      np_select_to_series = AST_attr_call(ast.Name("pd"),
                                          "Series", args=[np_select])

      rewritten_body = [ast.Return(value=np_select_to_series)]
      original_body = [ast.Return(value=patt.apply_call.call.get_obj())]

      conditioned = \
        AST_if_else(
          cond=precond_is_DataFrame,
          then_block=rewritten_body,
          else_block=original_body
        )

      apply_vec_name = "_REWR_apply_vec"
      apply_vec = AST_function_def(apply_vec_name,
                                   ["df"], body=[conditioned])

      # Insert before the current (top-level) statement
      stmt_list.insert(0, apply_vec)

      # Replace the apply()
      new_call = AST_call(func=ast.Name(id=apply_vec_name), args=[the_series])
      patt.apply_call.call.set_enclosed_obj(new_call)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [apply_vec, new_call]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.RemoveAxis1Lambda):
      # Remove axis=1
      call = patt.call_name.attr_call.call.get_obj()
      orig_lineno = call.lineno
      orig_end_lineno = call.end_lineno
      kwargs = call.keywords
      axis_1_idx = -1
      for idx, kw in enumerate(kwargs):
        if kw.arg == "axis":
          axis_1_idx = idx
          break
        # END IF #
      ### END FOR ###
      assert axis_1_idx != -1
      call.keywords.pop(axis_1_idx)
      # Make row.apply -> row[Name].apply
      attr_ast = patt.call_name.attr_call.get_attr_ast()
      attr_ast.value = \
        AST_sub_const(patt.call_name.get_name(), patt.the_one_series)
      
      arg0_name = patt.lam.args.args[0].arg

      # Rewrite all subscripts. We need to handle the special
      # case where the subscript is top-level. This can happen e.g., here:
      #   df.apply(lambda row: row['A'], axis=1)
      if isinstance(patt.lam.body, ast.Subscript):
        enclosed_sub = patt_matcher.get_enclosed_attr(patt.lam, "body")
        rewrite_enclosed_sub(enclosed_sub, arg0_name, patt.the_one_series)
      # END IF #
      rewrite_subscripts(patt.lam.body, arg0_name, patt.the_one_series)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [patt.lam]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.FuseApply):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      
      orig_lineno = patt.left_apply.call.get_obj().lineno
      orig_end_lineno = patt.right_apply.call.get_obj().end_lineno

      def call_function_and_assign_result(assign_to: ast.Name, 
                                          function: ast.AST, arg: ast.Name):
        assert isinstance(function, ast.Name) or isinstance(function, ast.Lambda)
        rhs = None
        if isinstance(function, ast.Name):
          # Just call the function with the argument as its first argument
          rhs = AST_call(func=function, args=[arg])
        else:
          # Replace all occurrences of the Lambda's first argument with
          # our first argument
          # TODO: This is fragile. Probably it's ok though because Python
          # The alternative is to create a new function for the lambda and
          # just pass our arg as first arg but this seems a lot of hassle.
          larg0 = patt_matcher.lambda_has_only_one_arg(function)
          # We should have checked it when pattern-matching this
          assert larg0 is not None
          for n in ast.walk(function.body):
            if isinstance(n, ast.Name) and n.id == larg0:
              n.id = arg.id
          rhs = function.body

        return AST_assign(lhs=assign_to, rhs=rhs)
            

      left_arg0 = patt.left_apply.call.get_obj().args[0]
      right_arg0 = patt.right_apply.call.get_obj().args[0]

      # Create a function. We'll add our code there, along with precondition
      # checks and we'll replace the apply() with a call to this function.

      # Local argument so it's fine in terms of name collisions.
      the_series = AST_name("ser")

      # We will only allow it to be a Series because otherwise
      # we have many cases on how to translate it into a loop.
      precond_is_Series = \
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[the_series]
        ),
        rhs=pd_Series_prefix,
        op=ast.Eq()
      )

      ### Generate loop ###

      loop_res = AST_name("res")
      loop_res_init = AST_assign(loop_res, ast.List(elts=[]))

      # Name for the loop iterator.
      loop_it = AST_name("it")
      # ls = <obj>.tolist()
      ls_name = AST_name("ls")
      ls = AST_assign(ls_name, AST_attr_call(called_on=the_series, name="tolist"))

      left_res = AST_name("left")
      left_call = call_function_and_assign_result(
        assign_to=left_res,
        function=left_arg0, arg=loop_it
      )

      right_res = AST_name("right")
      right_call = call_function_and_assign_result(
        assign_to=right_res,
        function=right_arg0,
        arg=left_res
      )
      fused_append = gen_append(append_to=loop_res, value=right_res)
      for_loop = AST_iter_container(
        it=loop_it, cont=ls_name, 
        body=[left_call, right_call, fused_append]
      )
      res_to_series = \
        AST_attr_call(ast.Name("pd"), "Series", args=[loop_res])
      return_fused = ast.Return(value=res_to_series)

      rewritten_body = [loop_res_init, ls, for_loop, return_fused]

      # Original: Call the two apply()'s on the series
      # Basically, set the called_on of the left apply to the series
      original_left_called_on = patt.left_apply.get_called_on()
      patt.left_apply.get_attr_ast().value = the_series
      original_apply = patt.right_apply.call.get_obj()

      # Wrap in if-else
      conditioned = \
        AST_if_else(
          cond=precond_is_Series,
          then_block=rewritten_body,
          else_block=[ast.Return(value=original_apply)]
        )

      # Create the function
      fused_apply_func_name = "fused_apply"
      # Check if it's already in use
      if fused_apply_func_name in globals():
        fused_apply_func_name = "_REWR_fused_apply"
        assert fused_apply_func_name not in globals()
      fused_apply_func = AST_function_def(fused_apply_func_name,
                                   args=[the_series.id], body=[conditioned])
      # Insert before the current (top-level) statement
      stmt_list.insert(0, fused_apply_func)

      # Finally, replace the call to the right apply with a called
      # to fused_apply_func, passing it as arg anything on the left
      # of the left apply
      patt.right_apply.call.set_enclosed_obj(
        AST_call(func=ast.Name(id=fused_apply_func_name),
        args=[original_left_called_on])
      )

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [fused_apply_func,
                                 patt.right_apply.call.get_obj()]}
      stats.append(patt_info)
    elif isinstance(patt, patt_matcher.LenUnique):
      orig_call = patt.enclosed_call.get_obj()
      orig_lineno = orig_call.lineno
      orig_end_lineno = orig_call.end_lineno
      
      new_call = AST_attr_call(
          called_on=dias_rewriter_prefix,
          name="len_unique",
          keywords={'series': patt.series}
        )
      patt.enclosed_call.set_enclosed_obj(new_call)

      patt_info = {"patt": type(patt).__name__,
                   "orig-lineno": orig_lineno,
                   "orig-end-lineno": orig_end_lineno,
                   "delegates": [new_call]}
      stats.append(patt_info)
    else:
      # No matched pattern, don't change stmt
      pass

  new_source = ""
  for stmt_list in list_of_lists:
    for stmt in stmt_list:
      new_source = new_source + astor.to_source(stmt)
    # END: for stmt ...
  # END: for stmt_list ...
  
  if line_info:
    # Convert list_of_lists to an ast.Module
    body = []
    for stmt_list in list_of_lists:
      for stmt in stmt_list:
        body.append(stmt)
      ### END FOR ###
    ### END FOR ###
    mod = ast.Module(body=body)

    parsed_rewritten = ast.parse(new_source)
    # parsed_rewritten should have the exact same structure as `mod`. We're
    # going to co-traverse them to find line information about nodes in `mod`
    # from `parsed_rewritten`.
    nodes_to_extract_info_for = set()
    for patt_info in stats:
      for delg in patt_info['delegates']:
        nodes_to_extract_info_for.add(delg)
      ### END FOR ###
    ### END FOR ###
    co_traverse_and_extract_line_info(mod, parsed_rewritten, nodes_to_extract_info_for)
    
    new_stats = []
    for patt_info in stats:
      delegates = patt_info['delegates']
      first_delg = delegates[0]
      last_delg = delegates[-1]
      assert hasattr(first_delg, "lineno")
      assert hasattr(last_delg, "end_lineno")
      new_patt_info = copy.deepcopy(patt_info)
      del new_patt_info['delegates']
      new_patt_info['rewr-lineno'] = first_delg.lineno
      new_patt_info['rewr-end-lineno'] = last_delg.end_lineno
      new_stats.append(new_patt_info)
    ### END FOR ###
    stats = new_stats
  else:
    # TODO: Just being backwards compatible with the previous format, but this
    # could be better.
    stats = [s['patt'] for s in stats]
  # END IF #

  return new_source, stats


# In call_rewrite(), we modify the code such that it calls rewrite(). Inside
# rewrite() we run code using ip.run_cell(). This function, however, calls the
# cell transformers, and thus call_rewrite() again. This leads to infinite
# recursion. We want to apply call_rewrite() only if the execution does _not_
# originate from Dias itself. So, we use a global to track whether we reached
# call_rewrite() from Dias.
_inside_dias = False

def rewrite_ast_from_source(cell, line_info=False):
  cell_ast = ast.parse(cell)
  return rewrite_ast(cell_ast, line_info=line_info)

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
