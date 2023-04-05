from IPython.core.magic import register_cell_magic
from IPython import get_ipython, InteractiveShell
from IPython.display import display, Markdown
import ast
import os
import sys
from typing import Dict, List, Union, Optional, Tuple, Any, Set
import pickle
import inspect
import json
import time
import pandas as pd
import numpy as np

### NON-DEFAULT PACKAGES: The user has to have these installed
import astor

import dias.patt_matcher as patt_matcher
import dias.nb_utils as nb_utils

############ CONFIGURATIONS ############

_IREWR_JSON_STATS=False
if "_IREWR_JSON_STATS" in os.environ and os.environ["_IREWR_JSON_STATS"] == "True":
  _IREWR_JSON_STATS=True

_IREWR_DISABLE_SLICED_EXEC=False
if "_IREWR_DISABLE_SLICED_EXEC" in os.environ and os.environ["_IREWR_DISABLE_SLICED_EXEC"] == "True":
  _IREWR_DISABLE_SLICED_EXEC=True

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

def get_mod_time(filepath):
  return os.path.getmtime(filepath)

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
    handlers=[ast.ExceptHandler(body=except_block)],
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

# If `name` exists in the current IPython namespace, check that
# its type is DataFrame
def check_DataFrame(name: str, ipython) -> bool:
  # There is a problem here in that the `name` may be created (as a DataFrame) and accessed
  # in the same cell. So, before running it, we can't know its type.
  # So, this function returns False if it _does_ exist, but it's a pandas.DataFrame.
  # Of course, it might be changed to a pandas.DataFrame in this cell, before the point at
  # which we want it to be a DataFrame. In this case, we bail, although we shouldn't. But
  # we're still sound.
  if name in ipython.user_ns and type(ipython.user_ns[name]) != pd.DataFrame:
    return False
  return True

# Return the func_ast and the name of the first argument.
def get_func_ast_and_arg_name(func_name: str, ipython: InteractiveShell) -> Tuple[ast.FunctionDef, str]:
  # We have executed all the code up to here, so it should be available.
  assert func_name in ipython.user_ns
  func_obj = ipython.user_ns[func_name]
  # I don't of a way to get the AST directly.
  func_source = inspect.getsource(func_obj)
  mod_ast = ast.parse(func_source)
  func_ast = mod_ast.body[0]
  assert isinstance(func_ast, ast.FunctionDef)
  # Either we have one argument or if we have more, they
  # have default values. We must have only one argument that
  # we pass because we have checked that the apply gets a single
  # argument as its first argument, which is a name, this name
  # will be called with a single argument. Also, note that in Python
  # all default arguments should be after all non-default. In other words,
  # you cannot e.g., default, non-default.
  assert (len(func_ast.args.args) == 1 or 
          len(func_ast.args.defaults) == (len(func_ast.args.args) - 1))
  arg0_name = func_ast.args.args[0].arg
  return func_ast, arg0_name

def rewrite_remove_axis_1(remove_axis_1: patt_matcher.RemoveAxis1, 
                          func_ast: ast.FunctionDef, arg0_name: str,
                          the_one_series: str) -> str:
  # All the following modify `blocking_stmt`
  # implicitly
  # Remove axis=1
  call = remove_axis_1.apply_call.call_name.attr_call.call.get_obj()
  kwargs = call.keywords
  axis_1_idx = -1
  for idx, kw in enumerate(kwargs):
    if kw.arg == "axis":
      axis_1_idx = idx
      break
  assert axis_1_idx != -1
  call.keywords.pop(axis_1_idx)
  # Make row.apply -> row[Name].apply
  call_name = remove_axis_1.apply_call.call_name
  attr_ast = call_name.attr_call.get_attr_ast()
  attr_ast.value = AST_sub_const(AST_name(call_name.get_name()), the_one_series)
  # Modify all the subscripts in the function.
  # Unfortunately, we have to do another walk here
  # TODO: Find a way to fix that because it looks horrible.
  # I tried saving the parent of `n` before next iteration of ast.walk()
  # But ast.walk() does not tell which attribute it uses to visit the note.
  # astor may be able to help with that by inserting code that is executed
  # before children are visited: https://astor.readthedocs.io/en/latest/
  for n in ast.walk(func_ast):
    subs = patt_matcher.search_enclosed(n, ast.Subscript)
    for enclosed_sub in subs:
      sub = enclosed_sub.get_obj()
      compat_sub = patt_matcher.is_compatible_sub(sub)
      if compat_sub is None:
        continue
      if (compat_sub.get_Series() == the_one_series and
          compat_sub.get_df() == arg0_name):
        enclosed_sub.set_enclosed_obj(ast.Name(id=arg0_name))

  # Change the function name
  new_func_name = get_unique_id()
  func_ast.name = new_func_name
  new_func_source = astor.to_source(func_ast)
  call_name.attr_call.call.get_obj().args[0] = ast.Name(id=new_func_name)

  return new_func_source

def has_only_math__numeric_ty(ty) -> bool:
  numeric_types = {int, float, np.int64, np.float64}
  # NOTE: For some reason we have to do a for loop.
  # I couldn't make it work by putting all acceptable
  # types in a set and checking if `dt` is in the set. 
  for num_dt in numeric_types:
    if ty == num_dt:
      return True
  return False


def has_only_math__preconds(called_on: ast.AST,
                            default_args: List[ast.expr], subs: List[str],
                            external_names: List[str],
                            ipython: InteractiveShell) -> bool:
  # For simplicity, check that the object the .apply()
  # is called on is a Name, and then verify it's a DataFrame.
  if not isinstance(called_on, ast.Name):
    return False
  the_df = ipython.user_ns[called_on.id]
  if not isinstance(the_df, pd.DataFrame):
    return False
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
      body=AST_sub_const(obj_to_idx, idx),
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

def str_split__wrap_in_if(sub_to_split: ast.Subscript,
                          then: List[ast.stmt],
                          else_: List[ast.stmt]):
  return AST_if_else(
    cond=
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[sub_to_split]
        ),
        rhs=AST_attr_chain("pd.Series"),
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

def sliced_execution(list_of_lists: List[List[ast.stmt]],
                     ipython: InteractiveShell,
                     deferred_patts: Dict[ast.stmt, List[Tuple[patt_matcher.Available_Patterns, List[int]]]],
                     stats: dict()):
  time_spent_in_exec = 0

  new_source = ""
  curr_idx = 0
  while curr_idx < len(list_of_lists):
    # Build the code slice that is not blocked by a statement.
    code_slice = ""
    while curr_idx < len(list_of_lists) and list_of_lists[curr_idx][0] not in deferred_patts:
      stmt_list = list_of_lists[curr_idx]
      for stmt in stmt_list:
        code_slice = code_slice + astor.to_source(stmt)
      curr_idx = curr_idx + 1

    new_source = new_source + code_slice
    if curr_idx == len(list_of_lists):
      # That was the last slice. Execute it and exit.
      start = time.perf_counter_ns()
      ipython.run_cell(code_slice)
      end = time.perf_counter_ns()
      time_spent_in_exec = time_spent_in_exec + (end-start)
      break
    
    # Execute the slice
    start = time.perf_counter_ns()
    ipython.run_cell(code_slice)
    end = time.perf_counter_ns()
    time_spent_in_exec = time_spent_in_exec + (end-start)
    
    # Deal with the blocking statement.
    blocking_stmt = list_of_lists[curr_idx][0]
    assert blocking_stmt in deferred_patts
    ls_of_patts = deferred_patts[blocking_stmt]
    for patt, stmt_idxs in ls_of_patts:
      if isinstance(patt, patt_matcher.ApplyCallMaybeRemoveAxis1):
        func_to_call = patt.apply_call.get_func_to_call()
        called_on = patt.apply_call.call_name.attr_call.get_called_on()
        func_ast, arg0_name = get_func_ast_and_arg_name(func_to_call, ipython)
        all_arg_names = [arg.arg for arg in func_ast.args.args]
        buf_list = []
        buf_set = set()
        if patt_matcher.can_be_vectorized_func(func_ast):
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
          # END OF handle_if()
          cond_value_pairs = []
          # This will have a single element but basically we're emulating
          # a pointer.
          else_val = []
          rewrite_if(func_ast.body[0], arg0_name, called_on, cond_value_pairs, else_val)
          assert len(else_val) == 1

          # Replace the apply()
          patt.apply_call.call_name.attr_call.call.set_enclosed_obj(
            vec__build_np_select(cond_value_pairs, else_val)
          )

          stats["ApplyVectorized"] = 1
        elif patt_matcher.has_only_math(func_ast, all_arg_names, subs=buf_list, external_names=buf_set):
          subs = buf_list
          external_names = buf_set
          attr_call = patt.apply_call.call_name.attr_call
          called_on = attr_call.get_called_on()
          default_args = func_ast.args.defaults
          if has_only_math__preconds(called_on, default_args,
                                     subs, external_names, ipython):
            # Just call the original function with the df as the first
            # argument
            # Replace the call to apply with a call to our new function.
            the_series = attr_call.get_called_on()
            assert isinstance(the_series, ast.Name)
            attr_call.call.set_enclosed_obj(
              AST_call(func=AST_name(func_ast.name), args=[AST_name(the_series.id)])
            )
            stats["ApplyOnlyMath"] = 1
        elif patt.get_kind() == patt_matcher.ApplyCallMaybeRemoveAxis1.Kind.REMOVE_AXIS_1:
          remove_axis_1 = patt.get_remove_axis_1()
          passed_checks, the_one_series = patt_matcher.can_remove_axis_1(func_ast, arg0_name)
          if passed_checks:
            new_func_source = rewrite_remove_axis_1(remove_axis_1, func_ast, arg0_name, the_one_series)
            stats["RemovedAxis1"] = 1
            start = time.perf_counter_ns()
            ipython.run_cell(new_func_source)
            end = time.perf_counter_ns()
            time_spent_in_exec = time_spent_in_exec + (end-start)
            new_source = new_source + new_func_source
        else:
          # Don't do anything
          pass

        stmt_source = astor.to_source(blocking_stmt)
        start = time.perf_counter_ns()
        ipython.run_cell(stmt_source)
        end = time.perf_counter_ns()
        time_spent_in_exec = time_spent_in_exec + (end-start)
        new_source = new_source + stmt_source
      else:
        # It must be a specific pattern
        assert 0
    # END: for patt, stmt_idxs ...
    curr_idx = curr_idx + 1
  # END: while curr_idx < len(list_of_lists):
  return new_source, stats, time_spent_in_exec

# I don't think we can declare a type for `ipython`
def rewrite_and_exec(cell_ast: ast.Module, ipython: InteractiveShell) -> Tuple[str, Dict, Dict]:
  assert isinstance(cell_ast, ast.Module)
  body = cell_ast.body
  matched_patts = patt_matcher.patt_match(body)

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
  #
  # And now, to go to my final point, generally I don't think that we will have context-sensitive
  # patterns _in terms of code_. So, some patterns will be context-sensitive. For example,
  # the ApplyAxis1 needs to know the code of the function. And maybe the function is defined
  # or re-defined in the same cell. But, I think that if you start basing your rewrites
  # on static analysis, you will fail fast because of the craziness of Python. Rather,
  # I think we should base ourselves on runtime checks. For example, for ApplyAxis1, we should
  # execute all the code up to this point, then get the code of the function object passed
  # to apply(). And that's why I don't think it's worth worrying about the order of patterns.
  #
  # What we should have though is the concept of "deferred" patterns. These are patterns
  # that require all the code to have been executed up to a statement, before the can be processed.
  # So, we then slice the body on these deferred statements and we proceed as follows: execute a slice,
  # process the deferred pattern, execute next slice, process the other deferred pattern etc.

  # For each statement, save a list of patterns that can be executed only after the statement is
  # executed.
  deferred_patts: Dict[ast.stmt, List[Tuple[patt_matcher.Available_Patterns, List[int]]]] = dict()

  #
  # First, execute all the context-insensitive patterns (read above; those that do not need
  # to be deferred) and gather the deferred ones.
  #

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
  stats = dict()
  for patt, stmt_idxs in matched_patts:
    if patt is not None and type(patt).__name__ in disabled_patts:
      print(type(patt).__name__)
      continue

    if isinstance(patt, patt_matcher.HasSubstrSearchApply):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      stmt = stmt_list[0]

      df = patt.series_call.get_sub().get_df()
      if not check_DataFrame(df, ipython):
        continue

      sub = patt.series_call.get_sub().get_sub_ast()

      tmp_sub_id: str = get_unique_id()
      tmp_sub_name = ast.Name(id=tmp_sub_id)
      tmp_sub_asgn = ast.Assign(targets=[tmp_sub_name], value=sub)

      listified_id: str = get_unique_id()
      listified = ast.Name(id=listified_id)

      listify_stmt = \
        ast.Assign(targets=[listified],
          value=AST_attr_call(called_on=tmp_sub_name, name='tolist'))

      pred_id: str = get_unique_id()
      pred = ast.Name(id=pred_id)
      pred_stmt = \
      ast.Assign(targets=[pred],
        value=ast.ListComp(
          elt=patt.compare,
          generators=[ast.comprehension(target=patt.get_needle(), iter=listified, ifs=[], is_async=0)]),
        type_comment=None)

      stmt_list[0] = tmp_sub_asgn
      stmt_list.append(listify_stmt)
      stmt_list.append(pred_stmt)
      stmt_list.append(stmt)

      # This implicitly updates `stmt`
      opt_call = patt.series_call.attr_call.call
      res_to_series = \
        AST_attr_call(ast.Name("pd"), 
                      ast.Name("Series"), args=[pred])
      res_to_series.keywords = [ast.keyword(arg="index", value=ast.Attribute(value=tmp_sub_name, attr="index"))]
      opt_call.set_enclosed_obj(res_to_series)

      assert isinstance(patt, patt_matcher.HasSubstrSearchApply)


      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.IsInplaceUpdate):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      stmt = stmt_list[0]

      df = patt.series_call.get_sub().get_df()
      if not check_DataFrame(df, ipython):
        continue

      # Let's limit it to top-level for now
      # TODO: Can we remove that?
      if patt.assign != stmt:
        continue
      series_call: patt_matcher.SeriesCall = patt.series_call

      func = series_call.attr_call.get_func()
      # TODO: For now, only do it for `fillna` because I haven't tested
      # the rest.
      if func != "fillna":
        continue
      call: ast.Call = series_call.attr_call.call.get_obj()
      call.keywords.append(ast.keyword(arg='inplace', value=ast.Constant(value=True, kind=None)))
      stmt_list[0] = ast.Expr(value=call)

      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.HasToListConcatToSeries):
      # Pass for now
      pd_Series_call = patt.enclosed_call.get_obj()
      # Change the call to concat.
      pd_Series_call.func.attr = 'concat'
      pd_concat_call = pd_Series_call
      left_df = patt.left_ser_call.get_sub().get_df()
      left_ser = patt.left_ser_call.get_sub().get_Series()
      right_df = patt.right_ser_call.get_sub().get_df()
      right_ser = patt.right_ser_call.get_sub().get_Series()
      pd_concat_call.args[0] = \
        ast.List(
          elts=[ast.Subscript(value=ast.Name(id=left_df), slice=ast.Constant(value=left_ser)),
                ast.Subscript(value=ast.Name(id=right_df), slice=ast.Constant(value=right_ser))]
        )
      pd_concat_call.keywords = [ast.keyword(arg="ignore_index", value=ast.Constant(value=True))]
      # The above implicitly modifies stmt

      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.SortHead):
      # Replace the call to sort_values() with nsmallest/nlargest.

      func = "nsmallest"
      if not patt.is_sort_ascending():
        func = "nlargest"
      
      n = patt.get_head_n()
      by = patt.get_sort_by()

      new_call = \
        AST_attr_call(
          called_on=patt.sort_values.get_called_on(),
          name=func,
          keywords={'n': n, 'columns': by}
        )
      patt.head.call.set_enclosed_obj(new_call)

      astor.to_source(patt.head.call.get_obj())

      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.ApplyCallMaybeRemoveAxis1):
      # We can't process it until we know the function passed to apply().
      # Save it as a deferred pattern.
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      stmt = stmt_list[0]

      if stmt not in deferred_patts:
        deferred_patts[stmt] = []
      deferred_patts[stmt].append((patt, stmt_idxs))

      # Don't log it into the stats here. Only if we apply it when processing
      # deferred stmts.
    elif isinstance(patt, patt_matcher.ReplaceRemoveList):
      # We can't process it until we know the function passed to apply().
      # Save it as a deferred pattern.
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      
      to_replace = patt.to_replace
      replace_with = patt.replace_with
      call_raw = patt.attr_call.call.get_obj()
      call_raw.args[0] = to_replace
      call_raw.args[1] = replace_with

      if patt.inplace:
        # The parent of the call must be an assignment.
        # This assignment must be top-level because then we
        # can replace it in the stmt_list[]. Otherwise, we don't
        # know the parent of the assignment and if an assignment
        # is not top-level, we probably don't want to touch it
        # anyway.
        if patt.attr_call.call.get_encloser() == stmt_list[0]:
          # Add the inplace=True argument.
          call_raw.keywords.append(
            ast.keyword(arg="inplace", value=ast.Constant(value=True)))
          stmt_list[0] = ast.Expr(value=call_raw)

      stats[type(patt).__name__] = 1
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
        str_in_col.cmp_encl.set_enclosed_obj(ast.parse(new_expr, mode='eval'))
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
      stmt_list[0] = ast.parse(index_contains_func_str)
      stmt_list.append(save_stmt)

      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.FuseIsIn):
      call = patt.the_call
      call.args[0] = ast.BinOp(left=patt.left_name, op=ast.Add(), right=patt.right_name)
      patt.binop_encl.set_enclosed_obj(call)
      
      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.StrSplitPython):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]
      stmt = stmt_list[0]

      # TODO: Possibly set maxsplits (in split()) to idx_from_split_list+1
      # to save allocations and time.

      # Otherwise, we wouldn't have a gotten StrSplitPython directly.
      assert patt.expand_true == True

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

      targ0_append = gen_append(target_names[0], AST_sub_const(spl_name, 0))
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
            lhs=AST_sub_const(patt.lhs_root_name, patt.lhs_targets[idx].value),
            rhs=target_names[idx]
          )
        assign_series.append(assign)

      rewritten_body = init_lists + [ls] + [for_loop] + assign_series

      ### Wrap in if-else block ###

      new_stmt = \
        str_split__wrap_in_if(sub_to_split, then=[stmt_list[0]], else_=rewritten_body)
      stmt_list[0] = new_stmt

      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.FusableStrSplit):
      assert len(stmt_idxs) == 2
      split_stmt_idx = stmt_idxs[0]
      split_stmt_list = list_of_lists[split_stmt_idx]

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
        value_to_append = AST_sub_const(spl_name, idx_from_split_list)
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
        str_split__wrap_in_if(sub_to_split,
                              then=[assign_orig_res],
                              else_=rewritten_body)
      # Replace the split with the new_stmt
      split_stmt_list[0] = new_stmt
      # Replace the index expression with the final result
      patt.expr_to_replace.set_enclosed_obj(res_name)

      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.FusableReplaceUnique):
      assert len(stmt_idxs) == 2
      replace_stmt_idx = stmt_idxs[0]
      replace_stmt_list = list_of_lists[replace_stmt_idx]
      unique_stmt_idx = stmt_idxs[1]
      unique_stmt_list = list_of_lists[unique_stmt_idx]

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
        rhs=AST_attr_chain("pd.Series"),
        op=ast.Eq()
      )

      # The argument to replace() should be a dict
      precond_is_dict = \
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[patt.replace_arg]
        ),
        rhs=AST_name("dict"),
        op=ast.Eq()
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
      
      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.ApplyVectorizedLambda):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]

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
        rhs=AST_attr_chain("pd.DataFrame"),
        op=ast.Eq()
      )

      np_select = vec__build_np_select(cond_value_pairs, else_val)
      # Wrap it in pd.Series(). The conversion to Series will happen
      # implicitly if we have sth like this:
      #   df['x'] = df.apply(...)
      # but in the following we'll have a problem:
      #   ... = df.apply(...).str.replace()
      np_select_to_series = \
        AST_attr_call(ast.Name("pd"), 
                      ast.Name("Series"), args=[np_select])

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
      patt.apply_call.call.set_enclosed_obj(
        AST_call(func=ast.Name(id=apply_vec_name), args=[the_series])
      )

      stats[type(patt).__name__] = 1
    
    elif isinstance(patt, patt_matcher.RemoveAxis1Lambda):
      # Remove axis=1
      call = patt.call_name.attr_call.call.get_obj()
      kwargs = call.keywords
      axis_1_idx = -1
      for idx, kw in enumerate(kwargs):
        if kw.arg == "axis":
          axis_1_idx = idx
          break
      assert axis_1_idx != -1
      call.keywords.pop(axis_1_idx)
      # Make row.apply -> row[Name].apply
      attr_ast = patt.call_name.attr_call.get_attr_ast()
      attr_ast.value = \
        AST_sub_const(AST_name(patt.call_name.get_name()), patt.the_one_series)
      # Modify all the subscripts in the function.
      arg0_name = patt.lam.args.args[0].arg
      for n in ast.walk(patt.lam.body):
        subs = patt_matcher.search_enclosed(n, ast.Subscript)
        for enclosed_sub in subs:
          sub = enclosed_sub.get_obj()
          compat_sub = patt_matcher.is_compatible_sub(sub)
          if compat_sub is None:
            continue
          if (compat_sub.get_Series() == patt.the_one_series and
              compat_sub.get_df() == arg0_name):
            enclosed_sub.set_enclosed_obj(AST_name(arg0_name))
      
      stats[type(patt).__name__] = 1
    elif isinstance(patt, patt_matcher.FuseApply):
      assert len(stmt_idxs) == 1
      stmt_idx = stmt_idxs[0]
      stmt_list = list_of_lists[stmt_idx]

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

      the_series = AST_name("_REWR_ser")

      # We will only allow it to be a Series because otherwise
      # we have many cases on how to translate it into a loop.
      precond_is_Series = \
      AST_cmp(
        lhs=AST_call(
          func=AST_name("type"),
          args=[the_series]
        ),
        rhs=AST_attr_chain("pd.Series"),
        op=ast.Eq()
      )

      ### Generate loop ###

      loop_res = AST_name("_REWR_res")
      loop_res_init = AST_assign(loop_res, ast.List(elts=[]))

      # Name for the loop iterator.
      loop_it = AST_name("_REWR_it")
      # _REWR_ls = <obj>.tolist()
      ls_name = AST_name("_REWR_ls")
      ls = AST_assign(ls_name, AST_attr_call(called_on=the_series, name="tolist"))

      left_res = AST_name("_REWR_left")
      left_call = call_function_and_assign_result(
        assign_to=left_res,
        function=left_arg0, arg=loop_it
      )

      right_res = AST_name("_REWR_right")
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
        AST_attr_call(ast.Name("pd"), 
                      ast.Name("Series"), args=[loop_res])
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
      fused_apply_func_name = "_REWR_fused_apply"
      fused_apply_func = AST_function_def(fused_apply_func_name,
                                   args=[the_series], body=[conditioned])
      # Insert before the current (top-level) statement
      stmt_list.insert(0, fused_apply_func)

      # Finally, replace the call to the right apply with a called
      # to fused_apply_func, passing it as arg anything on the left
      # of the left apply
      patt.right_apply.call.set_enclosed_obj(
        AST_call(func=ast.Name(id=fused_apply_func_name),
        args=[original_left_called_on])
      )

      stats[type(patt).__name__] = 1
    else:
      # No matched pattern, don't change stmt
      pass
  # Now, start interleaving execution and rewrites.

  if not _IREWR_DISABLE_SLICED_EXEC:
    return sliced_execution(list_of_lists, ipython, deferred_patts, stats)

  # Don't slice the execution. Just execute all the statements.
  # TODO: It seems this would be faster with some list comprehension.
  new_source = ""
  for stmt_list in list_of_lists:
    for stmt in stmt_list:
      new_source = new_source + astor.to_source(stmt)
    # END: for stmt ...
  # END: for stmt_list ...
  start = time.perf_counter_ns()
  ipython.run_cell(new_source)
  end = time.perf_counter_ns()
  time_spent_in_exec = end-start

  return new_source, stats, time_spent_in_exec



# In call_rewrite(), we modify the code such that it calls rewrite(). Inside
# rewrite() we run code using ip.run_cell(). This function, however, calls the
# cell transformers, and thus call_rewrite() again. This leads to infinite
# recursion. We want to apply call_rewrite() only if the execution does _not_
# originate from Dias itself. So, we use a global to track whether we reached
# call_rewrite() from Dias.
_inside_dias = False


def rewrite(line: str, cell: str):
  global _inside_dias
  _inside_dias = True

  # TODO: Is this a good idea or we should ask it every time we want to use it?
  ipython = get_ipython()

  # Note: `cell` has the raw source code, but it doesn't include %%rewrite
  # dbg_print("--------- Original AST -----------")
  cell_ast = ast.parse(cell)
  # dbg_print(astor.dump(cell_ast))
  # dbg_print("--------- Modify AST -----------")
  new_source, hit_stats, time_spent_in_exec = rewrite_and_exec(cell_ast, ipython)
  if line.strip() == "verbose":
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
  if _IREWR_JSON_STATS:
    eprint("[IREWRITE JSON]")
    json_out = dict()
    # NOTE: The original and modified codes might be the same but because the modified code was round-tripped
    # using astor, the strings might not be the same. If you want to test for equality, don't
    # just compare the two strings. Rather, round-trip the original source too and then test.
    json_out['raw'] = cell
    json_out['modified'] = new_source
    json_out['patts-hit'] = hit_stats
    # In ns.
    json_out['rewritten-exec-time'] = nb_utils.ns_to_ms(time_spent_in_exec)
    dumped_stats = json.dumps(json_out, indent=2)
    eprint(dumped_stats)
    eprint("[IREWRITE END JSON]")

  _inside_dias = False
  return None

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
ip.input_transformers_cleanup.append(call_rewrite)