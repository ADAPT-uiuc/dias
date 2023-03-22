from os import remove
from tabnanny import check
from typing import Dict, Final, List, Union, Optional, Tuple, Any, Set, TypeVar, Generic, Type
import ast
from enum import Enum

from pandas import Series
import astor

def _a(e: bool):
  return e

############ PATTERN RECOGNIZERS ############

_T = TypeVar('_T')

# Optionally enclosed object. Why:
# Suppose we want to match a call.
# We can search for a Call. But the problem
# is that then we need to replace this Call
# with our cached thing. Now, the problem is
# that this Call is basically a pointer. We need
# to change the pointer to this pointer, and we cannot
# do that. So, we save the parent and the attribute that has the Call.
# Then, we can change the value of this attribute
class OptEnclosed(Generic[_T]):
  class Kind(Enum):
    NOT_ENCLOSED = 0,
    ATTR = 1,
    LIST = 2

  def __init__(self, kind: Kind, obj, encloser, idx: int, access_attr: str, ty: Type, dont_use_the_constructor_directly: bool) -> None:
    assert _a(dont_use_the_constructor_directly)
    
    self.kind = kind
    if kind == OptEnclosed.Kind.ATTR:
      self.encloser = encloser
      self.access_attr = access_attr
      # TODO: It appears that you can't do and I'm not sure there's a workaround.
      # assert isinstance(getattr(self.encloser, self.access_attr), _T)
      # So, we have to pass the type as an argument and be completely weird:
      assert _a(isinstance(getattr(self.encloser, self.access_attr), ty))
    elif kind == OptEnclosed.Kind.LIST:
      self.encloser = encloser
      self.idx = idx
      assert _a(isinstance(self.encloser, list))
      assert _a(isinstance(self.encloser[idx], ty))
    else:
      self.obj = obj
      assert _a(isinstance(obj, ty))

  def get_obj(self) -> _T:
    if self.kind == OptEnclosed.Kind.ATTR:
      return getattr(self.encloser, self.access_attr)
    elif self.kind == OptEnclosed.Kind.LIST:
      return self.encloser[self.idx]
    else:
      assert _a(self.kind == self.Kind.NOT_ENCLOSED)
      return self.obj

  def get_encloser(self) -> Any:
    assert _a(self.kind != self.Kind.NOT_ENCLOSED)
    return self.encloser

  def get_access_attr(self) -> str:
    assert _a(self.kind == self.Kind.ATTR)
    return self.access_attr

  def set_enclosed_obj(self, obj):
    assert _a(self.kind != self.Kind.NOT_ENCLOSED)
    if self.kind == OptEnclosed.Kind.ATTR:
      setattr(self.encloser, self.access_attr, obj)
    elif self.kind == OptEnclosed.Kind.LIST:
      self.encloser[self.idx] = obj
  
  def is_enclosed_list(self) -> bool:
    return self.kind == OptEnclosed.Kind.LIST
  
  def is_enclosed_attr(self) -> bool:
    return self.kind == OptEnclosed.Kind.ATTR

  def is_enclosed(self) -> bool:
    return self.is_enclosed_attr() or self.is_enclosed_list()

  def is_enclosed_in_ty(self, ty: type) -> bool:
    return self.is_enclosed() and isinstance(self.get_encloser(), ty)

def get_non_enclosed(obj) -> OptEnclosed:
  return OptEnclosed(kind=OptEnclosed.Kind.NOT_ENCLOSED, obj=obj, encloser=None, idx=-1, access_attr="", ty=type(obj), dont_use_the_constructor_directly=True)

def get_enclosed_list(encloser: List, idx: int) -> OptEnclosed:
  assert _a(isinstance(encloser, list))
  ty = type(encloser[idx])
  return OptEnclosed(kind=OptEnclosed.Kind.LIST, obj=None, encloser=encloser, idx=idx,
                     access_attr="", ty=ty, dont_use_the_constructor_directly=True)

def get_enclosed_attr(encloser, access_attr: str):
  ty = type(getattr(encloser, access_attr))
  return OptEnclosed(kind=OptEnclosed.Kind.ATTR, obj=None, encloser=encloser, 
                     access_attr=access_attr, idx=-1, ty=ty, dont_use_the_constructor_directly=True)

def search_enclosed(n: ast.AST, ty_to_search: type) -> List[OptEnclosed]:
  # This might be a little slow... But it allows us to match more patterns
  # than having 2-3 predefined possible attributes.

  res: List[OptEnclosed] = []
  # Silence mypy
  assert isinstance(ty_to_search, type)

  possible_attrs = [att for att in dir(n) if not att.startswith("__")]

  for access_attr in possible_attrs:
    obj = getattr(n, access_attr)
    if isinstance(obj, ty_to_search):
      res.append(get_enclosed_attr(n, access_attr))
    elif isinstance(obj, list):
      ls = obj
      for idx, el in enumerate(ls):
        # We don't need to recurse to search_enclosed() here because
        # lists won't generally have lists. They will have AST objects
        # which have attributes.
        if isinstance(el, ty_to_search):
          res.append(get_enclosed_list(ls, idx))
  
  return res

class AttrCall:
  def __init__(self, call: OptEnclosed[ast.Call], dont_use_the_constructor_directly: bool) -> None:
    self.call = call
    assert _a(dont_use_the_constructor_directly)

    # Checks
    assert _a(isinstance(self.call.get_obj().func, ast.Attribute))

  def get_attr_ast(self) -> ast.Attribute:
    # Silence mypy
    attr = self.call.get_obj().func
    assert isinstance(attr, ast.Attribute)
    return attr

  def get_called_on(self) -> ast.expr:
    attr_ast = self.get_attr_ast()
    return attr_ast.value
  
  def get_func(self) -> str:
    attr_ast = self.get_attr_ast()
    return attr_ast.attr

def is_attr_call_unkn(encl: OptEnclosed) -> Optional[AttrCall]:
  if not isinstance(encl.get_obj(), ast.Call):
    return None
  return is_attr_call(encl)

def is_attr_call(opt_call: OptEnclosed[ast.Call]) -> Optional[AttrCall]:
  call = opt_call.get_obj()
  if (isinstance(call.func, ast.Attribute)):
    return AttrCall(call=opt_call, dont_use_the_constructor_directly=True)
  return None

# The general strategy is:
# - Every field is represented by a list.
# - If the list has one element, then this field is terminal. It doesn't necessarily
#   mean it has no child fields, but if it does, we're not interested in visiting them.
# - Otherwise, the second element of this list is a list of all the
#   the child fields we should check.
# TODO: It doesn't work if the field has a list as a value (e.g., ast.Call.args)
def helper(root, path, types):
  assert isinstance(path, list)
  assert isinstance(types, list)
  assert len(path) == len(types)
  
  if len(path) == 1:
    f = getattr(root, path[0])
    return isinstance(f, types[0])
  else:
    root = getattr(root, path[0])
    if not isinstance(root, types[0]):
      return False
    for idx, p in enumerate(path[1]):
      if not helper(root, p, types[1][idx]):
        return False
  return True

# The start of the recursion is special because the `root` is
# an object and not a field of some other object.
# See helper() for the canonical case.
# TODO: Create a similar matcher but instead of types, match values.
def is_instance_chain(root, path, types) -> bool:
  assert len(path) == len(types[1])
  if not isinstance(root, types[0]):
    return False
  types = types[1]
  for idx, p in enumerate(path):
    if not helper(root, p, types[idx]):
      return False
  return True

# Meaning a.foo() not anything like a['b'].foo()
class CallOnName:
  def __init__(self, attr_call: AttrCall, dont_use_the_constructor_directly: bool) -> None:
    self.attr_call = attr_call
    assert dont_use_the_constructor_directly

  def get_name(self) -> str:
    called_on = self.attr_call.get_called_on()
    assert isinstance(called_on, ast.Name)
    return called_on.id

def is_call_on_name(attr_call: AttrCall) -> Optional[CallOnName]:
  attr_ast = attr_call.get_attr_ast()
  if isinstance(attr_ast.value, ast.Name):
    return CallOnName(attr_call=attr_call, dont_use_the_constructor_directly=True)
  return None

class IsTrivialDFCall:
  # NOTE: 'solution' and 'hint' are functions from Kaggle to help users.
  # TODO: We have to be careful with plot functions because they can contain a lot of code inside.
  TRIVIAL_FUNCTIONS = {'head', 'describe', 'summary', 'plot', 'scatterplot', 'plot_top_losses', 'catplot', 'boxplot', 'pairplot', 'pointplot', 'jointplot', 'lmplot', 'hist', 'distplot', 'countplot', 'info', 'show', 'sample', 'solution', 'hint', 'tail', 'sample'}
  def __init__(self) -> None:
    pass

def is_trivial_df_call(n: ast.stmt) -> bool:
  if not isinstance(n, ast.Expr):
    return False
  call = n.value
  if not isinstance(call, ast.Call):
    return False
  attr_call = is_attr_call(get_enclosed_attr(n, "value"))
  if attr_call is None:
    return False
  call_name = is_call_on_name(attr_call)
  if call_name is None:
    return False
  # Obviously, the caller should make sure that the type of the name is a DataFrame
  if call_name.attr_call.get_func() in IsTrivialDFCall.TRIVIAL_FUNCTIONS:
    return True
  return False

# TODO: Here, we assume that any attribute is a trivial computation. This is not entirely true...
# Attributes can actually call functions when computed (and even possibly cached later).
class IsTrivialDFAttr:
  def __init__(self) -> None:
    pass

def is_trivial_df_attr(n: ast.stmt) -> bool:
  if not isinstance(n, ast.Expr):
    return False
  attr = n.value
  if not isinstance(attr, ast.Attribute):
    return False

  if isinstance(attr.value, ast.Name) and isinstance(attr.attr, str):
    return True
  return False

class TrivialName:
  def __init__(self) -> None:
    pass

class TrivialCall:
  TRIVIAL_FUNCS = {'print', 'display', 'len', 'HTML', 'dhtml', 'type', 'help'}
  def __init__(self) -> None:
    pass

def is_trivial_call(n: ast.stmt) -> bool:
  if not isinstance(n, ast.Expr):
    return False
  call = n.value
  if not isinstance(call, ast.Call):
    return False

  func = call.func
  if not isinstance(func, ast.Name) or func.id not in TrivialCall.TRIVIAL_FUNCS:
    return False

  if len(call.args) != 1:
    return False

  if isinstance(call.args[0], ast.Constant) or isinstance(call.args[0], ast.Name):
    return True

  return False

class CompatSub:
  def __init__(self, sub: ast.Subscript, dont_use_the_constructor_directly: bool) -> None:
    self.sub = sub
    assert _a(dont_use_the_constructor_directly)

    # Checks
    # TODO: here we would like to call is_compatible_sub(). This pattern
    # happens a lot. But if we do, we will end up in an infinite recursion.
    # The alternative would be to have this method as a static method
    # that returns an instance. But we can't do that because apparently
    # you can't put e.g., CompatSub as the return type of one of its
    # methods.
    # That's why we have this weird argument in the constructors.

  def get_df(self) -> str:
    # Silence mypy
    assert isinstance(self.sub.value, ast.Name)
    return self.sub.value.id

  def get_Series(self) -> str:
    # Silence mypy
    assert isinstance(self.sub.slice, ast.Constant)
    return self.sub.slice.value

  def get_sub_ast(self) -> ast.Subscript:
    return self.sub

def compat_sub_eq(lhs: CompatSub, rhs: CompatSub) -> bool:
  return (lhs.get_df() == rhs.get_df() and
          lhs.get_Series() == rhs.get_Series())
    

def is_compatible_sub(sub: ast.Subscript) -> Optional[CompatSub]:
  df_Name = sub.value
  col_Const = sub.slice
  if not isinstance(df_Name, ast.Name):
    return None
  if not isinstance(col_Const, ast.Constant):
    return None
  return CompatSub(sub=sub, dont_use_the_constructor_directly=True)


def is_compatible_sub_unkn(n: ast.AST) -> Optional[CompatSub]:
  if not isinstance(n, ast.Subscript):
    return None
  return is_compatible_sub(n)

class SeriesCall:
  def __init__(self, attr_call: AttrCall, dont_use_the_constructor_directly: bool) -> None:
    self.attr_call = attr_call
    assert _a(dont_use_the_constructor_directly)

  def get_sub(self) -> CompatSub:
    sub = self.attr_call.get_called_on()
    # Silence mypy
    assert isinstance(sub, ast.Subscript)
    assert _a(is_compatible_sub(sub) is not None)
    return CompatSub(sub=sub, dont_use_the_constructor_directly=True)

def is_Series_call(attr_call: AttrCall) -> Optional[SeriesCall]:
  sub = attr_call.get_called_on()
  if not isinstance(sub, ast.Subscript):
    return None
  compat_sub = is_compatible_sub(sub)
  if compat_sub is None:
    return None
  return SeriesCall(attr_call=attr_call, dont_use_the_constructor_directly=True)

def is_Series_call_from_encl_call(encl_call: OptEnclosed[ast.Call]) -> Optional[SeriesCall]:
  attr_call = is_attr_call(encl_call)
  if attr_call is None:
    return None
  return is_Series_call(attr_call)

def is_Series_call_unkn(n: ast.AST) -> Optional[SeriesCall]:
  if not isinstance(n, ast.Call):
    return None

class HasSubstrSearchApply:
  # We could compute all the arguments instead of storing them, but it will be a lot of
  # horrible code.
  def __init__(self, series_call: SeriesCall, compare: ast.Compare, dont_use_the_constructor_directly: bool) -> None:
    self.series_call = series_call
    self.compare = compare
    assert _a(dont_use_the_constructor_directly)

    # Checks
    func = series_call.attr_call.get_func()
    assert _a(func == "apply")

  def get_needle(self) -> Union[ast.Name, ast.Constant]:
    comp = self.compare.comparators[0]
    # Silence mypy
    assert isinstance(comp, ast.Name) or isinstance(comp, ast.Constant)
    return comp

# -- Example Code --
#   df['Name'].apply(lambda s: 'G' in s)
# -- Corresponding AST --
# Module(
#     body=[
#         Expr(
#             value=Call(
#                 func=Attribute(
#                     value=Subscript(value=Name(id='df'), slice=Constant(value='Name', kind=None)),
#                     attr='apply'),
#                 args=[
#                     Lambda(
#                         args=arguments(posonlyargs=[],
#                             args=[arg(arg='s', annotation=None, type_comment=None)],
#                             vararg=None,
#                             kwonlyargs=[],
#                             kw_defaults=[],
#                             kwarg=None,
#                             defaults=[]),
#                         body=Compare(left=Constant(value='G', kind=None), ops=[In], comparators=[Name(id='s')]))],
#                 keywords=[]))],
#     type_ignores=[])
def has_substring_search_apply(series_call: SeriesCall) -> Optional[HasSubstrSearchApply]:
  func = series_call.attr_call.get_func()
  if func == "apply":
    apply_Call = series_call.attr_call.call.get_obj()
    assert isinstance(apply_Call, ast.Call)
    args = apply_Call.args
    if len(args) != 1:
      return None
    lambda_ = args[0]
    if not isinstance(lambda_, ast.Lambda):
      return None
    arg0_str = lambda_has_only_one_arg(lambda_)
    if arg0_str is None:
      return None
    compare = lambda_.body
    if not isinstance(compare, ast.Compare):
      return None
    # TODO: Is this restrictive?
    if (not isinstance(compare.left, ast.Constant) and
        not isinstance(compare.left, ast.Name)):
      return None
    comparators = compare.comparators
    if len(comparators) != 1:
      return None
    if (not isinstance(comparators[0], ast.Name) or
        comparators[0].id != arg0_str):
      return None
    if (len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.In)):
      return None

    return HasSubstrSearchApply(series_call=series_call, compare=compare, dont_use_the_constructor_directly=True)


  return None

class IsInplaceUpdate:
  # Ironically, I thank this article: https://towardsdatascience.com/why-you-should-probably-never-use-pandas-inplace-true-9f9f211849e4
  FUNCS_THAT_SUPPORT_INPLACE = {"fillna", "replace", "rename", "dropna", "sort_values", "query", "drop_duplicates", "sort_index"}
  def __init__(self, assign: ast.Assign, series_call: SeriesCall, dont_use_the_constructor_directly: bool) -> None:
    self.assign = assign
    self.series_call = series_call
    assert _a(dont_use_the_constructor_directly)

# Example:
#   raw_x["Age"]=raw_x["Age"].fillna(raw_x["Age"].mean())
# AST:
# Assign(
#     targets=[Subscript(value=Name(id='raw_x'), slice=Constant(value='Age', kind=None))],
#     value=Call(
#         func=Attribute(
#             value=Subscript(value=Name(id='raw_x'), slice=Constant(value='Age', kind=None)),
#             attr='fillna'),
#         args=[
#             Call(
#                 func=Attribute(
#                     value=Subscript(value=Name(id='raw_x'), slice=Constant(value='Age', kind=None)),
#                     attr='mean'),
#                 args=[],
#                 keywords=[])],
#         keywords=[]),
#     type_comment=None)],
# TODO - IMPORTANT: We have to check that it's not followed by anything.
# - df.fillna(0) is ok
# - df.fillna(0)['x'] is not
def is_inplace_update(assign: ast.Assign, series_call: SeriesCall) -> Optional[IsInplaceUpdate]:
  assert assign == series_call.attr_call.call.get_encloser()
  assert isinstance(assign.value, ast.Call)

  attr_call = series_call.attr_call
  func = attr_call.get_func()
  rhs_df = series_call.get_sub().get_df()
  rhs_col = series_call.get_sub().get_Series()
  if func not in IsInplaceUpdate.FUNCS_THAT_SUPPORT_INPLACE:
    return None

  if len(assign.targets) != 1:
    return None
  
  lhs_sub = assign.targets[0]
  if not isinstance(lhs_sub, ast.Subscript):
    return None

  lhs_compat_sub = is_compatible_sub(lhs_sub)
  if lhs_compat_sub is None:
    return None
  lhs_df = lhs_compat_sub.get_df()
  lhs_col = lhs_compat_sub.get_Series()

  if lhs_df != rhs_df or lhs_col != rhs_col:
    return None

  # Check that the call doesn't already have an `inplace` argument.
  ast_call = attr_call.call.get_obj()
  # NOTE: The inplace argument is in the keywords because it's named.
  for kw in ast_call.keywords:
    assert isinstance(kw, ast.keyword)
    if kw.arg == "inplace":
      return None
    
  return IsInplaceUpdate(assign=assign, series_call=series_call, dont_use_the_constructor_directly=True)

# TODO: Change that to save a SeriesCall inside inside
class HasToListConcatToSeries:
  def __init__(self, enclosed_call: OptEnclosed[ast.Call],
               left_ser_call: SeriesCall, right_ser_call: SeriesCall,
               dont_use_the_constructor_directly: bool) -> None:
    self.enclosed_call = enclosed_call
    self.left_ser_call = left_ser_call
    self.right_ser_call = right_ser_call
    assert _a(dont_use_the_constructor_directly)

def has_tolist_concat_toSeries(call_name: CallOnName) -> Optional[HasToListConcatToSeries]:
  enclosed_call: OptEnclosed[ast.Call] = call_name.attr_call.call
  if (call_name.get_name() != "pd" or
      call_name.attr_call.get_func() != "Series"):
    return None
  args = enclosed_call.get_obj().args
  if len(args) != 1:
    return None
  arg0 = args[0]
  if not is_instance_chain(arg0, [["left"], ["right"]], [ast.BinOp, [[ast.Call], [ast.Call]]]):
    return None
  # Silence mypy
  assert isinstance(arg0, ast.BinOp)
  left_ser_call = is_Series_call_from_encl_call(get_non_enclosed(arg0.left))
  right_ser_call = is_Series_call_from_encl_call(get_non_enclosed(arg0.right))
  if left_ser_call is None or right_ser_call is None:
    return None
  if left_ser_call.attr_call.get_func() != 'tolist' or right_ser_call.attr_call.get_func() != 'tolist':
    return None
  return HasToListConcatToSeries(enclosed_call=enclosed_call, left_ser_call=left_ser_call,
                                  right_ser_call=right_ser_call,
                                  dont_use_the_constructor_directly=True)

def series_call_patts(series_call: SeriesCall) -> Optional[Union[HasSubstrSearchApply, IsInplaceUpdate]]:
  substr_apply: Optional[HasSubstrSearchApply] = has_substring_search_apply(series_call)
  if substr_apply is not None:
    return substr_apply

  if (series_call.attr_call.call.is_enclosed_attr()):
    assign = series_call.attr_call.call.get_encloser()
    if isinstance(assign, ast.Assign):
      # Assert that the Call is in the .value
      assert series_call.attr_call.call.get_access_attr() == "value"
      assert isinstance(series_call.attr_call.call.get_obj(), ast.Call)
      inplace_update = is_inplace_update(assign, series_call)
      if inplace_update is not None:
        return inplace_update

  return None

class ApplyOrMap:
  def __init__(self, call_name: CallOnName) -> None:
    self.call_name = call_name
    assert call_name.attr_call.get_func() in {"apply", "map"}
  
  def get_func_to_call(self) -> str:
    call = self.call_name.attr_call.call.get_obj()
    assert len(call.args) >= 1
    arg0 = call.args[0]
    assert isinstance(arg0, ast.Name)
    return arg0.id

class ApplyVectorizedLambda:
  def __init__(self, apply_call: AttrCall,
               lam: ast.Lambda, arg0_name: str,
               called_on_name: ast.Name) -> None:
    self.apply_call = apply_call
    self.lam = lam
    self.arg0_name = arg0_name
    self.called_on_name = called_on_name

# Should have only one argument. Return it.
def lambda_has_only_one_arg(l: ast.Lambda) -> Optional[str]:
  args = l.args
  assert isinstance(args, ast.arguments)
  # Should have only one argument for either axis=1 or 0
  assert len(args.args) == 1
  # All these should be empty for simplicity.
  should_be_empty = \
    ["posonlyargs", "kwonlyargs", "kw_defaults", "defaults"]
  if not all([len(getattr(args, attr)) == 0 for attr in should_be_empty]):
    return None
  if args.vararg is not None or args.kwarg is not None:
    return None
  
  return args.args[0].arg

def vec__test_side(hs, arg_name):
  accepted_insts = {ast.Name, ast.Constant, ast.Subscript, ast.Attribute}
  if type(hs) not in accepted_insts:
    return False

  if isinstance(hs, ast.Subscript):
    compat_sub = is_compatible_sub(hs)
    if compat_sub is None:
      return False
    row = compat_sub.get_df()
    if row != arg_name:
      return False

  # Accessing column as df.Name instead of df['Name']
  if isinstance(hs, ast.Attribute):
    if not isinstance(hs.value, ast.Name):
      return False
    if hs.value.id != arg_name:
      return False
  return True

def vec__cond_component(c, arg_name):
  if isinstance(c, ast.BoolOp):
    assert len(c.values) >= 2
    for val in c.values:
      proceed = vec__cond_component(val, arg_name)
      if proceed == False:
        return False
    if not isinstance(c.op, ast.And) and not isinstance(c.op, ast.Or):
      return False
  elif isinstance(c, ast.Compare):
    if len(c.comparators) != 1:
      return False
    comp = c.comparators[0]
    
    if not vec__test_side(comp, arg_name) or not vec__test_side(c.left, arg_name):
      return False

    if len(c.ops) != 1:
      return False
    # NOTE: `in` is vectorized with isin()
    vect_ops = {ast.Eq, ast.NotEq, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.In}
    if type(c.ops[0]) not in vect_ops:
      return False
  else:
    return False
  return True
# --- End of Function ---

# TODO: Remove the func_ast arg. It's for debugging purposes.
def vec__is_if_chain(if_: ast.If, func_ast, arg_name: str):
  # The body of the `if` must be a return.
  # The body of `orelse` should be either `If` or return.
  # Aaaaand, loop.
  then = if_.body
  if len(then) != 1:
    return False
  then_ret = then[0]
  # TODO: We need more tests on the return value.
  if not isinstance(then_ret, ast.Return):
    return False
  if len(if_.orelse) != 1:
    return False
  orelse = if_.orelse[0]
  check_condition = False
  if isinstance(orelse, ast.Return):
    check_condition = True
  elif isinstance(orelse, ast.If):
    check_condition = vec__is_if_chain(orelse, func_ast, arg_name)

  if check_condition:
    # Initially, I was checking that the only ast.Name that appears in the
    # condition is `arg_name`. Unfortunately, this is not enough. For example:
    # def cab_to_deck(cab):
    #   if type(cab) is float:
    #       return 'N'
    #   else:
    #       return cab[0]
    #
    # How do you vectorize the `is` operator?

    # Check that in the condition, the only name that appears is
    # the arg_name.
    # NOTE: It is intentionally put after the recursive call so that
    # the latter can stop the checks early.

    check_ = vec__cond_component(if_.test, arg_name)
    return check_
  else:
    return False

# I think that the same ideas can be applied to map().
# TODO: It may be required to know if axis=1 to deduce whether
# it can be vectorized (e.g., if we axis is not 1, then we might want
# to allow subscripts and attributes)
def can_be_vectorized_func(func_ast: ast.FunctionDef) -> bool:
  false_res = None
  body = func_ast.body
  # Body should have a single If
  if len(body) != 1:
    return false_res
  if_ = body[0]
  if not isinstance(if_, ast.If):
    return False
  if len(func_ast.args.args) != 1:
    return False
  row_name = func_ast.args.args[0].arg
  if not vec__is_if_chain(body[0], func_ast, row_name):
    return False
  return True

def can_be_vectorized_lambda(lam: ast.Lambda, attr_call: AttrCall) -> Optional[ApplyVectorizedLambda]:
  called_on_name = attr_call.get_called_on()
  if not isinstance(called_on_name, ast.Name):
    return None
  
  arg_name = lambda_has_only_one_arg(lam)
  if arg_name is None:
    return None
  
  # Here, we'll keep it simpler than the FunctionDef version.
  # The expression should be an IfExp. The body and the else
  # can be one of:
  # - CompatSub
  # - Name
  # - Constant
  # This means we don't recurse (e.g., we cannot have an IfExp
  # that has an IfExp for body)
  #
  # The condition follows the same rules as the FunctionDef

  ifexp = lam.body
  if not isinstance(ifexp, ast.IfExp):
    return None
  
  accept_types = {ast.Constant, ast.Name}
  exprs = [ifexp.body, ifexp.orelse]
  for e in exprs:
    if (type(e) not in accept_types and
        not is_compatible_sub_unkn(e)):
      return None
  
  is_valid_cond = vec__cond_component(ifexp.test, arg_name)
  if not is_valid_cond:
    return None
  
  return ApplyVectorizedLambda(apply_call=attr_call,
                               lam=lam,
                               arg0_name=arg_name,
                               called_on_name=called_on_name)

def is_apply_or_map(attr_call: AttrCall) -> Optional[Union[ApplyOrMap, ApplyVectorizedLambda]]:
  func = attr_call.get_func()
  if func not in {"map", "apply"}:
    return None
  # TODO - IMPORTANT: This is a hidden requirement.
  # We should make that clearer in the structure of the code.
  # It's not clear why having an ApplyOrMap requires that
  # they're called on Name.
  call_name = is_call_on_name(attr_call)
  if call_name is None:
    return None
  call = attr_call.call.get_obj()
  if not (len(call.args) >= 1):
    return None
  return ApplyOrMap(call_name=call_name)



# NOTE: Only apply() is a DataFrame call. map() is for Series
class RemoveAxis1:
  # A little stupid because `apply_call` and `call_name` share
  # `attr_call`
  def __init__(self, apply_call: ApplyOrMap) -> None:
    attr_call = apply_call.call_name.attr_call
    assert attr_call.get_func() == "apply"
    self.apply_call = apply_call

class RemoveAxis1Lambda:
  def __init__(self, call_name: CallOnName, lam: ast.Lambda, 
               the_one_series: str) -> None:
    self.call_name = call_name
    self.lam = lam
    self.the_one_series = the_one_series

def can_remove_axis_1(tree: ast.AST, arg0_name: str) -> Tuple[bool, str]:
  # Check that all the times the row_name appears, it is in a subscript
  # and check that all these subscripts the same Series.
  # NOTE: If we have other subscripts, on different names, then
  # we don't care about those.
  
  only_one_series = True
  # Be careful if you try to use Set[CompatSub]. These are objects
  # and Python may miscount two "equal" CompatSub's as the same.
  seen_subs: Dict[str, CompatSub] = dict()

  # Parents of subs and the attribute to access the sub.
  for n in ast.walk(tree):
    if len(seen_subs.keys()) > 1:
      only_one_series = False
      break
    names = search_enclosed(n, ast.Name)
    for enclosed_name in names:
      name = enclosed_name.get_obj()
      if len(seen_subs.keys()) > 1:
        only_one_series = False
        break
      assert isinstance(name, ast.Name)
      if name.id == arg0_name:
        sub = enclosed_name.get_encloser()
        if not isinstance(sub, ast.Subscript):
          only_one_series = False
          break
        compat_sub = is_compatible_sub(sub)
        if compat_sub is None:
          only_one_series = False
          break
        seen_subs[compat_sub.get_Series()] = compat_sub
      if only_one_series == False:
        break
# -- END OF LOOP --

  # TODO: Can we handle no sub? My samples show that it doesn't
  # exist.
  if len(seen_subs.keys()) != 1:
    return False, ""
  if not only_one_series:
    return False, ""
  the_one_compat_sub = seen_subs[[k for k in seen_subs.keys()][0]]
  the_one_series = the_one_compat_sub.get_Series()
  return True, the_one_series

def is_remove_axis_1(apply_call: ApplyOrMap) -> Optional[Union[RemoveAxis1, RemoveAxis1Lambda]]:
  call_name = apply_call.call_name
  call = call_name.attr_call.call.get_obj()
  if call_name.attr_call.get_func() != "apply":
    return None
  keywords = call.keywords
  has_axis_1 = False
  for kw in keywords:
    if kw.arg == "axis":
      value = kw.value
      # IMPORTANT - TODO: Can also be axis = 'columns'. Same thing.
      if isinstance(value, ast.Constant) and value.value == 1:
        has_axis_1 = True
        break
      else:
        return None
  
  if not has_axis_1:
    return None
  
  if len(call.args) != 1:
    return None
  arg0 = call.args[0]
  if isinstance(arg0, ast.Name):
    return RemoveAxis1(apply_call=apply_call)
  elif isinstance(arg0, ast.Lambda):
    lam = arg0
    if len(lam.args.args) != 1:
      return None
    lam_arg0 = lam.args.args[0].arg
    can_remove, the_one_series = can_remove_axis_1(arg0.body, lam_arg0)
    if can_remove:
      return RemoveAxis1Lambda(call_name=call_name, 
                               lam=lam, the_one_series=the_one_series)
  return None

# TODO: Should we have a generic MultipleOptions type or sth ?
# The current solution looks a little bad. The generic solution would be for recognize_pattern()
# to return a list of overlapping patterns (this list can have 1 element).
# Then patt_match() would return a list of lists. Each of these lists will
# be a list of tuples. Each list of tuples denotes a list of overlapping patterns.
# Each tuple is a pattern along with its stmt indexes.
# However, this complicates the code a lot (I kind of tried it) and I am not sure it's
# worth it. I.e., I think that solution looked even more horrible.
class ApplyCallMaybeRemoveAxis1:
  class Kind(Enum):
    APPLY_CALL = 0,
    REMOVE_AXIS_1 = 1

  def __init__(self, kind: Kind, apply_call: Optional[ApplyOrMap], remove_axis_1: Optional[RemoveAxis1] = None) -> None:
    assert kind == self.Kind.APPLY_CALL or kind == self.Kind.REMOVE_AXIS_1
    self.kind = kind
    self.apply_call = apply_call
    self.remove_axis_1 = None
    if kind == self.Kind.REMOVE_AXIS_1:
      self.remove_axis_1 = remove_axis_1
  
  def get_kind(self) -> Kind:
    return self.kind
  
  def get_apply_call(self) -> ApplyOrMap:
    assert self.apply_call is not None
    return self.apply_call

  def get_remove_axis_1(self) -> RemoveAxis1:
    assert self.get_kind() == self.Kind.REMOVE_AXIS_1
    assert self.remove_axis_1 is not None
    return self.remove_axis_1

class SortHead:
  def __init__(self, head_: AttrCall, sort_values_: AttrCall) -> None:
    self.head = head_
    self.sort_values = sort_values_

  def __get_keyword_value(keywords: List[ast.keyword], search_for: str) -> ast.AST:
    arg_value = None
    for kw in keywords:
      if kw.arg == search_for:
        arg_value = kw.value
        break
    # Can be None
    return arg_value

  def get_sort_by(self) -> ast.AST:
    sort_values_raw_call = self.sort_values.call.get_obj()
    # NOTE: Can have up to 1.
    if len(sort_values_raw_call.args) == 1:
      return sort_values_raw_call.args[0]
    else:
      by_value = SortHead.__get_keyword_value(sort_values_raw_call.keywords, "by")
      return by_value
  
  def is_sort_ascending(self) -> bool:
    sort_values_raw_call = self.sort_values.call.get_obj()
    asc_value = SortHead.__get_keyword_value(sort_values_raw_call.keywords, "ascending")
    if asc_value is None:
      # Default
      return True
    else:
      assert isinstance(asc_value, ast.Constant)
      assert isinstance(asc_value.value, bool)
      return asc_value.value

  def get_head_n(self) -> ast.AST:
    head_raw_call = self.head.call.get_obj()
    if len(head_raw_call.args) == 1:
      return head_raw_call.args[0]
    elif len(head_raw_call.keywords) == 1:
      return head_raw_call.keywords[0].value
    else:
      # Default
      return ast.Constant(value=5)


# df.sort_values('x').head(4)
# -----------------------------
# Call(
#   func=Attribute(
#       value=Call(func=Attribute(value=Name(id='df'), attr='sort_values'),
#           args=[Constant(value='x', kind=None)],
#           keywords=[]),
#       attr='head'),
#   args=[Constant(value=4, kind=None)],
#   keywords=[])
def is_sort_head(attr_call: AttrCall) -> Optional[SortHead]:
  head_ = attr_call
  if head_.get_func() != "head":
    return None
  head_attr_ast = head_.get_attr_ast()
  encl_sort_values = head_attr_ast.value
  if not isinstance(encl_sort_values, ast.Call):
    return None
  sort_values_ = is_attr_call(get_enclosed_attr(head_attr_ast, "value"))
  if sort_values_ is None:
    return None
  if sort_values_.get_func() != "sort_values":
    return None
  head_raw_call = head_.call.get_obj()
  sort_values_raw_call = sort_values_.call.get_obj()
  # If we have more than one positional arguments, just bail. It's rare.
  # NOTE: If we have one positional argument, then it's necessarily the first
  # argument (i.e., `by`). In Python, we cannot have a positional argument follow
  # a keyword arg.
  if len(sort_values_raw_call.args) > 1:
    return None
  # These args are a pain to handle but they're rare, so just bail.
  sort_bail_args = {"axis", "na_position", "key", "ignore_index"}
  sort_keywords = sort_values_raw_call.keywords
  for kw in sort_keywords:
    if kw.arg in sort_bail_args:
      return None
    # The `ascending` argument must either not appear (we know the default is `True`)
    # or if it appears, it must be a compile-time constant, because we need to know
    # whether we'll use nlargest() or nsmallest().
    if kw.arg == "ascending":
      asc_const = kw.value
      if (not isinstance(asc_const, ast.Constant) or
          not isinstance(asc_const.value, bool)):
        return None
  
  return SortHead(head_, sort_values_)

class ReplaceRemoveList:
  def __init__(self, attr_call: AttrCall,
               to_replace: ast.Constant, replace_with: ast.Constant,
               inplace: bool = False) -> None:
    self.attr_call = attr_call
    self.to_replace = to_replace
    self.replace_with = replace_with
    self.inplace = inplace

# TODO: Check this regarding type-checking: https://github.com/baziotis/pandas-opt/issues/59
# We need it here for what comes before .replace()
def is_replace_remove_list(attr_call: AttrCall) -> Optional[ReplaceRemoveList]:
  if attr_call.get_func() != "replace":
    return None
  # Check that it has exactly two positional arguments
  # TODO: Support their keyword version
  call_raw = attr_call.call.get_obj()
  args = call_raw.args
  if len(args) != 2:
    return None
  # Check that the first one is a list with a single element
  # that is a constant string.
  arg0 = args[0]
  if not isinstance(arg0, ast.List):
    return None
  if len(arg0.elts) != 1:
    return None
  el0 = arg0.elts[0]
  if not isinstance(el0, ast.Constant) or not isinstance(el0.value, str):
    return None
  to_replace = el0

  # Check that the second arg is either a constant string or a list with
  # a single string.
  replace_with = None
  arg1 = args[1]
  if isinstance(arg1, ast.List):
    if len(arg1.elts) != 1:
      return None
    el0 = arg1.elts[0]
    if not isinstance(el0, ast.Constant) or not isinstance(el0.value, str):
      return None
    replace_with = el0
  elif isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
    replace_with = arg1
  else:
    return None

  # Check if the parent is an assignment and then check if we can
  # add inplace=True. This is optional.

  res = ReplaceRemoveList(attr_call, to_replace=to_replace, replace_with=replace_with)

  assign = attr_call.call.get_encloser()
  if not isinstance(assign, ast.Assign):
    return res
  if len(assign.targets) != 1:
    return res

  called_on = attr_call.get_called_on()  
  lhs = assign.targets[0]

  same_names = \
    (isinstance(lhs, ast.Name) and
     isinstance(called_on, ast.Name) and
     lhs.id == called_on.id)
  lhs_sub = is_compatible_sub_unkn(lhs)
  rhs_sub = is_compatible_sub_unkn(called_on)
  same_subs = \
    (lhs_sub is not None and
      rhs_sub is not None and
      lhs_sub.get_df() == rhs_sub.get_df() and
      lhs_sub.get_Series() == rhs_sub.get_Series())

  if same_names or same_subs:
    # NOTE: We have checked that the parent of the call is
    # an Assign. So, there's nothing after the .replace()
    # (i.e., a chained call) because that would be the parent
    # of the call.
    res.inplace = True

  return res

class StrInCol:
  def __init__(self, cmp_encl: OptEnclosed[ast.Compare], the_str: ast.Constant, the_sub: ast.Subscript) -> None:
    self.cmp_encl = cmp_encl
    self.the_str = the_str
    self.the_sub = the_sub

def is_str_in_col(cmp_encl: OptEnclosed[ast.Compare]) -> Optional[StrInCol]:
  cmp = cmp_encl.get_obj()
  lhs = cmp.left
  # The LHS should be a constant string. We need to know the value
  # of the string because if it contains trailing whitespace, then it will probably
  # be in .to_string() but not in the two .contains()
  if (not isinstance(lhs, ast.Constant) or 
      not isinstance(lhs.value, str)):
    return None
  if lhs.value.isspace() or lhs.value.strip() != lhs.value:
    return None
  # We should have the `in` operator
  if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.In):
    return None
  # The rhs should be df[col].to_string()
  if len(cmp.comparators) != 1:
    return None
  rhs = cmp.comparators[0]
  if not isinstance(rhs, ast.Call):
    return None
  rhs_attr_call = is_attr_call(get_non_enclosed(rhs))
  if rhs_attr_call is None:
    return None
  if rhs_attr_call.get_func() != "to_string":
    return None
  called_on = rhs_attr_call.get_called_on()
  if not isinstance(called_on, ast.Subscript):
    return None
  # Just to be sure, the .value should be Name.
  # The .slice can be either a Name or a Constant string.
  if not isinstance(called_on.value, ast.Name):
    return None
  
  # You'd think that we could relax this because but then
  # there's a problem of correctness with double execution
  # because of the preconditions.
  slice_ = called_on.slice
  valid_slice = \
    (isinstance(slice_, ast.Name) or
    (isinstance(slice_, ast.Constant) and
     isinstance(slice_.value, str)
    ))
  if not valid_slice:
    return None

  return StrInCol(cmp_encl=cmp_encl, the_str=lhs, the_sub=called_on)

class MultipleStrInCol:
  def __init__(self, str_in_cols: List[StrInCol]) -> None:
    self.str_in_cols = str_in_cols

class FuseIsIn:
  def __init__(self, binop_encl: OptEnclosed[ast.BinOp], the_call: ast.Call,
               left_name: ast.Name, right_name: ast.Name) -> None:
    self.binop_encl = binop_encl
    self.the_call = the_call
    self.left_name = left_name
    self.right_name = right_name

def is_fuse_isin(binop_encl: OptEnclosed[ast.BinOp]) -> Optional[FuseIsIn]:
  binop = binop_encl.get_obj()
  if not isinstance(binop.op, ast.BitOr):
    return None
  left_encl = get_enclosed_attr(binop, "left")
  right_encl = get_enclosed_attr(binop, "right")
  left_call = left_encl.get_obj()
  right_call = right_encl.get_obj()
  if (not isinstance(left_call, ast.Call) or
      not isinstance(right_call, ast.Call)):
      return None
  left = is_attr_call(left_encl)
  right = is_attr_call(right_encl)
  if left is None or right is None:
    return None
  if left.get_func() != "isin" or right.get_func() != "isin":
    return None
  # They should both have a single arg and it should be a Name.
  # TODO: We can have a hardcoded list.
  if len(left_call.args) != 1 or len(right_call.args) != 1:
    return None
  left_name = left_call.args[0]
  right_name = right_call.args[0]
  if not isinstance(left_name, ast.Name) or not isinstance(right_name, ast.Name):
    return None
  # This could work for CallOnName too...
  left_ser = is_Series_call(left)
  right_ser = is_Series_call(right)
  if left_ser is None or right_ser is None:
    return None
  if left_ser.get_sub().get_df() != right_ser.get_sub().get_df():
    return None
  if left_ser.get_sub().get_Series() != right_ser.get_sub().get_Series():
    return None
  # Any of the two
  the_call = left_call

  return FuseIsIn(binop_encl=binop_encl, the_call=the_call,
                  left_name=left_name, right_name=right_name)

def is_str_attr_on_sub(str_attr: ast.AST) -> Optional[CompatSub]:
  if (not isinstance(str_attr, ast.Attribute) or
      str_attr.attr != "str"):
    return None
  attrd_obj = str_attr.value
  compat_sub = is_compatible_sub_unkn(attrd_obj)
  if compat_sub is not None:
    return compat_sub
  return None

class StrSplitPython:
  def __init__(self, lhs_root_name: ast.Name, 
               lhs_obj: ast.Subscript,
               lhs_targets: List[ast.Constant],
               rhs_obj: CompatSub,
               whole_rhs_encl: OptEnclosed[ast.Call],
               split_args: ast.arguments, expand_true: bool) -> None:
    self.lhs_root_name = lhs_root_name
    self.lhs_obj = lhs_obj
    self.lhs_targets = lhs_targets
    self.rhs_obj = rhs_obj
    self.whole_rhs_encl = whole_rhs_encl
    self.split_args = split_args
    self.expand_true = expand_true

    if not expand_true:
      assert len(lhs_targets) == 1

# Match either:
#  1) <Name>[Constant] = <Name | Subscript>.str.split(<length 1 str>, [<max splits>])
#  2) <Name>[[<Constant>*]] = <Name | Subscript>.str.split(<length 1 str>, [<max splits>], expand=True)
# We can speed up 2) but not 1). But 1) may be matched with a succeeding statement and we
# may speed it up. We cannot do it here because we only see one statement. See patt_match(). 
def is_str_split_python(assgn: ast.Assign) -> Optional[StrSplitPython]:
  # TODO: We may need to cut it faster, without using our API
  # like AttrCall etc.
  rhs = get_enclosed_attr(assgn, "value")
  rhs_call = is_attr_call_unkn(rhs)
  if rhs_call is None:
    return None
  if rhs_call.get_func() != "split":
    return None
  rhs_call_raw = rhs_call.call.get_obj()
  # There are more combinations that we can support here
  # if you read the doc, but we'll constrain it to the common
  # case:
  # - At least one and up to two positional args. One is the
  #   string, the other is the max number of splits.
  # - The string should be of length 1 so that it's not considered
  #   a regex.
  # - Possibly, one keyword with expand=True
  if not (1 <= len(rhs_call_raw.args) <= 2):
    return None

  if len(rhs_call_raw.keywords) > 1:
    return None
  
  expand_true = False
  if len(rhs_call_raw.keywords) == 1:
    expand_arg = rhs_call_raw.keywords[0]
    if expand_arg.arg != "expand":
      return None
    if (not isinstance(expand_arg.value, ast.Constant) or
        expand_arg.value.value != True):
      return None
    expand_true = True
  
  split_by = rhs_call_raw.args[0]
  if (not isinstance(split_by, ast.Constant) or
      not isinstance(split_by.value, str) or
      len(split_by.value) != 1):
    return None
  
  if len(rhs_call_raw.args) > 1:
    max_splits = rhs_call_raw.args[1]
    if (not isinstance(max_splits, ast.Constant) or
        not isinstance(max_splits.value, int)):
      return None

  # NOTE: We know there's no chain after the .split() otherwise
  # it would be the rhs_call. Start checking children (i.e., the
  # left of .split())
  str_attr = rhs_call.get_called_on()
  rhs_obj = is_str_attr_on_sub(str_attr)
  if rhs_obj is None:
    return None

  # Start now checking the LHS. If we have expand=True, check:
  #   <Name>[[Constant, Constant, ...]]
  # else:
  #   <Name>[Constant]
  if len(assgn.targets) > 1:
    return None
  lhs = assgn.targets[0]
  if not isinstance(lhs, ast.Subscript):
    return None
  # The indexed obj should be a Name for simplicity.
  lhs_root_name = lhs.value
  if not isinstance(lhs_root_name, ast.Name):
    return None

  slice_ = lhs.slice
  target_series = []
  if expand_true:
    if not isinstance(slice_, ast.List):
      return None
    # For simplicity in correctness, we'll require all the elements
    # to be constant strings.
    for elt in slice_.elts:
      if (not isinstance(elt, ast.Constant) or
          not isinstance(elt.value, str)):
        return None
      target_series.append(elt)
  else:
    if not isinstance(slice_, ast.Constant):
      return None
    target_series.append(slice_)

  return StrSplitPython(lhs_root_name=lhs_root_name, 
                        lhs_obj=lhs,
                        lhs_targets=target_series,
                        rhs_obj=rhs_obj,
                        whole_rhs_encl=rhs,
                        split_args=rhs_call_raw.args,
                        expand_true=expand_true)

class StrAttrIndexed:
  def __init__(self, encl_sub: OptEnclosed[ast.Subscript], 
               series: CompatSub, index: int) -> None:
    self.encl_sub = encl_sub
    self.series = series
    self.index = index

# Match:
#   [<anything> =] <Name | Subscript>.str[<Constant.int>]
# We want it to be a top-level statement so that we know that nothing
# changes the object in between.
def is_str_attr_indexed(n: ast.AST) -> Optional[StrAttrIndexed]:
  encl_expr = None
  if isinstance(n, ast.Assign) or isinstance(n, ast.Expr):
    encl_expr = get_enclosed_attr(n, "value")
  else:
    return None
  assert encl_expr is not None
  sub = encl_expr.get_obj()
  if not isinstance(sub, ast.Subscript):
    return None
  # For simplicity, only Constant. We could allow Name easily.
  if (not isinstance(sub.slice, ast.Constant) or
      type(sub.slice.value) != int):
    return None
  index = sub.slice.value
  series = is_str_attr_on_sub(sub.value)
  if series is None:
    return None
  return StrAttrIndexed(encl_sub=encl_expr, series=series, index=index)

class SubToSubReplace:
  def __init__(self, lhs_sub: CompatSub, rhs_sub: CompatSub, 
               replace_arg: ast.Name) -> None:
    self.lhs_sub = lhs_sub
    self.rhs_sub = rhs_sub
    self.replace_arg = replace_arg

def is_sub_to_sub_replace(assign: ast.Assign) -> Optional[SubToSubReplace]:
  if len(assign.targets) != 1:
    return None
  lhs = assign.targets[0]
  rhs = get_enclosed_attr(assign, "value")
  rhs_call_raw = rhs.get_obj()
  if not isinstance(rhs_call_raw, ast.Call):
    return None
  rhs_ser_call = is_Series_call_from_encl_call(rhs)
  if rhs_ser_call is None:
    return None
  if rhs_ser_call.attr_call.get_func() != "replace":
    return None
  
  lhs_sub = is_compatible_sub_unkn(lhs)
  if lhs_sub is None:
    return None
  
  # Single positional argument that is a Name and no keywords.
  # TODO: Eventually, we should combine this with
  # ReplaceRemoveList
  if len(rhs_call_raw.args) != 1:
    return None
  if len(rhs_call_raw.keywords) != 0:
    return None
  arg0 = rhs_call_raw.args[0]
  if not isinstance(arg0, ast.Name):
    return None
  return SubToSubReplace(lhs_sub=lhs_sub, 
                         rhs_sub=rhs_ser_call.get_sub(), replace_arg=arg0)

class UniqueOnSub:
  def __init__(self, sub: CompatSub, expr_to_replace: OptEnclosed) -> None:
    self.sub = sub
    self.expr_to_replace = expr_to_replace

def is_unique_on_sub(n: ast.AST) -> Optional[UniqueOnSub]:
  if not isinstance(n, ast.Expr):
    return None
  unique_encl = get_enclosed_attr(n, "value")
  if not isinstance(unique_encl.get_obj(), ast.Call):
    return None
  uniq_ser_call = is_Series_call_from_encl_call(unique_encl)
  if uniq_ser_call is None:
    return None
  if uniq_ser_call.attr_call.get_func() != "unique":
    return None
  if len(uniq_ser_call.attr_call.call.get_obj().args) != 0:
    return None
  return UniqueOnSub(sub=uniq_ser_call.get_sub(),
                     expr_to_replace=unique_encl)

class FuseApply:
  def __init__(self, right_apply: AttrCall, left_apply: AttrCall) -> None:
    self.right_apply = right_apply
    self.left_apply = left_apply

# Pattern: <anything>.apply(Name|Lambda).apply(Name|Lambda)
def is_fuse_apply(attr_call: AttrCall) -> Optional[FuseApply]:
  def first_arg_is_name_or_lambda(attr_call: AttrCall) -> bool:
    raw_call = attr_call.call.get_obj()
    if len(raw_call.args) != 1:
      return False
    arg0 = raw_call.args[0]
    if isinstance(arg0, ast.Name):
      return True
    elif isinstance(arg0, ast.Lambda):
      # Make sure it has one argument
      args = arg0.args.args
      if len(args) != 1:
        return False
      # Check that there are no other args
      return True 

    return False

  # We will match the rightmost apply() because in the AST,
  # the left of X in a call chain is a child of X. So, having the
  # rightmost apply(), we can access everything we need (i.e.,
  # everything to its left)
  right_apply = attr_call
  if right_apply.get_func() != "apply":
    return None
  if not first_arg_is_name_or_lambda(right_apply):
    return None

  left_apply = right_apply.get_called_on()
  if not isinstance(left_apply, ast.Call):
    return None
  # We don't care to enclose it because if we match the pattern,
  # we'll be able to access it through the first rightmost apply().
  left_apply = get_non_enclosed(left_apply)
  left_apply = is_attr_call(left_apply)
  if left_apply is None:
    return None
  if left_apply.get_func() != "apply":
    return None
  if not first_arg_is_name_or_lambda(left_apply):
    return None
  return FuseApply(right_apply=right_apply, left_apply=left_apply)
   
  

# Dispatch all the ugliness of what pattern we have recognized here.
# Because we can return immediately, we don't have to have a ridiculously
# nested if-else chain.
#
# TODO: This will match the first pattern and return. Apply the recognizers
# over and over in the statement and return a list with all the
# patterns that match this stmt.
Single_Stmt_Patts = \
Union[
  HasSubstrSearchApply,
  IsInplaceUpdate,
  HasToListConcatToSeries,
  ApplyCallMaybeRemoveAxis1,
  RemoveAxis1Lambda,
  IsTrivialDFCall,
  IsTrivialDFAttr,
  TrivialName,
  TrivialCall,
  SortHead,
  ReplaceRemoveList,
  MultipleStrInCol,
  FuseIsIn,
  StrSplitPython,
  StrAttrIndexed,
  SubToSubReplace
]
def recognize_pattern(stmt: ast.stmt) ->  Optional[Single_Stmt_Patts]:
  # Start with the trivial patterns.
  # NOTE: Those should generally match top-level Expr's. Be careful
  # if you try to put them in the walk() below.
  if is_trivial_call(stmt):
    return TrivialCall()

  if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Name):
    return TrivialName()

  if is_trivial_df_attr(stmt):
    return IsTrivialDFAttr()

  if is_trivial_df_call(stmt):
    return IsTrivialDFCall()

  # Otherwise, proceed with the actual patterns

  ### Top-Level Patterns ###
  str_attr_indexed = is_str_attr_indexed(stmt)
  if str_attr_indexed is not None:
    return str_attr_indexed
  
  unique_on_sub = is_unique_on_sub(stmt)
  if unique_on_sub is not None:
    return unique_on_sub

  if isinstance(stmt, ast.Assign):
    str_split_python = is_str_split_python(stmt)
    if str_split_python is not None:
      return str_split_python
    
    sub_to_sub_replace = is_sub_to_sub_replace(stmt)
    if sub_to_sub_replace is not None:
      return sub_to_sub_replace

  str_in_cols: List[StrInCol] = []
  # TODO: We need to somehow stop this walk early. For example, if the node
  # is a Subscript, then it won't match any pattern.
  for n in ast.walk(stmt):
    for cmp_encl in search_enclosed(n, ast.Compare):
      str_in_col: StrInCol = is_str_in_col(cmp_encl)
      # Let's experiment with matching multiple patterns per
      # top-level statement. The StrInCol's should be independent
      # so we can continue. So, let's not stop at the first. If another
      # pattern is matched we'll miss them but that's ok for now.
      if str_in_col is not None:
        str_in_cols.append(str_in_col)

    for binop_encl in search_enclosed(n, ast.BinOp):
      fuse_isin: FuseIsIn = is_fuse_isin(binop_encl)
      if fuse_isin is not None:
        return fuse_isin

    cand_enclosed = search_enclosed(n, ast.Call)

    for enclosed_call in cand_enclosed:
      attr_call = is_attr_call(enclosed_call)
      # This has a big nesting depth so it'd make sense to use continue.
      # However, in this way, it's clear who needs e.g., the call to be an
      # AttrCall. Furthermore, if you want to add a pattern that doesn't need it,
      # it's clear that you have to add it outside the `if` whereas you might
      # forget that you the continue
      ## ATTR CALL IF ##
      if attr_call is not None:
        fuse_apply = is_fuse_apply(attr_call)
        if fuse_apply is not None:
          return fuse_apply
        # TODO: Should be able to factor these and the following
        # because this tests for an AttrCall.
        apply_call = is_apply_or_map(attr_call)
        if isinstance(apply_call, ApplyOrMap):
          # TODO: These are overlapping. And this is the first
          # instance that tells us that we should support giving
          # to the user multiple options.
          remove_axis_1 = is_remove_axis_1(apply_call)
          if remove_axis_1 is not None:
            if isinstance(remove_axis_1, RemoveAxis1Lambda):
              return remove_axis_1
            else:
              assert isinstance(remove_axis_1, RemoveAxis1)
              return ApplyCallMaybeRemoveAxis1(
                kind=ApplyCallMaybeRemoveAxis1.Kind.REMOVE_AXIS_1,
                apply_call=remove_axis_1.apply_call,
                remove_axis_1=remove_axis_1)
          # END OF IF #

          call = attr_call.call.get_obj()
          arg0 = call.args[0]
          # Return only if the function to call is a Name.
          # TODO: This needs immediate cleanup. It has become
          # very ugly. The whole ApplyCallMaybeRemoveAxis1 has
          # to leave.
          if isinstance(arg0, ast.Name):
            return ApplyCallMaybeRemoveAxis1(
              kind=ApplyCallMaybeRemoveAxis1.Kind.APPLY_CALL,
              apply_call=apply_call)
          elif isinstance(arg0, ast.Lambda):
            lam = arg0
            vec_lam = can_be_vectorized_lambda(lam, attr_call)
            if vec_lam is not None:
              return vec_lam
        elif isinstance(apply_call, ApplyVectorizedLambda):
          return apply_call

        call_name = is_call_on_name(attr_call)
        if call_name is not None:
          tolist_concat_toSeries = has_tolist_concat_toSeries(call_name)
          if tolist_concat_toSeries is not None:
            return tolist_concat_toSeries
        
        replace_remove_list = is_replace_remove_list(attr_call)
        if replace_remove_list is not None:
          return replace_remove_list

        series_call: Optional[SeriesCall] = is_Series_call(attr_call)
        if series_call is not None:
          ser_call_patt = series_call_patts(series_call)
          if ser_call_patt is not None:
            return ser_call_patt

        sort_head = is_sort_head(attr_call)
        if sort_head is not None:
          return sort_head
      ## END OF ATTR CALL IF ##

  if len(str_in_cols) != 0:
    return MultipleStrInCol(str_in_cols=str_in_cols)

  return None

class FusableStrSplit:
  def __init__(self, source_split: StrSplitPython,
               index: int, expr_to_replace: OptEnclosed[ast.Subscript]) -> None:
    self.source_split = source_split
    self.index = index
    self.expr_to_replace = expr_to_replace

class FusableReplaceUnique:
  def __init__(self, replace_on_sub: CompatSub, 
               replace_lhs_sub: CompatSub,
               replace_arg: ast.Name, expr_to_replace: OptEnclosed) -> None:
    self.replace_on_sub = replace_on_sub
    self.replace_lhs_sub = replace_lhs_sub
    self.replace_arg = replace_arg
    self.expr_to_replace = expr_to_replace

# TODO: Not all Single_Stmt_Patts can be returned. Some are used only to
# match multi-stmt patterns.
Available_Patterns = \
Union[
  Single_Stmt_Patts,
  FusableStrSplit,
  FusableReplaceUnique,
]
# Returns a list of matched patterns, along with a list of indexes (in the input
# list) of all the statements that take part in the pattern.
# It matches the biggest possible pattern. The patterns don't overlap.
def patt_match(body: List[ast.stmt]) -> List[Tuple[Available_Patterns, List[int]]]:
  res: List[Tuple[Available_Patterns, List[int]]] = []

  single_stmt_patts: List[Single_Stmt_Patts] = []
  for stmt_idx, stmt in enumerate(body):
    patt = recognize_pattern(stmt)
    single_stmt_patts.append(patt)
  assert len(single_stmt_patts) == len(body)

  stmt_idx = 0
  while stmt_idx < len(single_stmt_patts):
    patt = single_stmt_patts[stmt_idx]

    ##############################################
    ### DON'T continue IN THIS LOOP WITHOUT
    ### INCREMENTING stmt_idx
    ##############################################

    # NOTE: We don't preserve the program order from statements to patterns.
    # Meaning, imagine that we have only N statements, and one pattern for each.
    # In the result list, the pattern corresponding to statement 0 won't necessarily
    # appear before the pattern corresponding to statement 4. I found that it isn't worth
    # it trying to preserve this because the patterns can span multiple statements.

    if patt is None:
      pass
    if isinstance(patt, StrSplitPython):
      if patt.expand_true:
        res.append((patt, [stmt_idx]))
      else:
        split_patt = patt
        next_stmt_idx = stmt_idx + 1
        index_patt = single_stmt_patts[next_stmt_idx]
        if isinstance(index_patt, StrAttrIndexed):
          # Make sure they're operating on the same obj
          # Both should be CompatSub
          lhs_split = is_compatible_sub(split_patt.lhs_obj)
          assert lhs_split is not None
          rhs_index = index_patt.series
          if compat_sub_eq(lhs_split, rhs_index):
            new_patt = \
              FusableStrSplit(
                source_split=split_patt,
                index=index_patt.index,
                expr_to_replace=index_patt.encl_sub)
            res.append((new_patt, [stmt_idx, next_stmt_idx]))
            stmt_idx = next_stmt_idx
    elif isinstance(patt, SubToSubReplace):
      replace_patt = patt
      next_stmt_idx = stmt_idx + 1
      unique_patt = single_stmt_patts[next_stmt_idx]
      if (isinstance(unique_patt, UniqueOnSub) and
          compat_sub_eq(replace_patt.lhs_sub, unique_patt.sub)):
        new_patt = \
          FusableReplaceUnique(replace_on_sub=replace_patt.rhs_sub,
                               replace_lhs_sub=replace_patt.lhs_sub,
                               replace_arg=replace_patt.replace_arg,
                               expr_to_replace=unique_patt.expr_to_replace)
        res.append((new_patt, [stmt_idx, next_stmt_idx]))
        stmt_idx = next_stmt_idx
    else:
      res.append((patt, [stmt_idx]))

    stmt_idx = stmt_idx + 1
  # END OF WHILE LOOP
  
  return res

# I think this should be pure but I'm not sure...
def get_available_patterns():
  # These are all the patterns we can recognize
  # Pure Python magic!
  avail_patts = Available_Patterns.__args__
  # Remove None
  patterns = [p for p in avail_patts if p != type(None)]
  patt_names = [p.__name__ for p in patterns]
  return patterns, patt_names


############ UTILS ############

def only_math_ops_on_names(e: ast.expr, internal_names: Set[str], external_names: Set[str]) -> bool:
  if isinstance(e, ast.Name):
    if e.id not in internal_names:
      external_names.add(e.id)
    return True
  if isinstance(e, ast.BinOp):
    accept_ops = {ast.Add, ast.Sub, ast.Mult, ast.Div}
    if type(e.op) not in accept_ops:
      return False
    return (only_math_ops_on_names(e.left, internal_names, external_names) and 
            only_math_ops_on_names(e.right, internal_names, external_names))
  else:
    return False

def has_only_math(func_ast: ast.FunctionDef, args: List[str], subs: List[str], external_names: Set[str]) -> bool:
  assert len(args) > 0
  arg0_name = args[0]
  args_set = set(args)
  body = func_ast.body

  introduced_names: Set[str] = set()
  # All statements before the last should be Assign
  for stmt in body[:-1]:
    if not isinstance(stmt, ast.Assign):
      return False
    if len(stmt.targets) != 1:
      return False
    lhs = stmt.targets[0]
    # The LHS should be Name that is not in the arguments
    # (to avoid weird Python trickery)
    if not isinstance(lhs, ast.Name):
      return False
    introduced_names.add(lhs.id)
    # RHS should be a CompatSub whose Name is
    # the arg0 name
    rhs_sub = is_compatible_sub_unkn(stmt.value)
    if rhs_sub is None:
      return False
    if rhs_sub.get_df() != arg0_name:
      return False
    subs.append(rhs_sub.get_Series())
  # END OF LOOP #

  # The last stmt should be a Return, with only binops
  # on Names. The Names should be either an argument
  # or one of the introduced names.
  ret = body[-1]
  if not isinstance(ret, ast.Return):
    return False
  return only_math_ops_on_names(ret.value, args_set.union(introduced_names), external_names)
  
