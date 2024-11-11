from os import remove
from tabnanny import check
from typing import Dict, Final, List, Union, Optional, Tuple, Any, Set, TypeVar, Generic, Type
import ast
from enum import Enum
from dataclasses import dataclass

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
  
  def get_call_encl(self) -> OptEnclosed[ast.Call]:
    return self.call
  
  def get_call(self) -> ast.Call:
    return self.call.get_obj()
  
  def set_called_on(self, new_obj):
    attr_ast = self.get_attr_ast()
    attr_ast.value = new_obj

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
  if not isinstance(attr_call.get_called_on(), ast.Name):
    return False
  # Obviously, the caller should make sure that the type of the name is a DataFrame
  if attr_call.get_func() in IsTrivialDFCall.TRIVIAL_FUNCTIONS:
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

#   @{expr: df}.drop(@{expr: col}, axis=1, inplace=True)
#   -->
#   df = @{df}
#   df.pop(@{col})
#   rewr = df

# Preconditions:
# - isinstance(@{df}, pd.DataFrame)
# - isinstance(@{col}, str) or
#   (isinstance(@{col}, list) and len(@{col}) == 1 and isinstance(@{col}[0], str))
#   

# We don't do `axis=@{expr: axis}` and add a precondition that it should be one,
# because that's very rare and we'd be doing extra dynamic checks.
@dataclass
class DropToPop:
  call_encl: OptEnclosed[ast.Call]
  df: ast.expr
  col: ast.expr

  def __repr__(self):
    df_t = astor.dump_tree(self.df)
    col_t = astor.dump_tree(self.col)
    return f"DropToPop(df={df_t}, col={col_t})"

def drop_to_pop_helper(call_encl: OptEnclosed[ast.Call]) -> Optional[DropToPop]:
  call = call_encl.get_obj()
  attr_call = is_attr_call(call_encl)
  if not attr_call:
    return None
  if attr_call.get_func() != 'drop':
    return None
  if len(call.args) != 1:
    return False
  if len(call.keywords) != 2:
    return None
  
  kw_axis = call.keywords[0]
  kw_inplace = call.keywords[1]
  if kw_axis.arg == 'inplace':
    kw_inplace, kw_axis = kw_axis, kw_inplace
  
  succ = True
  match kw_axis:
    case ast.keyword(arg='axis', value=ast.Constant(value=1)):
      pass
    case _:
      succ = False
  # END MATCH #

  match kw_inplace:
    case ast.keyword(arg='inplace', value=ast.Constant(value=True)):
      pass
    case _:
      succ = False
  # END MATCH #

  if not succ:
    return None
  
  return DropToPop(call_encl=call_encl,
                   df=attr_call.get_called_on(), col=call.args[0])
  

# @{expr: df}.drop(@{expr: col}, axis=1)
#
def is_drop_to_pop(n: ast.AST) -> Optional[DropToPop]:
  calls = search_enclosed(n, ast.Call)

  for encl_call in calls:
    ret = drop_to_pop_helper(encl_call)
    if ret:
      return ret
  ### END FOR ###
  return None

# @{expr: df}[@{expr: pred}][@{Const(str): col}]
# -->
# @{df}[@{pred}, @{col}]

# TODO: Support lists of booleans.
# Preconditions:
# - `isinstance(@{df}, pd.DataFrame)`
# - `isinstance(@{pred}, pd.Series) and @{pred}.dtype == bool`
@dataclass
class SubSeq:
  df: ast.expr
  pred: ast.expr
  col: ast.Constant
  sub_encl: OptEnclosed[ast.Subscript]
  
  def __repr__(self):
    df_t = astor.dump_tree(self.df)
    pred_t = astor.dump_tree(self.pred)
    col_t = astor.dump_tree(self.col)
    return f"SubSeq(df={df_t}, pred={pred_t}, col={col_t})"

# df[df['a'] == 1]['col']
# ===
# value=Subscript(
#   value=Subscript(value=Name(id='df'),
#       slice=Compare(
#           left=Subscript(value=Name(id='df'), slice=Constant(value='a', kind=None)),
#           ops=[Eq],
#           comparators=[Constant(value=1, kind=None)])),
#   slice=Constant(value='col', kind=None))
def is_subseq_helper(sub_encl: OptEnclosed[ast.Subscript]) -> Optional[SubSeq]:
  sub = sub_encl.get_obj()
  if ((not isinstance(sub.slice, ast.Constant)) or 
      not isinstance(sub.slice.value, str)):
    return None
  if not isinstance(sub.value, ast.Subscript):
    return None
  in_sub = sub.value
  df = in_sub.value
  pred = in_sub.slice
  col = sub.slice
  return SubSeq(df=df, pred=pred, col=col, sub_encl=sub_encl)
  
def is_subseq(n: ast.AST) -> Optional[SubSeq]:
  subs = search_enclosed(n, ast.Subscript)

  for encl_sub in subs:
    __ret_ifnn(is_subseq_helper(encl_sub))
  ### END FOR ###
  return None


# @{expr: ser}.replace(@{expr: map_})
# -->
# ser = @{ser}
# default = {x: x for x in ser.drop_duplicates().values}
# default.update(@{map_})
# ser.map(default)

# Preconditions:
# - isinstance(@{ser}, pd.Series)
# - isinstance(@{map_}, map)
@dataclass
class ReplaceToMap:
  ser: ast.expr
  map_: ast.expr
  call_encl: OptEnclosed[ast.Call]
  
  def __repr__(self):
    ser_t = astor.dump_tree(self.ser)
    map_t = astor.dump_tree(self.map_)
    return f"ReplaceToMap(ser={ser_t}, map_={map_t})"


# df['Sex'].replace({'male': 0, 'female': 1})
# ===
# Call(
#   func=Attribute(
#       value=Subscript(value=Name(id='df'), slice=Constant(value='Sex', kind=None)),
#       attr='replace'),
#   args=[
#       Dict(
#           keys=[Constant(value='male', kind=None), Constant(value='female', kind=None)],
#           values=[Constant(value=0, kind=None), Constant(value=1, kind=None)])],
#   keywords=[])
def is_repl_to_map_helper(call_encl: OptEnclosed[ast.Call]) -> Optional[ReplaceToMap]:
  call = call_encl.get_obj()
  attr_call = is_attr_call(call_encl)
  __ret_ifn(attr_call)
  if attr_call.get_func() != 'replace':
    return None

  args = call.args
  kws = call.keywords
  if len(kws) != 0:
    return None
  if len(args) != 1:
    return None

  ser = attr_call.get_called_on()
  map_ = args[0]

  return ReplaceToMap(ser=ser, map_=map_, call_encl=call_encl)
  
def is_replace_to_map(n: ast.AST) -> Optional[ReplaceToMap]:
  calls = search_enclosed(n, ast.Call)

  for call_encl in calls:
    __ret_ifnn(is_repl_to_map_helper(call_encl))
  ### END FOR ###
  return None



# @{expr: ser}.unique()
# -->
# @{ser}.drop_duplicates().values

# Preconditions:
# - `isinstance(@{ser}, pd.Series)`
# - `@{ser}.dtype == 'O'`
@dataclass
class UniqueToDropDup:
  ser: ast.expr
  call_encl: OptEnclosed[ast.Call]
  
  def __repr__(self):
    ser_t = astor.dump_tree(self.ser)
    return f"UniqueToDropDup(ser={ser_t})"

def uniq_to_drop_dup_helper(call_encl: OptEnclosed[ast.Call]) -> Optional[UniqueToDropDup]:
  call = call_encl.get_obj()
  attr_call = is_attr_call(call_encl)
  __ret_ifn(attr_call)
  if attr_call.get_func() != 'unique':
    return None

  args = call.args
  kws = call.keywords
  if len(kws) != 0:
    return None
  if len(args) != 0:
    return None
  
  ser = attr_call.get_called_on()
  return UniqueToDropDup(ser=ser, call_encl=call_encl)

def is_uniq_to_drop_dup(n: ast.AST) -> Optional[UniqueToDropDup]:
  calls = search_enclosed(n, ast.Call)

  for call_encl in calls:
    __ret_ifnn(uniq_to_drop_dup_helper(call_encl))
  ### END FOR ###
  return None



# @{expr: df}.sort_values([by=]@{expr: by}, [ascending={expr: asc}]).head([[n=]@{expr: n}])
# -->
# By default: @{asc} = True, @{n} = 5
#
# if @{asc} == True:
#   @{df}.nsmallest(n=@{n}, columns=@{by})
# elif @{asc} == False:
#   @{df}.nlargest(n=@{n}, columns=@{by})

# Preconditions:
# - isinstance(@{df}, pd.DataFrame)
@dataclass
class SortHeadDF:
  df: ast.expr
  by: ast.expr
  asc: ast.expr
  n: ast.expr
  call_encl: OptEnclosed[ast.Call]

  def __repr__(self):
    df_t = astor.dump_tree(self.df)
    by_t = astor.dump_tree(self.by)
    asc_t = astor.dump_tree(self.asc)
    n_t = astor.dump_tree(self.n)
    return f"SortHeadDF(df={df_t}, by={by_t}, asc={asc_t}, n={n_t})"

# @{expr: ser}.sort_values([by=]@{expr: by}, [ascending={expr: asc}]).head([[n=]@{expr: n}])
# -->
# By default: @{asc} = True, @{n} = 5
#
# if @{asc} == True:
#   @{ser}.nsmallest(n=@{n})
# elif @{asc} == False:
#   @{ser}.nlargest(n=@{n})

# Preconditions:
# - isinstance(@{ser}, pd.Series)
@dataclass
class SortHeadSer:
  ser: ast.expr
  asc: ast.expr
  n: ast.expr
  call_encl: OptEnclosed[ast.Call]

  def __repr__(self):
    ser_t = astor.dump_tree(self.ser)
    asc_t = astor.dump_tree(self.asc)
    n_t = astor.dump_tree(self.n)
    return f"SortHeadSer(df={ser_t}, asc={asc_t}, n={n_t})"

def is_sort_head_df(sort_values: AttrCall) -> Optional[Dict]:
  # Valid cases:
  # 1) One kw, `by`
  # 1) One pos arg (`by`)
  # 1) One pos arg (`by`) and one kw (`ascending`)
  # 2) Two kws (`by` and `ascending`) (theoretically in any order, in practice
  #    it's almost always `by` followed by `ascending`)
  sv_args = sort_values.get_call().args
  sv_kws = sort_values.get_call().keywords
  by = None
  asc = None
  if len(sv_args) == 1 and len(sv_kws) == 0:
    by = sv_args[0]
    asc = ast.Constant(value=True)
  elif len(sv_kws) == 1 and sv_kws[0].arg == 'by' and len(sv_args) == 0:
    by = sv_kws[0].value
    asc = ast.Constant(value=True)
  elif len(sv_kws) == 1 and sv_kws[0].arg == 'ascending' and len(sv_args) == 1:
    by = sv_args[0]
    asc = sv_kws[0].value
  elif (len(sv_kws) == 2 and sv_kws[0].arg == 'by' and
        sv_kws[1].arg == 'ascending' and len(sv_args) == 0):
    by = sv_kws[0].value
    asc = sv_kws[1].value
  
  __ret_ifn(by)
  __ret_ifn(asc)
  
  df = sort_values.get_called_on()
  return {'df': df, 'by': by, 'asc': asc}

def is_sort_head_ser(sort_values: AttrCall) -> Optional[Dict]:
  # Valid cases:
  # 1) No pos or kw args at all
  # 2) One kw, `ascending`
  sv_args = sort_values.get_call().args
  sv_kws = sort_values.get_call().keywords
  asc = None
  if len(sv_args) == 0 and len(sv_kws) == 0:
    asc = ast.Constant(value=True)
  elif len(sv_kws) == 1 and sv_kws[0].arg == 'ascending' and len(sv_args) == 0:
    asc = sv_kws[0].value
  
  __ret_ifn(asc)
  
  ser = sort_values.get_called_on()
  return {'ser': ser, 'asc': asc}

def sort_head_helper(call_encl: OptEnclosed[ast.Call]) -> Optional[Union[SortHeadDF, SortHeadSer]]:
  call = call_encl.get_obj()
  head_ = is_attr_call(call_encl)
  __ret_ifn(head_)
  if head_.get_func() != 'head':
    return None
  
  called_on = head_.get_called_on()
  if not isinstance(called_on, ast.Call):
    return None

  sort_values = is_attr_call(get_non_enclosed(called_on))
  __ret_ifn(sort_values)
  if sort_values.get_func() != 'sort_values':
    return None

  head_args = head_.get_call().args
  head_kws = head_.get_call().keywords
  
  # .head() either has exactly one kw ('n') and no args, or exactly one arg and
  # no kws, or no args and no kws at all
  n = None
  if len(head_args) == 1 and len(head_kws) == 0:
    n = head_args[0]
  elif len(head_args) == 0 and len(head_kws) == 1 and head_kws[0].arg == 'n':
    n = head_kws[0].value
  elif len(head_args) == 0 and len(head_kws) == 0:
    n = ast.Constant(value=5)
  
  __ret_ifn(n)
  
  df_d = is_sort_head_df(sort_values)
  if df_d is not None:
    return SortHeadDF(df=df_d['df'], by=df_d['by'],
                      asc=df_d['asc'], n=n, call_encl=call_encl)
  
  ser_d = is_sort_head_ser(sort_values)
  __ret_ifn(ser_d)
  return SortHeadSer(ser=ser_d['ser'], asc=ser_d['asc'], 
                     n=n, call_encl=call_encl)

def is_sort_head(n: ast.AST) -> Optional[UniqueToDropDup]:
  calls = search_enclosed(n, ast.Call)

  for call_encl in calls:
    __ret_ifnn(sort_head_helper(call_encl))
  ### END FOR ###
  return None

@dataclass
class PivotToGroupBy:
  df: ast.expr
  index: ast.Constant
  values: ast.Constant
  aggs: ast.expr
  call_encl: OptEnclosed[ast.Call]

  def __repr__(self):
    df_t = astor.dump_tree(self.df)
    index_t = astor.dump_tree(self.index)
    values_t = astor.dump_tree(self.values)
    aggs_t = astor.dump_tree(self.aggs)
    return f"PivotToGroupBy(df={df_t}, index={index_t}, values={values_t}, aggs={aggs_t})"

def is_const_str(val) -> bool:
  if isinstance(val, ast.Constant) and isinstance(val.value, str):
    return True
  return False

def pivot_to_gby_helper(call_encl: Optional[ast.Call]) -> Optional[PivotToGroupBy]:
  piv_call = is_attr_call(call_encl)
  __ret_ifn(piv_call)
  pd_ = piv_call.get_called_on()
  if not isinstance(pd_, ast.Name) or pd_.id != "pd":
    return None
  if piv_call.get_func() != 'pivot_table':
    return None
  
  args = piv_call.get_call().args
  kws = piv_call.get_call().keywords
  kws2 = kws.copy()
  kw_d = {}
  for i, kw in enumerate(kws):
    cond1 = (kw.arg == 'index' or kw.arg == 'values') and is_const_str(kw.value)
    cond2 = kw.arg == 'aggfunc'
    if cond1 or cond2:
      kw_d[kw.arg] = kw.value
      kws2[i] = None
  ### END FOR ###
  kws2 = [kw for kw in kws2 if kw is not None]
  df = None
  case1 = len(kws2) == 0 and len(args) == 1
  case2 = len(kws2) == 1 and kws[0].arg == 'data'
  if case1:
    df = args[0]
  elif case2:
    df = kws2[0].value
  else:
    return None
  # END IF #
  assert df is not None
  return PivotToGroupBy(df=df, index=kw_d['index'], values=kw_d['values'], 
                        aggs=kw_d['aggfunc'], call_encl=call_encl)


def is_pivot_to_gby(n: ast.AST) -> Optional[PivotToGroupBy]:
  calls = search_enclosed(n, ast.Call)

  for call_encl in calls:
    __ret_ifnn(pivot_to_gby_helper(call_encl))
  ### END FOR ###
  return None

Available_Patterns = \
Union[
  IsTrivialDFCall,
  IsTrivialDFAttr,
  TrivialName,
  TrivialCall,

  DropToPop,
  SubSeq,
  ReplaceToMap,
  UniqueToDropDup,
  SortHeadDF,
  SortHeadSer,
  PivotToGroupBy
]

# Unlike the original Dias patt matcher, this is not hierarchical to reduce
# complexity.
def recognize_pattern(stmt: ast.stmt) ->  Optional[Available_Patterns]:
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
  
  funcs = [is_drop_to_pop, is_subseq, is_replace_to_map, is_uniq_to_drop_dup, 
           is_sort_head, is_pivot_to_gby]

  # TODO: We need to somehow stop this walk early. For example, if the node
  # is a Subscript, then it won't match any pattern.
  for n in ast.walk(stmt):
    for func in funcs:
      __ret_ifnn(func(n))
    ### END FOR ###
  ### END FOR ###

  return None

def patt_match(mod: ast.Module):
  res: List[Available_Patterns] = []
  for stmt in mod.body:
    patt = recognize_pattern(stmt)
    if patt:
      res.append(patt)
  ### END FOR ###
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