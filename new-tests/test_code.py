import os
os.environ['DIAS_USE_AS_LIB'] = "True"
import dias.rewriter
import ast, astor


def test_fusable_str_split():
  with open(os.path.join('code-tests', 'fusable_str_split_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'fusable_str_split_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_fusable_replace_unique():
  with open(os.path.join('code-tests', 'fusable_replace_unique_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'fusable_replace_unique_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_str_in_col():
  with open(os.path.join('code-tests', 'str_in_col_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'str_in_col_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_sort_head__tricky():
  with open(os.path.join('code-tests', 'sort_head__tricky_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'sort_head__tricky_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_str_split_python():
  with open(os.path.join('code-tests', 'str_split_python_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'str_split_python_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_len_unique():
  with open(os.path.join('code-tests', 'len_unique_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'len_unique_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_search_with_apply():
  with open(os.path.join('code-tests', 'search_with_apply_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'search_with_apply_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_str_in_col__tricky():
  with open(os.path.join('code-tests', 'str_in_col__tricky_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'str_in_col__tricky_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_inplace_update():
  with open(os.path.join('code-tests', 'inplace_update_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'inplace_update_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_replace_remove_list_inplace():
  with open(os.path.join('code-tests', 'replace_remove_list_inplace_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'replace_remove_list_inplace_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_apply_vectorized_lambda():
  with open(os.path.join('code-tests', 'apply_vectorized_lambda_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'apply_vectorized_lambda_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_apply_vectorized_lambda_chained():
  with open(os.path.join('code-tests', 'apply_vectorized_lambda_chained_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'apply_vectorized_lambda_chained_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_fuse_apply():
  with open(os.path.join('code-tests', 'fuse_apply_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'fuse_apply_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_tolist_concat_toSeries():
  with open(os.path.join('code-tests', 'tolist_concat_toSeries_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'tolist_concat_toSeries_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_sort_head():
  with open(os.path.join('code-tests', 'sort_head_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'sort_head_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_replace_remove_list():
  with open(os.path.join('code-tests', 'replace_remove_list_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'replace_remove_list_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr

def test_search_with_apply__bigger_called_on():
  with open(os.path.join('code-tests', 'search_with_apply__bigger_called_on_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('code-tests', 'search_with_apply__bigger_called_on_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr
