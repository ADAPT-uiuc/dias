import os
os.environ['DIAS_USE_AS_LIB'] = "True"
import dias.rewriter
import ast, astor


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
