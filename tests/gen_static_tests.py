import os
import glob

SUFF = "_orig.py"
CODE_TEST_DIR = "static-tests"
orig_files = glob.glob(os.path.join(CODE_TEST_DIR, f"*{SUFF}"))

preamble = \
"""import os
os.environ['DIAS_USE_AS_LIB'] = "True"
import dias.rewriter
import ast, astor
"""

print(preamble)

for orig_file in orig_files:
  filename = orig_file.split('/')[-1]
  no_suff = filename[:-len(SUFF)]
  proper_name = no_suff
  test_func = \
f"""
def test_{proper_name}():
  with open(os.path.join('{CODE_TEST_DIR}', '{proper_name}_orig.py'), 'r') as fp:
    orig = fp.read()
  # END WITH #
  with open(os.path.join('{CODE_TEST_DIR}', '{proper_name}_rewr.py'), 'r') as fp:
    rewr_corr = fp.read()
  # END WITH #
  rewr_to_test, _ = dias.rewriter.rewrite_ast_from_source(orig)

  # Round-trip both through astor to ignore differences due to formatting.
  corr_tree = ast.parse(rewr_corr)
  to_test_tree = ast.parse(rewr_to_test)

  new_corr = astor.to_source(corr_tree)
  new_to_test = astor.to_source(to_test_tree)

  assert new_to_test == new_corr"""
  print(test_func)
### END FOR ###