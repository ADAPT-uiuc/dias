import os
os.environ["_IREWR_USE_AS_LIB"] = "True"

import patt_matcher
import rewriter
import ast, astor


code = "df.drop('col', axis=1)"
# t = ast.parse(code)
# patt = patt_matcher.patt_match(t)
print(rewriter.rewrite_ast_from_source(code))