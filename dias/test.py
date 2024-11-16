import ast, astor

code = """
all_df.iloc[:train_size]['Family survival']
"""

t = ast.parse(code)
print(astor.dump_tree(t))