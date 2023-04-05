import json
import sys
import difflib

sys.path.insert(1, '../../')
import dias.nb_utils as nb_utils

to_test = sys.argv[1]
correct = sys.argv[2]

def get_correct_json(file):
  assert file.endswith(".json")
  f = open(file, "r")
  json_loaded = json.load(f)
  f.close()
  return json_loaded

def get_to_test_json(file):
  assert file.endswith(".json")
  f = open(file, "r")
  text = f.read()
  cells = nb_utils.extract_json_cell_stats(text)
  comma_sep = ','.join(cells)
  json_text = '{"cells": [' + comma_sep + "]}"
  f.close()
  json_loaded = json.loads(json_text)
  return json_loaded


to_test_json = get_to_test_json(to_test)
correct_json = get_correct_json(correct)

there_was_error = False
assert len(to_test_json['cells']) == len(correct_json['cells'])
for cell1, cell2 in zip(to_test_json['cells'], correct_json['cells']):
  to_test_mod = cell1['modified']
  correct_mod = cell2['modified']
  if to_test_mod != correct_mod:
    there_was_error = True
    diff = difflib.ndiff(to_test_mod.splitlines(keepends=True), 
                         correct_mod.splitlines(keepends=True))
    print(''.join(diff), end="")

if there_was_error:
  sys.exit("Error")