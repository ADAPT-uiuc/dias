import os

def get_nb_source_cells(nb_as_json):
  source_cells = []
  for cell in nb_as_json["cells"]:
    assert "cell_type" in cell.keys()
    if cell["cell_type"] == "markdown":
        continue
    
    # It would be great if we could return the cell ID for every cell,
    # but it's not available. So, callers have to use the index
    # as a unique id.

    # Here `<=` means subset
    assert {"source"} <= cell.keys()
    # Apparently every line is a different list element even in JSON.
    # I think this will just create problems so I'm joining them.
    source_as_list_of_lines = cell["source"]
    # Apparently, some cells can be null
    if source_as_list_of_lines is None:
      continue
    source = "".join(source_as_list_of_lines)
    source_cells.append(source)
  
  return source_cells

def jsonify_string(s: str) -> str:
  return s.replace('\n', '\\n').replace('"', "\\\"")

def extract_json_cell_stats(s: str):
  # Yes, I know I can use regular expressions but the
  # code becomes totally unreadable.
  lines = s.split('\n')
  into = False
  buf = ""
  cells = []
  for l in lines:
    if "[IREWRITE JSON]" in l:
      assert into == False
      into = True
      continue
    if "[IREWRITE END JSON]" in l:
      assert into == True
      into = False
      cells.append(buf)
      buf = ""
      continue
    if into:
      buf = buf + l + "\n"
  
  return cells

def ns_to_ms(ns):
  return ns / 1_000_000

def write_to_file(filename, txt):
  f = open(filename, 'w')
  f.write(txt + "\n")
  f.close()

def get_dir_size(path):
  total = 0
  with os.scandir(path) as it:
    for entry in it:
      if entry.is_file():
        total += entry.stat().st_size
      elif entry.is_dir():
        total += get_dir_size(entry.path)
  return total

def has_tabular_data(root: str) -> bool:
  tabular_exts = (
    ".csv",
    ".zip",
      # We possibly need to unzip them.
    ".gzip",
      # Might be a parquet file.
    ".h5",
      # See: https://www.kaggle.com/datasets/jpmiller/simplified-dataset
    ".tsv",
      # See: https://www.kaggle.com/code/rahulpatel11315/read-data-from-tsv-file-using-pandas-dataframe
  )

  tabular_files_size = 0
  total_size = 0
  for path, dir, filelist in os.walk(root):
    for f in filelist:
      fp = os.path.join(path, f)
      file_size = os.path.getsize(fp)
      total_size = total_size + file_size
      if f.endswith(tabular_exts):
        tabular_files_size = tabular_files_size + file_size

  is_tabular = (tabular_files_size / total_size) > 0.4
  return is_tabular