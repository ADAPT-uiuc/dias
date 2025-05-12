#!/bin/bash

for ipynb_file in *.ipynb; do
  [ -f "$ipynb_file" ] || break
  ipython $ipynb_file > tmp.err 2>&1
  # IPython sucks so there's no easy way to find out whether we got an error.
  # So, we save the output and we check if it contains the string "Traceback"
  if grep -q "Traceback" tmp.err; then
    echo "--- ${ipynb_file} FAILED"
  fi
done
rm tmp.err
