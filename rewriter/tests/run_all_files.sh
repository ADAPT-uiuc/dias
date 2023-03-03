#!/bin/bash

for ipynb_file in *.ipynb; do
    [ -f "$ipynb_file" ] || break
    correct_output_json="${ipynb_file%.ipynb}.json"
    if [ ! -f "$correct_output_json" ]; then
      echo "File ${correct_output_json} does not exist."
      exit 1
    fi
    # Changing directory just makes everything easier
    cd ..
    _IREWR_JSON_STATS="True" ipython tests/$ipynb_file 2> tests/tmp.json 1> /dev/null
    cd tests
    python tester.py tmp.json $correct_output_json 2> /dev/null
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
      echo "--- ${ipynb_file} FAILED"
    fi
done
# Remove tmp.json if it exists
[ -e tmp.json ] && rm tmp.json