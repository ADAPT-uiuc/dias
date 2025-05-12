cd ..
python3 -m build
pip uninstall dias -y
pip install dist/dias*.whl
cd new-tests

python gen_static_tests.py > test_static.py
echo "*************** RUNNING STATIC TESTS ***************"
pytest

echo "*************** RUNNING DYNAMIC TESTS ***************"
cd dynamic-tests
./run_dynamic_tests.sh

echo "*************** DONE ***************"
