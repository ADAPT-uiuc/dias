#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
pip uninstall dias -y
pip install ${SCRIPT_DIR}/.. --no-cache-dir --no-binary=dias