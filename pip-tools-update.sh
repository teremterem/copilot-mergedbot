#!/bin/sh
. venv/bin/activate
pip-compile --upgrade requirements.in
pip-compile --upgrade dev-requirements.in
pip-sync requirements.txt dev-requirements.txt editable-requirements.txt
