#!/bin/bash

export FLASK_APP=run.py
export HOST=local

flask run --host 0.0.0.0 --port 5000
