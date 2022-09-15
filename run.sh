#!/usr/bin/bash
cd src;
export FLASK_APP=api.py ; flask run --port 5001 --host 0.0.0.0 &
