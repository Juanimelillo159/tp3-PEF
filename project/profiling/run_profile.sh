#!/bin/bash
python -m cProfile -o profile.out project/app.py
snakeviz profile.out
