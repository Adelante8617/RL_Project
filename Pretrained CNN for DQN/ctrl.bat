@echo off
:loop
    echo Starting training...
    python train.py
    goto loop