@echo off
title Deep-Live-Cam (CUDA)
cd /d "%~dp0"
call wzf311\Scripts\activate.bat
python run.py --execution-provider cuda --lang zh
pause
