@echo off
cd /d "%~dp0"
set PY=C:\Users\Admin\miniconda3\envs\ds_course_3.11\python.exe
"%PY%" scripts\make_smoke_data.py
"%PY%" src\pipeline.py --smoke-test
pause