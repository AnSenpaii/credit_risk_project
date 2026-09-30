@echo off
call conda activate credit-risk
python scripts\make_smoke_data.py
python src\pipeline.py --smoke-test
