$ErrorActionPreference = 'Stop'
python -m pip install -r requirements.txt
python scripts/download_data.py
python src/pipeline.py
python -m streamlit run app/app.py
