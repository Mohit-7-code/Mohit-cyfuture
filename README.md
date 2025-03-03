# Mohit-cyfuture
STEP 1 -
python scripts/data_scrapper.py

STEP 2 -
python scripts/preprocessor.py

STEP 3 -
python scripts/train_model.py

STEP 4 -
python -m uvicorn app.api_fastapi:app
The API will be accessible at: http://127.0.0.1:8000.
Use the /predict endpoint to get predictions.

STEP 5 -
python -m streamlit run app/ui_streamlit.py
