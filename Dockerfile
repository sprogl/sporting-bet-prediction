FROM python:3.10-slim

COPY ./app /code/app
COPY ./archive/scaler.pkl /code/app/archive/scaler.pkl
COPY ./archive/model_svm.pkl /code/app/archive/model_svm.pkl
COPY ./archive/model_rfc.pkl /code/app/archive/model_rfc.pkl
COPY ./archive/elo_rates_enriched.csv /code/app/archive/elo_rates_enriched.csv

WORKDIR /code/app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]