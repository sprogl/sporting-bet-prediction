FROM python:3.10-slim

COPY ./app/main.py /code/app/main.py
COPY ./app/requirements.txt /code/app/requirements.txt
COPY ./app/core/utils.py /code/app/core/utils.py
COPY ./app/core/config.py /code/app/core/config.py

COPY ./archive/scaler.pkl /code/app/archive/scaler.pkl
COPY ./archive/model_svm.pkl /code/app/archive/model_svm.pkl
COPY ./archive/model_rfc.pkl /code/app/archive/model_rfc.pkl
COPY ./archive/elo_rates_enriched.csv /code/app/archive/elo_rates_enriched.csv

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r app/requirements.txt

EXPOSE 80

CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "80"]