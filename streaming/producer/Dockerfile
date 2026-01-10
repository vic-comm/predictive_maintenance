FROM python:3.9-slim

WORKDIR /app

RUN pip install -no-cache-dir pandas boto3

COPY data/partition/stream.csv /app/data/partition/stream.csv

COPY streaming/producer/simulate_sensors.py /app/streaming/producer/simulate_sensors.py

WORKDIR /app/streaming/producer

CMD ["python", "simulate_sensors.py"]