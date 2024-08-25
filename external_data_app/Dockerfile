FROM python:3.9-slim

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r requirements.txt
RUN pytest tests/

CMD ["python", "prediction_request.py"]