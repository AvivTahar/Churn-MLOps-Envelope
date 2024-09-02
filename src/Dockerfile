FROM python:3.9

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r prediction_model/requirements.txt
RUN python -u -m pytest tests/

EXPOSE 8080

CMD ["uvicorn", "prediction_model.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]