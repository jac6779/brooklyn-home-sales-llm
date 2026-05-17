FROM brooklynhomepricereg.azurecr.io/brooklyn-home-price-llm-base:v1

WORKDIR /app

COPY app ./app
COPY models ./models

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]