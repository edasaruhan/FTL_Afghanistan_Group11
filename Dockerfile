FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install fastapi uvicorn joblib skit-learn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
