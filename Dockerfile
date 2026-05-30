FROM python:3.9-slim

WORKDIR /app

COPY requirements_app.txt .

RUN pip install --no-cache-dir -r requirements_app.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
