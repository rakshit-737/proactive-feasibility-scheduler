FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x run_all_experiments.sh

CMD ["bash", "run_all_experiments.sh"]
