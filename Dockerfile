FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .[full]

ENTRYPOINT ["python", "-m", "synthetic_data_platform.cli"]
CMD ["spark-profile", "--input", "data/raw/customer_events.csv"]
