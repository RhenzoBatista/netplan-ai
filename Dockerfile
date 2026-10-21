FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY netplan ./netplan
RUN pip install --no-cache-dir .

COPY data/specs ./data/specs
RUN python -m netplan.cli init-db

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "netplan.api:app", "--host", "0.0.0.0", "--port", "8000"]
