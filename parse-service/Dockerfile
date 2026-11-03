# Python 3.12 on a slim Debian base
FROM python:3.12-slim

# PYTHONUNBUFFERED=1: logs show up immediately in "docker compose logs"
# PYTHONDONTWRITEBYTECODE=1: no .pyc files in the container
# PIP_DISABLE_PIP_VERSION_CHECK=1: pip does not check for a newer version of itself
# PYTHONPATH=/app: modules are found in /app, so "from shared.client_resilience import ..." works
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# dependencies first (cached layer); --no-cache-dir keeps the image small
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# then the service code
COPY scraper_agent.py /app/scraper_agent.py
COPY shared /app/shared
COPY app.py /app/app.py

# unprivileged user; nologin = no interactive shell
RUN useradd --create-home --shell /usr/sbin/nologin app && \
    chown -R app:app /app
USER app

EXPOSE 8000

# 0.0.0.0 = listen on every interface of the container, so Odoo reaches it over the Docker network
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
