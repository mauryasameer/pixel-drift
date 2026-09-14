FROM python:3.12-slim

# git is required at build time: tensorflow_examples installs from a git+https URL.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app/pixel-drift

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R app:app /home/app/pixel-drift

USER app

CMD ["python", "-m", "src.app"]
