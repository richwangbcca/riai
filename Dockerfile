FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

COPY . .
WORKDIR /app/riai

# ponytail: http.server is enough for one static file. Swap for nginx/caddy
# only if you ever need TLS termination, caching, or >1 concurrent reader.
# It runs in the background; if it dies the poller keeps going and the
# dashboard 502s until the next machine restart. Acceptable for a hobby box.
CMD python -m http.server 8080 --bind 0.0.0.0 --directory dashboard & \
    exec python poller.py --db /data/riai.db
