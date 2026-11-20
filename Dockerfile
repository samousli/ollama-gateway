# The gateway only needs the standard library.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY gateway/gateway.py ./gateway.py
EXPOSE 11435
CMD ["python3", "-u", "/app/gateway.py"]
