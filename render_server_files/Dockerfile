# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file (minimal for Docker)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies including gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY prediction.py .
COPY salary_model.pkl .

# Expose port for the Flask app (Render will provide PORT env var)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run gunicorn server with dynamic port from Render
CMD gunicorn -b 0.0.0.0:${PORT:-8080} --workers 2 --threads 4 --timeout 120 prediction:app