# Salary Prediction Server - Docker Setup

## Build Docker Image

```bash
docker build -t salary-prediction-server .
```

## Run Docker Container

```bash
docker run -p 8080:8080 salary-prediction-server
```

## Test the Server

### Health Check
```bash
curl http://localhost:8080/ping
```

### Make Prediction (Windows PowerShell)
```powershell
$body = @{
    "hourly" = 1
    "python_yn" = 1
    "spark" = 0
    "aws" = 1
    "job_simp" = "data_scientist"
    "job_state" = "ca"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/predict_salary" -Method POST -Body $body -ContentType "application/json"
```

### Make Prediction (Linux/Mac)
```bash
curl -X POST http://localhost:8080/predict_salary \
  -H "Content-Type: application/json" \
  -d '{
    "hourly": 1,
    "python_yn": 1,
    "spark": 0,
    "aws": 1,
    "job_simp": "data_scientist",
    "job_state": "ca"
  }'
```

## Docker Commands

### Stop container
```bash
docker ps                          # List running containers
docker stop <container_id>         # Stop specific container
```

### Remove container
```bash
docker rm <container_id>
```

### Remove image
```bash
docker rmi salary-prediction-server
```

### Run with custom port
```bash
docker run -p 9000:8080 salary-prediction-server
```

### Run in detached mode (background)
```bash
docker run -d -p 8080:8080 salary-prediction-server
```

### View logs
```bash
docker logs <container_id>
docker logs -f <container_id>     # Follow logs
```

## Configuration

- **Port**: 8080 (can be changed in Dockerfile CMD)
- **Workers**: 2 (adjust based on CPU cores)
- **Threads**: 4 per worker
- **Timeout**: 120 seconds

## Production Notes

- Gunicorn is used for production deployment
- Health check endpoint: `/ping`
- Prediction endpoint: `/predict_salary`
- Image uses Python 3.11-slim for smaller size
