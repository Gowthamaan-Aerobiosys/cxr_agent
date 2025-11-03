# Production Deployment Guide - CXR Agent MCP Server

## Overview

This guide covers deploying the CXR Agent MCP Server in a production environment with considerations for reliability, security, and performance.

## Deployment Architecture Options

### Option 1: Single Server Deployment

```
┌─────────────────────────────────────┐
│         Load Balancer (Nginx)       │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│      MCP Server (Python)            │
│  ┌──────────────────────────────┐   │
│  │  All Models (6-8GB GPU RAM)  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Option 2: Multi-Server Deployment

```
┌─────────────────────────────────────┐
│    Load Balancer + Router (Nginx)   │
└───┬─────────────┬──────────────┬────┘
    │             │              │
┌───▼────┐   ┌───▼────┐    ┌───▼────┐
│Server 1│   │Server 2│    │Server 3│
│Binary  │   │14-Class│    │  RAG   │
│Classify│   │ & Seg  │    │ Agent  │
└────────┘   └────────┘    └────────┘
```

### Option 3: Microservices (Recommended for Scale)

```
┌─────────────────────────────────────┐
│        API Gateway (Kong/AWS)       │
└───┬─────────────┬──────────────┬────┘
    │             │              │
┌───▼────────┐ ┌─▼──────────┐ ┌─▼────────┐
│Classification│ │Segmentation│ │   RAG    │
│  Service    │ │  Service   │ │ Service  │
│(MCP Server) │ │(MCP Server)│ │(MCP Srv) │
└─────────────┘ └────────────┘ └──────────┘
```

## Pre-Deployment Checklist

### ✅ Infrastructure

- [ ] GPU servers provisioned (NVIDIA T4/V100/A100 recommended)
- [ ] Storage for model weights (50GB+ recommended)
- [ ] Network bandwidth (10Gbps+ for high throughput)
- [ ] Monitoring tools setup (Prometheus, Grafana)
- [ ] Logging infrastructure (ELK stack or CloudWatch)

### ✅ Security

- [ ] API authentication implemented
- [ ] HTTPS/TLS certificates configured
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Input validation in place
- [ ] PHI handling compliance (HIPAA if applicable)

### ✅ Software

- [ ] Python 3.8+ installed
- [ ] CUDA drivers installed (if using GPU)
- [ ] Docker installed (for containerization)
- [ ] Dependencies installed from requirements.txt
- [ ] Model weights downloaded and verified

### ✅ Configuration

- [ ] Production config file created
- [ ] Environment variables set
- [ ] Secrets management configured
- [ ] Backup strategy defined

## Production Configuration

### Production Config Template

Create `config/production.json`:

```json
{
  "server_name": "cxr-agent-prod",
  "version": "1.0.0",

  "models": {
    "binary_classifier": {
      "enabled": true,
      "checkpoint_path": "/mnt/models/binary_classifier.pth",
      "model_type": "swin_transformer"
    },
    "multiclass_classifier": {
      "enabled": true,
      "checkpoint_path": "/mnt/models/fourteen_class_classifier",
      "num_classes": 14,
      "model_type": "densenet121"
    },
    "segmentation": {
      "enabled": true,
      "checkpoint_path": "/mnt/models/segmentation_model.pth"
    },
    "rag": {
      "enabled": true,
      "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
      "vector_db_path": "/mnt/data/chroma_db",
      "documents_path": "/mnt/data/medical_literature"
    }
  },

  "server_config": {
    "device": "cuda",
    "cache_models": true,
    "max_batch_size": 8,
    "num_workers": 8,
    "log_level": "INFO"
  },

  "performance": {
    "use_amp": true,
    "optimize_memory": true,
    "compile_models": false
  },

  "security": {
    "enable_authentication": true,
    "allowed_image_formats": ["jpg", "jpeg", "png", "dcm"],
    "max_image_size_mb": 50,
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100
    }
  },

  "monitoring": {
    "enable_metrics": true,
    "metrics_file": "/var/log/cxr-agent/metrics.json",
    "enable_timing": true
  }
}
```

### Environment Variables

Create `.env` file:

```bash
# Server Configuration
CXR_ENV=production
CXR_CONFIG_PATH=/etc/cxr-agent/production.json
CXR_LOG_LEVEL=INFO

# Model Paths
CXR_MODELS_PATH=/mnt/models
CXR_DATA_PATH=/mnt/data

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

# Security
CXR_API_KEY=your-secure-api-key-here
CXR_ALLOWED_IPS=10.0.0.0/8,192.168.0.0/16

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Database
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=8000
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/weights /app/logs /app/outputs

# Expose port (if using REST wrapper)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python3 -c "import torch; print(torch.cuda.is_available())"

# Run server
CMD ["python3", "mcp_server.py", "--config", "config/production.json"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  cxr-agent:
    build: .
    container_name: cxr-agent-server
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CXR_CONFIG_PATH=/app/config/production.json
    volumes:
      - ./weights:/app/weights:ro
      - ./logs:/app/logs
      - ./outputs:/app/outputs
      - ./config:/app/config:ro
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test:
        [
          "CMD",
          "python3",
          "-c",
          "import requests; requests.get('http://localhost:8000/health')",
        ]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

### Build and Deploy

```bash
# Build Docker image
docker build -t cxr-agent:1.0.0 .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f cxr-agent

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cxr-agent
  labels:
    app: cxr-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cxr-agent
  template:
    metadata:
      labels:
        app: cxr-agent
    spec:
      containers:
        - name: cxr-agent
          image: cxr-agent:1.0.0
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
              cpu: "4"
            requests:
              nvidia.com/gpu: 1
              memory: "8Gi"
              cpu: "2"
          ports:
            - containerPort: 8000
          env:
            - name: CXR_CONFIG_PATH
              value: "/app/config/production.json"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
          volumeMounts:
            - name: models
              mountPath: /app/weights
              readOnly: true
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: logs
              mountPath: /app/logs
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: cxr-models-pvc
        - name: config
          configMap:
            name: cxr-config
        - name: logs
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: cxr-agent-service
spec:
  selector:
    app: cxr-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Monitoring & Logging

### Prometheus Metrics

Add to your MCP server:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
inference_counter = Counter('cxr_inference_total', 'Total inferences', ['model'])
inference_duration = Histogram('cxr_inference_duration_seconds', 'Inference duration', ['model'])
model_loaded = Gauge('cxr_model_loaded', 'Model loaded status', ['model'])
gpu_memory = Gauge('cxr_gpu_memory_used_bytes', 'GPU memory used')

# Start metrics server
start_http_server(9090)
```

### Logging Configuration

```python
import logging
from logging.handlers import RotatingFileHandler

# Production logging setup
logger = logging.getLogger('cxr-agent')
logger.setLevel(logging.INFO)

# File handler with rotation
file_handler = RotatingFileHandler(
    '/var/log/cxr-agent/app.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=10
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)
```

## Performance Tuning

### GPU Optimization

```python
# Enable TF32 on Ampere GPUs
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Use channels last memory format
model = model.to(memory_format=torch.channels_last)
```

### Batch Processing

```python
async def batch_process_images(image_paths: List[str], batch_size: int = 8):
    """Process multiple images in batches"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            client.classify_binary(path) for path in batch
        ])
        results.extend(batch_results)
    return results
```

## Scaling Strategies

### Horizontal Scaling

- Deploy multiple MCP server instances
- Use load balancer (Nginx, HAProxy, AWS ALB)
- Session affinity for stateful operations

### Vertical Scaling

- Increase GPU memory (V100 → A100)
- More CPU cores for preprocessing
- Faster SSD for model loading

### Model Optimization

- Quantization (INT8, FP16)
- Model pruning
- Knowledge distillation
- TensorRT optimization

## Backup & Recovery

### Model Weights Backup

```bash
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backup/cxr-agent-models"
MODELS_DIR="/app/weights"
DATE=$(date +%Y%m%d)

# Create backup
tar -czf "$BACKUP_DIR/models-$DATE.tar.gz" "$MODELS_DIR"

# Keep only last 7 days
find "$BACKUP_DIR" -name "models-*.tar.gz" -mtime +7 -delete
```

### Database Backup (Vector DB)

```bash
#!/bin/bash
# backup_vectordb.sh

BACKUP_DIR="/backup/cxr-agent-db"
DB_DIR="/app/rag-pipeline/chroma_db"
DATE=$(date +%Y%m%d)

# Backup ChromaDB
tar -czf "$BACKUP_DIR/chromadb-$DATE.tar.gz" "$DB_DIR"

# Keep only last 30 days
find "$BACKUP_DIR" -name "chromadb-*.tar.gz" -mtime +30 -delete
```

## Security Hardening

### API Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.getenv("CXR_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("100/minute")
async def analyze(request: Request):
    # Process request
    pass
```

## Disaster Recovery

### Recovery Plan

1. **Model Weights Loss**

   - Restore from backup
   - Re-download from model repository
   - Verify checksums

2. **Server Failure**

   - Automatic failover to standby server
   - Load balancer redirects traffic
   - Monitor for recovery

3. **Database Corruption**
   - Restore from latest backup
   - Rebuild vector index if needed
   - Verify data integrity

## Maintenance

### Regular Tasks

**Daily:**

- Check logs for errors
- Monitor GPU memory usage
- Review inference metrics

**Weekly:**

- Backup model weights
- Update dependencies (if needed)
- Review and archive old logs

**Monthly:**

- Security patches
- Performance review
- Capacity planning

### Update Procedure

```bash
# 1. Backup current deployment
tar -czf backup-$(date +%Y%m%d).tar.gz /app

# 2. Pull new code
git pull origin main

# 3. Install new dependencies
pip install -r requirements.txt

# 4. Test in staging
python mcp_server.py --config config/staging.json

# 5. Deploy to production
docker-compose up -d --build

# 6. Monitor for issues
docker-compose logs -f
```

## Cost Optimization

### GPU Instance Selection

| Instance     | GPU  | RAM   | Price/hr | Use Case        |
| ------------ | ---- | ----- | -------- | --------------- |
| g4dn.xlarge  | T4   | 16GB  | $0.526   | Development     |
| p3.2xlarge   | V100 | 16GB  | $3.06    | Production      |
| p4d.24xlarge | A100 | 320GB | $32.77   | High throughput |

### Optimization Tips

1. **Autoscaling**: Scale down during off-hours
2. **Spot Instances**: Use for non-critical workloads
3. **Model Caching**: Reduce cold start times
4. **Batch Processing**: Amortize GPU setup costs
5. **Reserved Instances**: Commit for 1-3 years for discount

## Troubleshooting

### Common Production Issues

**Issue: High Memory Usage**

```bash
# Check memory
nvidia-smi

# Solution: Unload unused models
curl -X POST http://localhost:8000/unload_model -d '{"model_name": "segmentation"}'
```

**Issue: Slow Inference**

```bash
# Check GPU utilization
nvidia-smi dmon

# Solution: Enable AMP
# Set in config: "use_amp": true
```

**Issue: Connection Timeouts**

```bash
# Check server status
curl http://localhost:8000/health

# Solution: Increase timeout, check network
```

## Support Contacts

- **Technical Lead**: [email]
- **DevOps**: [email]
- **On-Call**: [phone]
- **Documentation**: [wiki-link]

## Compliance & Regulations

### HIPAA Compliance (if applicable)

- ✅ Encrypt data at rest
- ✅ Encrypt data in transit
- ✅ Access logging
- ✅ PHI de-identification
- ✅ Audit trails
- ✅ BAA with cloud providers

### GDPR Compliance

- ✅ Data minimization
- ✅ Right to erasure
- ✅ Data portability
- ✅ Consent management

---

**Remember**: Always test thoroughly in staging before production deployment!
