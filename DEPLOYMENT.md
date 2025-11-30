# Deployment Guide - SmolLM2 Finance Web UI

## Quick Start

### Option 1: Local Machine (Recommended for Development)

#### Prerequisites
- Python 3.10+
- pip/conda

#### Setup
```bash
# Clone repository
git clone https://github.com/anirudhpratap345/smolLM-finetuning.git
cd smolLM-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Run
```bash
# Gradio (recommended)
python app.py

# OR Streamlit
streamlit run streamlit_app.py
```

#### Access
- **Gradio**: http://localhost:7860
- **Streamlit**: http://localhost:8501

---

### Option 2: Docker (Recommended for Production)

#### Prerequisites
- Docker
- Docker Compose (optional)

#### Single Container
```bash
# Build
docker build -t smollm-finance .

# Run Gradio
docker run -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  smollm-finance python app.py

# Run Streamlit (different port)
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  smollm-finance streamlit run streamlit_app.py
```

#### Multiple Containers (Docker Compose)
```bash
# Start both Gradio and Streamlit
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

#### With GPU Support
```bash
# For nvidia-docker
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  smollm-finance python app.py
```

---

### Option 3: Cloud Deployment

#### Hugging Face Spaces (Free, Gradio Only)

**Steps:**
1. Create account on [huggingface.co](https://huggingface.co)
2. Go to [hf.co/spaces](https://huggingface.co/spaces)
3. Click "Create new Space"
4. Name: `smollm-finance`
5. Select Docker runtime
6. Upload these files:
   - `app.py`
   - `Dockerfile`
   - `requirements.txt`
   - `config/`
   - `scripts/`
7. Click "Create Space"

**Access:** `https://huggingface.co/spaces/YOUR_USERNAME/smollm-finance`

**Cost:** Free (with usage limits)

#### Streamlit Cloud (Free, Streamlit Only)

**Steps:**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Login with GitHub
4. Click "New app"
5. Select your repo and branch
6. Set main file: `streamlit_app.py`
7. Deploy

**Access:** `https://YOUR_USERNAME-smollm-finance.streamlit.app`

**Cost:** Free (with usage limits)

#### AWS EC2 (Paid, Full Control)

**Steps:**
1. Launch EC2 instance (Ubuntu 20.04)
2. SSH into instance
3. Install Docker and docker-compose
4. Clone repository
5. Run: `docker-compose up -d`
6. Configure security group (ports 7860, 8501)
7. Access via: `http://YOUR_IP:7860`

**Cost:** ~$5-10/month for small instance

#### Google Cloud Run (Serverless)

**Steps:**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/smollm-finance

# Deploy
gcloud run deploy smollm-finance \
  --image gcr.io/PROJECT_ID/smollm-finance \
  --platform managed \
  --port 7860 \
  --memory 2Gi
```

**Cost:** ~$0.40 per GB-hour

#### Azure Container Instances

**Steps:**
```bash
# Build and push
az acr build --registry YOUR_REGISTRY \
  --image smollm-finance:latest .

# Deploy
az container create \
  --resource-group YOUR_GROUP \
  --name smollm-finance \
  --image YOUR_REGISTRY.azurecr.io/smollm-finance \
  --ports 7860 \
  --cpu 2 --memory 2
```

**Cost:** ~$25/month for small instance

---

## Configuration for Different Environments

### Local (CPU)
```python
# config/training_config.py
num_train_epochs: 1
per_device_train_batch_size: 4
device: cpu
```

### Local (GPU)
```python
num_train_epochs: 3
per_device_train_batch_size: 16
device: cuda
use_fp16: True
```

### Production (High Traffic)
```python
num_train_epochs: 5
per_device_train_batch_size: 32
device: cuda
use_fp16: True
max_inference_batch: 10
cache_size: 100
```

---

## Environment Variables

```bash
# Device
export TORCH_DEVICE=cuda  # or cpu

# Model cache
export HF_HOME=/path/to/cache

# Disable telemetry
export GRADIO_ANALYTICS_ENABLED=False
export STREAMLIT_LOGGER_LEVEL=warning

# API keys (if needed)
export HUGGINGFACE_TOKEN=your_token
export WANDB_API_KEY=your_key
```

---

## Monitoring & Logs

### Local
```bash
# View real-time logs
python app.py 2>&1 | tee app.log

# Search logs
grep "ERROR" app.log
```

### Docker
```bash
# View logs
docker logs CONTAINER_ID

# Follow logs
docker logs -f CONTAINER_ID

# Save logs
docker logs CONTAINER_ID > app.log
```

### Cloud (Hugging Face Spaces)
- Logs visible in Space settings
- Error logs in web UI

### Cloud (Streamlit Cloud)
- Logs in "Logs" tab
- Notifications for crashes

---

## Performance Tuning

### Memory Optimization
```bash
# Reduce batch size
per_device_train_batch_size: 4

# Use CPU instead of GPU
TORCH_DEVICE: cpu

# Enable gradient checkpointing
gradient_checkpointing: True
```

### Speed Optimization
```bash
# Increase batch size (if GPU memory allows)
per_device_train_batch_size: 32

# Use fp16 for faster training
use_fp16: True

# Increase number of workers
num_proc: 4
```

### Cost Optimization
```bash
# Use smaller model
model_name: "TinyLlama"

# Shorter training
num_train_epochs: 1

# Smaller batch size
per_device_train_batch_size: 4
```

---

## Troubleshooting

### Port Already in Use
```bash
# Gradio
python app.py --server_port 7861

# Streamlit
streamlit run streamlit_app.py --server.port 8502

# Kill process using port 7860
lsof -ti:7860 | xargs kill -9
```

### CUDA Out of Memory
```bash
# Reduce batch size in config
per_device_train_batch_size: 4

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU
nvidia-smi
```

### Docker Build Fails
```bash
# Clear cache and rebuild
docker system prune -a
docker build --no-cache -t smollm-finance .
```

### Model Download Fails
```bash
# Set HF cache
export HF_HOME=/path/to/large/storage

# Manual download
huggingface-cli download HuggingFaceH4/SmolLM2-1.7B-Instruct
```

### Connection Issues
```bash
# Check if port is accessible
curl http://localhost:7860

# Check firewall
sudo ufw allow 7860

# Check docker network
docker network inspect smollm-network
```

---

## Backup & Recovery

### Backup Models
```bash
# Local
cp -r models/ models_backup/

# Docker
docker exec CONTAINER_ID tar -czf /app/models.tar.gz /app/models
docker cp CONTAINER_ID:/app/models.tar.gz ./models.tar.gz
```

### Restore Models
```bash
# Local
cp -r models_backup/ models/

# Docker
docker cp ./models.tar.gz CONTAINER_ID:/app/
docker exec CONTAINER_ID tar -xzf /app/models.tar.gz
```

---

## Scaling

### Multiple Instances (Load Balancing)
```bash
# Using Docker Compose
services:
  gradio_1:
    ...
    ports:
      - "7860:7860"
  
  gradio_2:
    ...
    ports:
      - "7861:7860"
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smollm-finance
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smollm-finance
  template:
    metadata:
      labels:
        app: smollm-finance
    spec:
      containers:
      - name: gradio
        image: YOUR_REGISTRY/smollm-finance:latest
        ports:
        - containerPort: 7860
```

---

## Support

- 📚 [WEB_UI_README.md](WEB_UI_README.md)
- 📖 [GRADIO_GUIDE.md](GRADIO_GUIDE.md)
- 🔗 [GitHub Issues](https://github.com/anirudhpratap345/smolLM-finetuning/issues)

---

## Cost Comparison

| Platform | CPU | GPU | Cost/Month | Setup Time |
|----------|-----|-----|-----------|-----------|
| **Local** | ✓ | ✓ | $0 | 5 min |
| **Docker** | ✓ | ✓ | $0 | 10 min |
| **HF Spaces** | ✓ | ✗ | Free | 5 min |
| **Streamlit Cloud** | ✓ | ✗ | Free | 5 min |
| **AWS EC2** | ✓ | ✓ | $5-20 | 15 min |
| **Google Cloud Run** | ✓ | ✗ | $10-50 | 20 min |
| **Azure** | ✓ | ✓ | $20-50 | 20 min |

---

**Happy Deploying! 🚀**
