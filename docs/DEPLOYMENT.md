# Deployment

End-to-end setup guide for the Airflow inference pipeline on EC2. Covers AWS infrastructure, Docker stack, and first-run verification.

## Architecture overview

```
AudioMoth clips
      │
      ▼
S3 (clips/incoming/)
      │  S3KeySensor
      ▼
EC2 t3.large (Airflow + Docker)
  ├── TinyCNN filter   (PyTorch CPU)
  ├── BirdNET FP16     (TFLite CPU)
  └── MLflow server    (SQLite backend, Docker service)
      │
      ├── DynamoDB  ← per-clip predictions + batch summary
      ├── S3 (clips/results/)  ← predictions.csv per run
      └── MLflow  ← pipeline_monitoring experiment (drift metrics)
```

---

## 1. AWS infrastructure

### S3

```bash
aws s3 mb s3://rainforest-audio-kwf --region us-east-1
```

The DAG reads from `clips/incoming/` and writes results to `clips/results/`. Processed clips are moved to `clips/processed/{dag_run_id}/` after each run so the sensor doesn't re-trigger.

### DynamoDB

```bash
aws dynamodb create-table \
  --table-name birdnet-predictions \
  --attribute-definitions \
      AttributeName=clip_id,AttributeType=S \
      AttributeName=processed_at,AttributeType=S \
  --key-schema \
      AttributeName=clip_id,KeyType=HASH \
      AttributeName=processed_at,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

Schema:
- `clip_id` (PK) — filename for clip records; `BATCH#{dag_run_id}` for the per-run summary record
- `processed_at` (SK) — ISO-8601 UTC timestamp

### IAM instance profile

Create a role with S3 read/write and DynamoDB read/write on the pipeline resources, then attach it to the EC2 instance. No AWS credentials are hardcoded anywhere — the Docker containers inherit credentials from the instance profile automatically.

Minimum permissions needed:
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` on `rainforest-audio-kwf`
- `s3:CopyObject` (for moving processed clips)
- `dynamodb:PutItem`, `dynamodb:BatchWriteItem` on `birdnet-predictions`

> **Why instance profile instead of access keys:** Hardcoding AWS credentials in `docker-compose.yml` or `.env` is a security risk and makes key rotation painful. The instance profile attaches permissions to the EC2 role — containers running on that host inherit credentials automatically via the metadata service, with no keys in any file.

### EC2 instance

- **Type:** t3.large (2 vCPU, 8 GB RAM)
- **AMI:** Ubuntu 22.04 LTS (`ami-0d7405d05f836d0d4`, us-east-1)
- **IAM instance profile:** `RainforestEC2Profile`
- **Security group inbound rules:**
  - Port 22 (SSH) — your IP
  - Port 8080 (Airflow UI) — your IP

---

## 2. EC2 first-time setup

SSH in and install Docker:

```bash
ssh -i ~/.ssh/rainforest-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu
newgrp docker
```

### Clone the repo

The repo is private. Generate a read-only deploy key on the EC2 instance:

```bash
ssh-keygen -t ed25519 -C "ec2-deploy" -f ~/.ssh/github_deploy_key -N ""
cat ~/.ssh/github_deploy_key.pub
```

Add the public key to GitHub: **repo → Settings → Deploy keys → Add deploy key** (read-only).

Configure SSH to use it:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile ~/.ssh/github_deploy_key
    StrictHostKeyChecking no
EOF
```

Clone:

```bash
sudo mkdir -p /opt/rainforest-audio-detection
sudo chown ubuntu:ubuntu /opt/rainforest-audio-detection
git clone git@github.com:QianFu520/rainforest-audio-detection.git /opt/rainforest-audio-detection
cd /opt/rainforest-audio-detection
```

> **Why a deploy key instead of a PAT:** GitHub PATs are tied to a user account and have broad scope. A read-only deploy key is scoped to one repo and lives only on this EC2 instance — safer and easier to rotate.

### Model weights

Model weights are not in the repo (too large). Upload them from your local machine:

```bash
# Run these locally
scp -i ~/.ssh/rainforest-key.pem \
    outputs/models/tinycnn_v3.pth \
    outputs/models/birdnet_v2.4_fp16.tflite \
    ubuntu@<EC2_PUBLIC_IP>:/opt/rainforest-audio-detection/outputs/models/
```

### `.env` file

```bash
cd /opt/rainforest-audio-detection

AIRFLOW_UID=$(id -u)
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

cat > .env <<EOF
AIRFLOW_UID=${AIRFLOW_UID}
AIRFLOW_FERNET_KEY=${FERNET_KEY}
AIRFLOW_SECRET_KEY=${SECRET_KEY}
AIRFLOW_ADMIN_PASSWORD=<choose a password>
BIRDNET_PIPELINE_BUCKET=rainforest-audio-kwf
AWS_DEFAULT_REGION=us-east-1
EOF
```

> **AIRFLOW_UID gotcha:** This must be set to the UID of the user on the EC2 host (`id -u` = 1000 for `ubuntu`), not your local machine's UID. If you copy a `.env` from your Mac (UID 501), Airflow containers will fail to write to mounted log and dag directories due to permission mismatches. Always generate `.env` directly on the EC2 instance.

---

## 3. Docker stack

### Build the image

```bash
cd /opt/rainforest-audio-detection
docker build -t rainforest-airflow:latest .
```

The image is based on `apache/airflow:2.9.3-python3.11` and adds:
- PyTorch 2.2.2 (CPU-only, via `--index-url https://download.pytorch.org/whl/cpu` — avoids pulling the 2 GB CUDA build)
- TensorFlow CPU 2.17.0 (last version with a separate `tensorflow-cpu` package — saves ~1 GB vs the full `tensorflow`)
- librosa, soundfile, `numpy==1.26.4` (pinned — PyTorch 2.2.2 was built before NumPy 2.0; if NumPy ≥ 2.0 is present, the `tinycnn_filter` task fails at the first model call with `RuntimeError: Numpy is not available`)
- Project source (`src/`, `assets/`, `config.py`)

Model weights are **not** baked into the image — they are mounted from `./outputs/models/` at runtime.

> **Airflow dependency isolation:** Do not install `apache-airflow` into the same Python environment as TensorFlow and MLflow. When we tried, `pip` silently downgraded `protobuf` from 5.28 to 4.25.3 (breaking TF) and `typing-extensions` from 4.14 to 4.12.2 (breaking MLflow/pydantic) — no errors at install time, only at import. Keep Airflow in its own environment (here: inside the Docker image); TF and MLflow stay in the local `ds` conda env for notebook work.

### Initialise Airflow (first time only)

```bash
mkdir -p logs
docker compose up airflow-init
```

Wait for `airflow-init` to exit 0, then:

> **docker-compose multiline command gotcha:** The `airflow-init` service uses a single-line `bash -c "..."` command in `docker-compose.yml`. The original version used a YAML `>` block scalar to split the command across lines — this looks clean in the file but YAML collapses the newlines into spaces, so `airflow users create \n --username admin \n --password ...` becomes one mangled string and `airflow-init` fails with a confusing unrecognised arguments error. Keep it on one line.

### Start the stack

```bash
docker compose up -d
```

Services: `postgres`, `airflow-webserver` (port 8080), `airflow-scheduler`.

Airflow UI: `http://<EC2_PUBLIC_IP>:8080` — login with username `admin` and the password set in `.env`.

> **After rebuilding the image**, use `--force-recreate` to ensure containers pick up the new image rather than reusing cached layers: `docker compose up -d --force-recreate airflow-webserver airflow-scheduler`

---

## 4. Airflow configuration

### AWS connection

In the Airflow UI go to **Admin → Connections** and verify `aws_default` exists. No credentials need to be set — the EC2 IAM instance profile provides them automatically. Region is set via the `AWS_DEFAULT_REGION` environment variable.

### Airflow Variables

Go to **Admin → Variables** and create:

| Key | Default | Description |
|---|---|---|
| `tinycnn_threshold` | `0.5` | Probability threshold for TinyCNN binary filter |
| `birdnet_top_k` | `5` | Number of top species to record per clip |

Values are deserialized as JSON, so enter plain numbers (e.g. `0.5`, `5`) — no quotes.

### Unpause the DAG

In the Airflow UI, toggle **rainforest_pipeline** from paused to active.

---

## 5. Running the pipeline

### Upload clips

```bash
# From local machine — upload a batch of .wav files to the incoming prefix
aws s3 cp /path/to/clips/ s3://rainforest-audio-kwf/clips/incoming/ --recursive --include "*.wav"

# Or use the demo batch script (samples 830 background + 170 meaningful from labeled dataset)
python scripts/upload_demo_batch.py
```

All clips must be in `clips/incoming/` **before** triggering the DAG — the `list_new_clips` task snapshots the prefix at run time.

### Trigger the DAG

In the Airflow UI, click the play button next to `rainforest_pipeline` → **Trigger DAG**.

The pipeline runs 8 tasks in sequence:

```
s3_sensor → list_new_clips → download_clips → tinycnn_filter → birdnet_infer → write_predictions → monitor_drift → cleanup_temp
```

### Validate with a single clip first

Before uploading a large batch, run one clip end-to-end to confirm the stack is wired up correctly:

```bash
aws s3 cp /path/to/any_clip.wav s3://rainforest-audio-kwf/clips/incoming/
```

Trigger the DAG and watch it in the Airflow UI. Expect ~3 minutes for the first run — PyTorch and TFLite both load their models from scratch on the first invocation (cold start). Subsequent clips in the same task run in ~40–70ms each because the models stay loaded in memory. If the first run looks slow, that's normal; if it looks slow on clip 500 of a batch, something else is wrong.

### Check results

```bash
# Batch summary (one record per DAG run)
aws dynamodb scan \
  --table-name birdnet-predictions \
  --filter-expression "begins_with(clip_id, :b)" \
  --expression-attribute-values '{":b": {"S": "BATCH#"}}' \
  --region us-east-1

# Predictions CSV
aws s3 ls s3://rainforest-audio-kwf/clips/results/ --recursive
```

> **BirdNET outputs raw logits, not probabilities:** After the first successful end-to-end run, DynamoDB showed confidence values like `-2.6275` — which looks like a bug at first glance. The BirdNET v2.4 TFLite model returns unbounded logit scores, not probabilities. The fix is to apply sigmoid (`1 / (1 + exp(-x))`) after inference. Without it, every downstream consumer of the table has to know to interpret negative numbers as high-confidence detections, which is unintuitive and error-prone.

> **TinyCNN mel spectrogram parameters must match training exactly:** The first 1000-clip demo batch showed TinyCNN filtering only 43% of clips — well below the expected ~83% given the labeled batch composition (830 not_meaningful + 170 meaningful). The batch summary in DynamoDB made the discrepancy obvious immediately. Tracing it back: the DAG was using `SR=22050, N_MELS=64` while the model was trained on `SR=48000, N_MELS=128, fmin=50, fmax=16000`. The model's `AdaptiveAvgPool2d` accepts variable input sizes, so no shape error was raised — the model ran silently on a completely wrong input distribution. After correcting the parameters, the same batch filtered 82.5% of clips, matching expectations. Always verify that inference preprocessing exactly replicates the training `compute_mel` function.

> **float64/float32 mismatch in TinyCNN:** During the first single-clip validation run, the `tinycnn_filter` task failed with `RuntimeError: Input type (double) and bias type (float) should be the same`. `librosa.power_to_db` returns a `float64` array by default, but PyTorch model weights are `float32`. The fix is a single cast before creating the tensor: `torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)`. No error at model load time — it only surfaces when the first forward pass runs.

---

## 6. CI/CD (GitHub Actions)

Two workflows run on every push to `main`:

### CI — lint and DAG syntax check (`.github/workflows/ci.yml`)

Runs in ~15 seconds. No Airflow install required.

1. **Lint** — `ruff check dags/ src/` catches style and unused-variable errors
2. **DAG syntax** — `ast.parse` on `dags/rainforest_pipeline.py` catches Python syntax errors before they reach EC2

> **Why not full DAG import validation:** Validating that the DAG actually imports correctly (i.e. `airflow dags list`) would require installing Airflow and all pipeline dependencies in the runner, adding ~2 minutes per push. For a single-developer project, `ast.parse` catches the vast majority of real mistakes fast enough to be useful.

> **YAML multiline gotcha:** The first CI push failed immediately (0s runtime, "workflow file issue"). The `run:` key for the DAG validation step used a bare multiline string starting with `python -c "` — YAML treats this as a double-quoted scalar and doesn't allow literal newlines inside it, so the file failed to parse before any job ran. The fix was to add `|` (literal block scalar) before the command so YAML passes the content as-is to the shell.

### CD — build and push to ECR (`.github/workflows/cd.yml`)

Runs in ~4-5 minutes. Triggers after CI passes.

1. Authenticates to ECR using `aws-actions/amazon-ecr-login`
2. Builds the Docker image from `Dockerfile`
3. Pushes two tags to `298247319628.dkr.ecr.us-east-1.amazonaws.com/rainforest-airflow`:
   - `:latest` — for easy pulls on EC2
   - `:<git-sha>` — for traceability (pinpoint exactly which commit a running container came from)

### GitHub Secrets required

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user `github-actions-ecr` — ECR push only |
| `AWS_SECRET_ACCESS_KEY` | Same user |
| `ECR_REGISTRY` | `298247319628.dkr.ecr.us-east-1.amazonaws.com` |

The `github-actions-ecr` IAM user has the minimum permissions needed to push to ECR (`ecr:GetAuthorizationToken` globally, push actions scoped to the `rainforest-airflow` repository only). It has no access to S3, DynamoDB, or EC2.

### Deploying the new image on EC2

CD does not auto-deploy to EC2 — you pull manually when ready:

```bash
ssh -i ~/.ssh/rainforest-key.pem ubuntu@<EC2_PUBLIC_IP>
cd /opt/rainforest-audio-detection

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    298247319628.dkr.ecr.us-east-1.amazonaws.com

docker pull 298247319628.dkr.ecr.us-east-1.amazonaws.com/rainforest-airflow:latest
docker tag 298247319628.dkr.ecr.us-east-1.amazonaws.com/rainforest-airflow:latest rainforest-airflow:latest
docker compose up -d --force-recreate airflow-webserver airflow-scheduler
```

> **Why no auto-deploy:** Wiring GitHub Actions to SSH into EC2 or use SSM requires storing EC2 credentials in GitHub Secrets and adds operational complexity that doesn't demonstrate additional ML engineering skill. For a production system this would be a natural next step.

---

## 7. MLflow experiment tracking

MLflow runs as a separate Docker service alongside Airflow. It provides a persistent experiment store for both pipeline monitoring (logged by the DAG) and model development work (logged from notebooks).

### Service setup

MLflow is defined in `docker-compose.yml` as a separate service using `Dockerfile.mlflow`:

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==3.13.0
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
```

Run data is stored in a named Docker volume (`mlflow_data`) so it survives container restarts and image rebuilds.

Start the service:

```bash
docker compose up -d mlflow
```

### Accessing the MLflow UI

Port 5000 is not exposed publicly (campus and corporate networks commonly block non-standard ports). Access via SSH tunnel instead:

```bash
# Open the tunnel (keep this terminal open)
ssh -i ~/.ssh/rainforest-key.pem -L 5000:localhost:5000 ubuntu@<EC2_PUBLIC_IP>

# Then open in browser
http://localhost:5000
```

### Experiments

| Experiment | Logged by | Contents |
|---|---|---|
| `tinycnn_binary_filter` | `notebooks/09_mlflow_tracking.ipynb` | v1–v4 training runs: params, val metrics |
| `birdnet_compression` | `notebooks/09_mlflow_tracking.ipynb` | PTQ variants: size, top-1 fidelity at multiple confidence thresholds |
| `pipeline_monitoring` | `monitor_drift` DAG task | Per-run: meaningful rate, delta vs baseline, drift flag, mean confidence |

> **MLflow 3.x `MLFLOW_ALLOWED_HOSTS` gotcha:** MLflow 3.x validates the HTTP `Host` header on every incoming request and rejects anything not on the allowlist with a `403 Forbidden`. Accessing the UI directly via `http://<EC2_PUBLIC_IP>:5000` from a browser sends `Host: <EC2_PUBLIC_IP>:5000` — which MLflow rejects unless you add the IP to `MLFLOW_ALLOWED_HOSTS` in `docker-compose.yml`. The SSH tunnel avoids this entirely because all requests arrive with `Host: localhost`, which is always allowed.

> **t3.medium → t3.large upgrade:** MLflow 3.x uses ~1.9 GB RAM at idle (FastAPI + SQLAlchemy + background workers). On a t3.medium (4 GB), adding MLflow alongside Airflow's scheduler, webserver, and postgres pushed the instance to near-OOM — tasks started getting killed silently. Upgrading to t3.large (8 GB) gave enough headroom for all services with ~2 GB to spare for actual inference workloads.

> **Artifact logging from local notebooks:** `mlflow.log_artifact()` tries to write files directly to the artifact store path (`/mlflow/artifacts` inside the container). When the MLflow server is on EC2 but the notebook runs locally, the local Python process cannot reach the container filesystem — it fails with `OSError: [Errno 30] Read-only file system: '/mlflow'`. The fix is to omit `log_artifact` calls in notebooks that connect to a remote server. Model weights are stored in `outputs/models/` and tracked in git by filename — no need to duplicate them in MLflow for this project.

---

## 8. Drift monitoring

The `monitor_drift` task (task 7 in the DAG) runs after every BirdNET inference pass and logs pipeline health metrics to the `pipeline_monitoring` MLflow experiment.

### What gets logged

| Metric | Description |
|---|---|
| `meaningful_rate` | Fraction of clips that passed TinyCNN and were inferred by BirdNET |
| `meaningful_rate_delta` | Delta from the baseline (positive = more meaningful than expected) |
| `drift_flag` | `1` if `abs(delta) > threshold`, else `0` |
| `mean_top1_confidence` | Mean BirdNET top-1 confidence score across meaningful clips |
| `n_clips_total` | Total clips in the batch |
| `n_meaningful` | Clips that passed TinyCNN |

### Baseline and threshold

```python
BASELINE_MEANINGFUL_RATE = 0.175   # established from 1000-clip demo batch (830/170 split)
DRIFT_THRESHOLD          = 0.15    # flag if meaningful rate shifts by more than 15 percentage points
```

The baseline was set from the empirical distribution in the labeled dataset: ~17.5% of AudioMoth recordings contain meaningful bird activity. A shift larger than 15 percentage points (e.g. rate drops to 2% or rises to 33%) suggests a change in recording conditions, equipment placement, or season — worth reviewing before trusting inference results.

### Viewing drift over time

Open the MLflow UI (via SSH tunnel), navigate to the `pipeline_monitoring` experiment, and use the **Chart** view to plot `meaningful_rate` and `drift_flag` across runs. Each point is one DAG run.

---

## 9. Updating the stack

After pushing code changes:

```bash
ssh -i ~/.ssh/rainforest-key.pem ubuntu@<EC2_PUBLIC_IP>
cd /opt/rainforest-audio-detection
git pull
# DAG changes (dags/) are picked up automatically by the scheduler — no restart needed.

# For Dockerfile changes, rebuild locally or pull the ECR image built by CD:
docker pull 298247319628.dkr.ecr.us-east-1.amazonaws.com/rainforest-airflow:latest
docker tag 298247319628.dkr.ecr.us-east-1.amazonaws.com/rainforest-airflow:latest rainforest-airflow:latest
docker compose up -d --force-recreate airflow-webserver airflow-scheduler
```
