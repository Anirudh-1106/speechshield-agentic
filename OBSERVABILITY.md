# SpeechShield Observability Guide

## Overview

SpeechShield v2 includes comprehensive observability infrastructure for production monitoring, debugging, and alerting. This guide covers structured logging, Prometheus metrics, health checks, and distributed tracing.

**Components:**
- **Structured Logging** — JSON-formatted logs with correlation IDs for tracing
- **Prometheus Metrics** — 15+ metrics capturing latency, throughput, and errors
- **Health Checks** — Component status tracking for orchestration platforms
- **Metrics HTTP Server** — FastAPI endpoints for scraping and debugging

---

## Table of Contents

1. [Structured Logging](#structured-logging)
2. [Prometheus Metrics](#prometheus-metrics)
3. [Health Checks](#health-checks)
4. [Distributed Tracing](#distributed-tracing)
5. [Metrics Server](#metrics-server)
6. [Configuration](#configuration)
7. [Prometheus Queries](#prometheus-queries)
8. [Troubleshooting](#troubleshooting)

---

## Structured Logging

### Overview

Structured logging uses **structlog** to emit JSON logs in production and human-readable logs in development. Logs include:
- **Timestamp** — ISO 8601 format
- **Level** — DEBUG, INFO, WARNING, ERROR
- **Logger** — Module name (e.g., `agents.asr_agent`)
- **Message** — Log message
- **Correlation ID** — Trace request through entire system
- **Metadata** — Custom fields for context

### Setup

```python
from core.logging import configure_logging, get_logger

# Call once at startup (already done in main.py)
configure_logging(
    level="INFO",           # DEBUG, INFO, WARNING, ERROR
    format="json",          # "json" or "text"
    log_dir="./logs",       # Directory for file logs (optional)
    enable_file_logging=False  # Set to True for production
)

# Get module logger
logger = get_logger("agents.asr_agent")

# Log structured events
logger.info("asr_inference_started", model="faster-whisper", language="en")
logger.error("asr_inference_failed", error="timeout", duration_ms=5000)
```

### JSON Log Format (Production)

```json
{
  "timestamp": "2024-12-19T14:30:45.123456Z",
  "level": "INFO",
  "logger": "agents.asr_agent",
  "event": "asr_inference_completed",
  "duration_ms": 1250,
  "model": "faster-whisper",
  "correlation_id": "trace-abc-123-xyz",
  "utterance_id": "utt-456"
}
```

### Text Log Format (Development)

```
2024-12-19 14:30:45.123 [INFO] agents.asr_agent: asr_inference_completed duration_ms=1250 model=faster-whisper correlation_id=trace-abc-123-xyz utterance_id=utt-456
```

### Best Practices

1. **Use named parameters instead of f-strings:**
   ```python
   # ✅ Good
   logger.info("event_occurred", event_id=123, status="success")
   
   # ❌ Bad
   logger.info(f"Event {123} occurred with status success")
   ```

2. **Include correlation IDs in every log:**
   ```python
   logger.info("processing_utterance", correlation_id=event.correlation_id, utterance_id=utt_id)
   ```

3. **Use consistent event names (snake_case):**
   ```python
   logger.info("asr_inference_started")   # ✅ Good
   logger.info("ASR inference started")   # ❌ Bad
   ```

---

## Prometheus Metrics

### Metric Types

#### 1. **Histograms** (Latencies)

Record request latencies with automatic quantile calculation.

```python
from core.metrics import metrics

m = metrics()

# Time a block of code
with m.asr_latency_seconds.time():
    result = asr_inference(audio_chunk)

# Or record manually
elapsed = time.time() - start_time
m.asr_latency_seconds.observe(elapsed)
```

**Available latency histograms:**
- `speechshield_asr_latency_seconds` — ASR inference (buckets: 0.1, 0.5, 1, 2, 5, 10, 30s)
- `speechshield_toxicity_latency_seconds` — Toxicity detection (buckets: 0.01, 0.05, 0.1, 0.5, 1, 5s)
- `speechshield_rewrite_latency_seconds` — LLM rewrite (buckets: 0.5, 1, 2, 5, 10, 20s)
- `speechshield_tts_latency_seconds` — Text-to-speech (buckets: 0.1, 0.5, 1, 2, 5, 15s)
- `speechshield_pipeline_latency_seconds` — End-to-end pipeline (buckets: 0.1, 0.3, 0.5, 1, 2, 5s)

#### 2. **Counters** (Cumulative Events)

Record total occurrences of events with optional labels.

```python
m = metrics()

# Simple counter
m.audio_chunks_dropped_total.inc()

# Counter with labels
m.toxicity_detections_total.labels(severity="high").inc()
m.toxicity_detections_total.labels(severity="low").inc(2)  # Increment by 2

m.rewrite_attempts_total.labels(status="success").inc()
m.rewrite_attempts_total.labels(status="timeout").inc()
```

**Available counters:**
- `speechshield_toxicity_detections_total` — (labels: severity=low/medium/high)
- `speechshield_rewrite_attempts_total` — (labels: status=success/timeout/error)
- `speechshield_rewrite_fallback_total` — (labels: reason=llm_timeout/llm_error/confidence_low)
- `speechshield_audio_chunks_dropped_total` — (no labels)
- `speechshield_utterances_processed_total` — (labels: status=success/error)
- `speechshield_asr_errors_total` — (labels: reason=timeout/no_speech/decode_error)

#### 3. **Gauges** (Current State)

Record instantaneous values that can go up or down.

```python
m = metrics()

# Set absolute value
m.audio_queue_depth_samples.set(10)
m.active_speakers.set(3)

# Increment/decrement
m.pipeline_pass_counter.inc()
m.pipeline_pass_counter.dec()

# Set time-series snapshot
m.avg_latency_ms.set(250.5)
```

**Available gauges:**
- `speechshield_audio_queue_depth_samples`
- `speechshield_asr_queue_depth_samples`
- `speechshield_toxicity_queue_depth_samples`
- `speechshield_rewrite_queue_depth_samples`
- `speechshield_active_speakers`
- `speechshield_pipeline_pass_counter`
- `speechshield_avg_latency_ms`

#### 4. **Summaries** (Quantiles)

Record quantile percentiles for latency analysis.

```python
m = metrics()

# Record value (calculates p50, p95, p99 automatically)
elapsed = time.time() - start_time
m.pipeline_latency_quantiles.observe(elapsed)
```

**Available summaries:**
- `speechshield_pipeline_latency_quantiles_seconds` — Quantiles: p50, p95, p99

### Agent Metrics Integration

Each agent should emit metrics on key operations:

```python
from core.metrics import metrics
from core.logging import get_logger

logger = get_logger("agents.asr_agent")
m = metrics()

class ASRAgent:
    async def process_audio(self, event):
        # Record correlation ID
        correlation_id = event.correlation_id
        logger.info("asr_inference_started", correlation_id=correlation_id)
        
        start_time = time.time()
        try:
            # Time the operation
            with m.asr_latency_seconds.time():
                result = await self.inference(event.data)
            
            m.utterances_processed_total.labels(status="success").inc()
            logger.info("asr_inference_completed", 
                       correlation_id=correlation_id,
                       duration_ms=(time.time()-start_time)*1000)
        except TimeoutError as e:
            m.asr_errors_total.labels(reason="timeout").inc()
            logger.error("asr_inference_timeout", 
                        correlation_id=correlation_id,
                        error=str(e))
```

---

## Health Checks

### Overview

Health checks track the status of all components (agents, event bus, etc.) for:
- Kubernetes readiness/liveness probes
- System diagnostics
- Automated alerting

### Component States

| State | Meaning |
|-------|---------|
| `UNKNOWN` | Initial state, not yet checked |
| `STARTING` | Component starting up |
| `READY` | Component healthy and operational |
| `DEGRADED` | Component working but with issues |
| `ERROR` | Component failed or unhealthy |
| `SHUTDOWN` | Component shutting down |

### Usage

```python
from core.health import health_checks, HealthStatus

hc = health_checks()

# Mark component ready
hc.mark_ready("ASRAgent")

# Mark component as error
hc.mark_error("ToxicityAgent", "Model failed to load")

# Mark component degraded
hc.mark_degraded("AudioAgent", "Queue latency high")

# Check system status
status = hc.get_status()
print(f"System ready: {status['ready']}")  # true/false
print(f"System alive: {status['alive']}")  # true/false

# Get component health
asr_health = hc.get_component("ASRAgent")
print(f"ASR Status: {asr_health.status}")
print(f"ASR Errors: {asr_health.error_count}")
```

### Health Endpoint Response

```json
{
  "ready": true,
  "alive": true,
  "uptime_seconds": 3600,
  "components": {
    "EventBus": {
      "status": "ready",
      "message": "Event bus operational",
      "error_count": 0,
      "last_check_time": "2024-12-19T14:30:45Z"
    },
    "ASRAgent": {
      "status": "ready",
      "message": "ASR model loaded",
      "error_count": 0,
      "last_check_time": "2024-12-19T14:30:42Z"
    }
  }
}
```

### Kubernetes Integration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: speechshield
spec:
  template:
    spec:
      containers:
      - name: speechshield
        livenessProbe:
          httpGet:
            path: /live
            port: 9501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9501
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## Distributed Tracing

### Correlation IDs

Every event gets a unique correlation ID for tracing through the entire pipeline:

```python
from core.event_bus import Event
from core.logging import set_correlation_id, get_logger

logger = get_logger("orchestrator")

# Event auto-generates correlation ID
event = Event("audio_received", {"audio": audio_data})
print(event.correlation_id)  # e.g., "550e8400-e29b-41d4-a716-446655440000"

# Or provide custom correlation ID (e.g., from HTTP request)
event = Event("audio_received", {"audio": audio_data}, 
              correlation_id="user-123-request-456")

# Set context for all logs in request scope
set_correlation_id(event.correlation_id)

# All logs now include correlation ID automatically
logger.info("processing_event", utterance_id="utt-123")
# → JSON: {"correlation_id": "550e8400...", "utterance_id": "utt-123", ...}
```

### Tracing Through Pipeline

```
HTTP Request (correlation_id: "trace-abc-123")
  ↓
[AudioAgent] logs: correlation_id=trace-abc-123 audio_received
  ↓
Event("audio_ready") with correlation_id=trace-abc-123
  ↓
[ASRAgent] logs: correlation_id=trace-abc-123 asr_inference_started
  ↓
Event("asr_complete") with correlation_id=trace-abc-123
  ↓
[ToxicityAgent] logs: correlation_id=trace-abc-123 toxicity_check_started
```

**Query all logs for a trace:**
```
# Using ELK Stack, DataDog, etc.
correlation_id: "trace-abc-123"
```

---

## Metrics Server

### Overview

FastAPI server runs on port `config.dashboard.port + 1000` (default: 9501) with 5 endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /metrics` | Prometheus scrape format |
| `GET /health` | Full health status + components |
| `GET /status` | System status with current metric values |
| `GET /ready` | Kubernetes readiness probe (200/503) |
| `GET /live` | Kubernetes liveness probe (200/503) |

### Running

```python
# main.py automatically starts metrics server
from core.metrics_server import start_metrics_server

# Or start manually
stop_server = start_metrics_server(host="127.0.0.1", port=9501)

# Stop when done
stop_server()
```

### Endpoints

#### `/metrics` — Prometheus Scrape Format

```bash
$ curl http://localhost:9501/metrics

# HELP speechshield_asr_latency_seconds ASR (speech-to-text) inference latency
# TYPE speechshield_asr_latency_seconds histogram
speechshield_asr_latency_seconds_bucket{le="0.1"} 45
speechshield_asr_latency_seconds_bucket{le="0.5"} 120
speechshield_asr_latency_seconds_bucket{le="+Inf"} 150
speechshield_asr_latency_seconds_sum 125.45
speechshield_asr_latency_seconds_count 150
```

#### `/health` — Full Health Status

```bash
$ curl http://localhost:9501/health | python -m json.tool

{
  "ready": true,
  "alive": true,
  "uptime_seconds": 3600.5,
  "components": {
    "EventBus": {...},
    "AudioAgent": {...},
    "ASRAgent": {...}
  }
}
```

#### `/status` — System Status with Metrics

```bash
$ curl http://localhost:9501/status | python -m json.tool

{
  "timestamp": "2024-12-19T14:30:45Z",
  "version": "2.0.0",
  "uptime_seconds": 3600.5,
  "active_speakers": 3,
  "audio_queue_depth": 10,
  "asr_queue_depth": 5,
  "pipeline_latency_ms": 250.5,
  "avg_latency_ms": 245.3,
  "metrics": {
    "asr_latency_p50": 0.85,
    "asr_latency_p95": 2.1,
    "toxicity_detections": 15,
    "rewrite_attempts": 23
  }
}
```

#### `/ready` — Kubernetes Readiness

```bash
$ curl -w "%{http_code}" http://localhost:9501/ready
200  # Ready

# Returns 503 if any component not ready
```

#### `/live` — Kubernetes Liveness

```bash
$ curl -w "%{http_code}" http://localhost:9501/live
200  # Alive

# Returns 503 if system has errors
```

---

## Configuration

### Environment Variables

```bash
# Logging
export SPEECHSHIELD_LOGGING_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
export SPEECHSHIELD_LOGGING_FORMAT=json             # json or text
export SPEECHSHIELD_LOGGING_LOG_DIR=./logs
export SPEECHSHIELD_LOGGING_ENABLE_FILE_LOGGING=true

# Dashboard/Metrics
export SPEECHSHIELD_DASHBOARD_PORT=8501             # Dashboard port
# Metrics server runs on port: DASHBOARD_PORT + 1000
```

### YAML Configuration

```yaml
# config/prod.yaml
logging:
  level: INFO
  format: json
  log_dir: /var/log/speechshield
  enable_file_logging: true

dashboard:
  port: 8501  # Metrics server on 9501
```

---

## Prometheus Queries

### Essential Queries

**1. Request Rate (requests/second)**
```
rate(speechshield_utterances_processed_total[5m])
```

**2. Error Rate**
```
rate(speechshield_asr_errors_total[5m]) + rate(speechshield_utterances_processed_total{status="error"}[5m])
```

**3. P95 Latency**
```
histogram_quantile(0.95, rate(speechshield_pipeline_latency_seconds_bucket[5m]))
```

**4. Active Queue Depth**
```
speechshield_audio_queue_depth_samples
```

**5. Toxicity Detection Rate**
```
rate(speechshield_toxicity_detections_total[5m])
```

### Alert Rules

```yaml
# prometheus-rules.yaml
groups:
  - name: speechshield
    rules:
      - alert: HighErrorRate
        expr: rate(speechshield_asr_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High ASR error rate"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(speechshield_pipeline_latency_seconds_bucket[5m])) > 5
        for: 5m
        annotations:
          summary: "Pipeline P95 latency > 5s"

      - alert: QueueBacklog
        expr: speechshield_audio_queue_depth_samples > 100
        for: 2m
        annotations:
          summary: "Audio queue depth exceeded 100 samples"

      - alert: ComponentDown
        expr: speechshield_health_checks_component_status{status="error"} == 1
        for: 1m
        annotations:
          summary: "Component {{ $labels.component }} is in ERROR state"
```

---

## Troubleshooting

### Metrics Not Appearing

1. **Check metrics server is running:**
   ```bash
   curl http://localhost:9501/metrics
   ```

2. **Verify Prometheus is scraping:**
   ```yaml
   # prometheus.yml
   scrape_configs:
     - job_name: 'speechshield'
       static_configs:
         - targets: ['localhost:9501']
   ```

3. **Check agent metrics emission:**
   ```python
   from core.metrics import metrics
   m = metrics()
   m.asr_latency_seconds.observe(1.5)  # Manually record metric
   ```

### Correlation IDs Missing

1. **Ensure correlation_id is set in event handlers:**
   ```python
   logger.info("event", correlation_id=event.correlation_id)
   ```

2. **Check log format is JSON:**
   ```bash
   export SPEECHSHIELD_LOGGING_FORMAT=json
   ```

### Health Checks Not Updating

1. **Verify health_checks() is being called:**
   ```python
   from core.health import health_checks
   hc = health_checks()
   print(hc.get_status())
   ```

2. **Mark components on startup/shutdown:**
   ```python
   hc.mark_starting("ASRAgent")
   # ...load model...
   hc.mark_ready("ASRAgent")
   ```

### High Queue Depths

1. **Monitor queue metrics:**
   ```
   http://localhost:9501/status → audio_queue_depth
   ```

2. **Adjust worker counts:**
   ```yaml
   # config/prod.yaml
   agents:
     asr:
       num_workers: 4  # Increase if queue backing up
   ```

3. **Check P95 latency:**
   ```
   histogram_quantile(0.95, rate(speechshield_asr_latency_seconds_bucket[5m]))
   ```

---

## References

- [structlog Documentation](https://www.structlog.org/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)
- [Prometheus Queries](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
