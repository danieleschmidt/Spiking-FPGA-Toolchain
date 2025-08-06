# Spiking-FPGA-Toolchain Deployment Guide

This document provides comprehensive deployment instructions for the Spiking-FPGA-Toolchain in various environments.

## 🐳 Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 20GB+ disk space for cache and outputs

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danieleschmidt/Spiking-FPGA-Toolchain
   cd Spiking-FPGA-Toolchain
   ```

2. **Build and run with Docker Compose:**
   ```bash
   cd docker
   mkdir -p input output cache logs
   docker-compose up -d
   ```

3. **Test the deployment:**
   ```bash
   docker-compose exec spiking-fpga-toolchain spiking-fpga --version
   ```

### Configuration

Environment variables can be set in `docker/docker-compose.yml`:

- `SPIKING_FPGA_CACHE_DIR`: Cache directory (default: `/app/cache`)
- `SPIKING_FPGA_LOG_DIR`: Log directory (default: `/app/logs`) 
- `SPIKING_FPGA_LOG_LEVEL`: Logging level (default: `INFO`)
- `SPIKING_FPGA_MAX_WORKERS`: Max concurrent workers (default: `4`)
- `SPIKING_FPGA_ENABLE_MONITORING`: Enable system monitoring (default: `true`)

### Volume Mounts

- `./input:/app/input:ro` - Network definition files (read-only)
- `./output:/app/output` - Generated HDL and reports
- `./cache:/app/cache` - Compilation cache for performance
- `./logs:/app/logs` - Application logs

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes 1.20+
- kubectl configured with cluster access
- StorageClass for persistent volumes
- Compute-optimized nodes recommended

### Deployment Steps

1. **Create namespace:**
   ```bash
   kubectl create namespace terragon-labs
   ```

2. **Deploy storage:**
   ```bash
   kubectl apply -f kubernetes/storage.yaml
   ```

3. **Deploy application:**
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   ```

4. **Verify deployment:**
   ```bash
   kubectl get pods -n terragon-labs
   kubectl logs -f deployment/spiking-fpga-toolchain -n terragon-labs
   ```

### Resource Requirements

**Minimum per pod:**
- CPU: 1000m (1 core)
- Memory: 2Gi
- Storage: 50Gi (cache) + 100Gi (output)

**Recommended per pod:**
- CPU: 4000m (4 cores)
- Memory: 8Gi
- Fast SSD storage for cache

### Scaling

Horizontal scaling is supported:

```bash
kubectl scale deployment spiking-fpga-toolchain --replicas=5 -n terragon-labs
```

### Monitoring

The deployment includes:
- Health checks (liveness/readiness probes)
- Prometheus metrics endpoint on port 8080
- Resource monitoring and alerting

## 🚀 Production Deployment

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Kubernetes     │────│   Storage       │
│   (nginx/traefik)│    │   Cluster        │    │   (PVC/NFS)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────────┐
                       │   Monitoring     │
                       │ (Prometheus/     │
                       │  Grafana)        │
                       └──────────────────┘
```

### High Availability Setup

1. **Multi-zone deployment:**
   ```yaml
   spec:
     affinity:
       podAntiAffinity:
         requiredDuringSchedulingIgnoredDuringExecution:
         - labelSelector:
             matchLabels:
               app: spiking-fpga-toolchain
           topologyKey: topology.kubernetes.io/zone
   ```

2. **Persistent storage:**
   - Use distributed storage (Ceph, GlusterFS, or cloud PVs)
   - Enable backup and disaster recovery
   - Configure storage classes with appropriate performance tiers

3. **Load balancing:**
   - Deploy ingress controller (nginx, traefik)
   - Configure SSL/TLS termination
   - Set up health check endpoints

### Security Configuration

1. **Network Policies:**
   ```bash
   kubectl apply -f kubernetes/network-policies.yaml
   ```

2. **Pod Security Policies:**
   - Non-root user execution
   - Read-only root filesystem
   - Dropped Linux capabilities
   - Resource limits enforced

3. **Secrets Management:**
   ```bash
   kubectl create secret generic spiking-fpga-secrets \
     --from-literal=cache-encryption-key=<key> \
     --from-literal=log-signing-key=<key>
   ```

### Performance Tuning

1. **Node Selection:**
   ```yaml
   nodeSelector:
     kubernetes.io/arch: amd64
     node-type: compute-optimized
   ```

2. **Resource Optimization:**
   - Use CPU affinity for compute-intensive tasks
   - Configure memory limits based on network sizes
   - Enable NUMA awareness for large deployments

3. **Storage Optimization:**
   - Use NVMe SSDs for cache storage
   - Configure appropriate filesystem (ext4/xfs)
   - Enable compression for logs and cache

## 📊 Monitoring and Observability

### Metrics Collection

The toolchain exposes metrics in Prometheus format:

- Compilation success/failure rates
- Processing latencies
- Resource utilization
- Cache hit rates
- System health metrics

### Log Aggregation

Structured JSON logging with:

- Request tracing
- Performance metrics  
- Error details
- Security events

Example log shipping to ELK stack:

```yaml
volumeMounts:
- name: logs
  mountPath: /app/logs
- name: filebeat-config
  mountPath: /etc/filebeat
```

### Alerting Rules

Critical alerts for:

- Pod crash loops
- High memory usage (>80%)
- Compilation failure rate (>10%)
- Cache storage full (>90%)
- Response time degradation

## 🔧 Maintenance

### Updates and Rollbacks

1. **Rolling updates:**
   ```bash
   kubectl set image deployment/spiking-fpga-toolchain \
     spiking-fpga-compiler=terragon/spiking-fpga-toolchain:v0.2.0
   ```

2. **Rollback if needed:**
   ```bash
   kubectl rollout undo deployment/spiking-fpga-toolchain
   ```

### Backup Procedures

1. **Configuration backup:**
   ```bash
   kubectl get configmap spiking-fpga-config -o yaml > config-backup.yaml
   ```

2. **Storage backup:**
   - Schedule regular PVC snapshots
   - Backup cache and compilation artifacts
   - Test restore procedures regularly

### Troubleshooting

1. **Check pod logs:**
   ```bash
   kubectl logs -f pod/<pod-name> -n terragon-labs
   ```

2. **Debug networking:**
   ```bash
   kubectl exec -it <pod-name> -- netstat -tlnp
   ```

3. **Resource inspection:**
   ```bash
   kubectl describe pod <pod-name> -n terragon-labs
   kubectl top pods -n terragon-labs
   ```

## 📈 Scaling Considerations

### Vertical Scaling

Increase resources per pod based on workload:

```yaml
resources:
  requests:
    cpu: "2000m"
    memory: "4Gi"
  limits:
    cpu: "8000m" 
    memory: "16Gi"
```

### Horizontal Scaling

Add more pods for increased throughput:

```yaml
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
```

### Auto-scaling

Configure HPA based on CPU/memory or custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spiking-fpga-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spiking-fpga-toolchain
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🌍 Multi-Region Deployment

For global deployment:

1. **Regional clusters:**
   - Deploy in multiple geographic regions
   - Use local storage for each region
   - Configure cross-region backup replication

2. **Traffic routing:**
   - Geographic DNS routing
   - Health-based failover
   - Load balancing between regions

3. **Data synchronization:**
   - Sync compilation cache between regions
   - Replicate configuration changes
   - Coordinate updates across regions

## 📞 Support and Troubleshooting

For deployment issues:

1. Check the troubleshooting section in README.md
2. Review application logs for error details
3. Verify resource requirements are met
4. Ensure network policies allow required traffic
5. Contact support at support@terragon-labs.ai

## 📋 Deployment Checklist

- [ ] Prerequisites installed and configured
- [ ] Resource requirements verified
- [ ] Storage provisioned and tested
- [ ] Security policies applied
- [ ] Monitoring and alerting configured
- [ ] Backup procedures tested
- [ ] Health checks validated
- [ ] Performance benchmarks run
- [ ] Documentation updated
- [ ] Team trained on operations

---

For the latest deployment updates and best practices, visit:
https://github.com/danieleschmidt/Spiking-FPGA-Toolchain/blob/main/DEPLOYMENT.md