#!/bin/bash
# Collect time-series data from k3s gpu_burn pods
# Captures full gpu_burn log + nvidia-smi SM util from host
#
# Usage: bash k3s_collect.sh <label> <gpucores>
#   label     - variant label (e.g., "v5", "sd", "aimd", "sdx3")
#   gpucores  - SM core limit (0 = no limit)
#
# Output:
#   /tmp/gpu-bench-ts/k3s/<label>_sm<gpucores>_gpuburn.log
#   /tmp/gpu-bench-ts/k3s/<label>_sm<gpucores>_smi.csv
set -e

DURATION=30
LABEL=${1:-"v5"}
SM=${2:-40}
OUTDIR="/tmp/gpu-bench-ts/k3s"
mkdir -p "$OUTDIR"

echo "=== k3s time-series: ${LABEL} gpucores=${SM} ${DURATION}s ==="

# Clean up
kubectl delete pod gpu-ts --ignore-not-found --wait=true 2>/dev/null || true
sleep 3

# Create pod
if [ "$SM" -eq 0 ]; then
  cat <<EOF | kubectl apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: gpu-ts
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: oguzpastirmaci/gpu-burn
      command: ["/bin/sh", "-c", "GPU_CORE_UTILIZATION_POLICY=FORCE ./gpu_burn $DURATION"]
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 3000
EOF
else
  cat <<EOF | kubectl apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: gpu-ts
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: oguzpastirmaci/gpu-burn
      command: ["/bin/sh", "-c", "GPU_CORE_UTILIZATION_POLICY=FORCE ./gpu_burn $DURATION"]
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 3000
          nvidia.com/gpucores: $SM
EOF
fi

echo "Waiting for pod to start..."
kubectl wait --for=condition=Ready pod/gpu-ts --timeout=60s 2>/dev/null

# Start nvidia-smi monitoring in background (100ms interval)
SMILOG="${OUTDIR}/${LABEL}_sm${SM}_smi.csv"
echo "timestamp,utilization_gpu" > "$SMILOG"
(
  while true; do
    TS=$(date +%s.%N)
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "-1")
    echo "${TS},${UTIL}" >> "$SMILOG"
    sleep 0.1
  done
) &
SMI_PID=$!

echo "Monitoring started (PID=${SMI_PID}), waiting for gpu_burn to complete..."

# Wait for completion
for i in $(seq 1 120); do
  PHASE=$(kubectl get pod gpu-ts -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
    break
  fi
  sleep 3
done

# Stop nvidia-smi monitoring
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

# Capture full gpu_burn log
GPULOG="${OUTDIR}/${LABEL}_sm${SM}_gpuburn.log"
kubectl logs gpu-ts > "$GPULOG" 2>/dev/null

# Get final proc'd count
PROCD=$(grep -oP "proc'd: \K\d+" "$GPULOG" | tail -1)
echo "Result: ${PROCD} proc'd"
echo "Logs: ${GPULOG}"
echo "SMI:  ${SMILOG}"

kubectl delete pod gpu-ts --ignore-not-found 2>/dev/null || true
echo "=== Done ==="
