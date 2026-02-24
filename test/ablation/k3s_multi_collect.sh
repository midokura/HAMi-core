#!/bin/bash
# Collect time-series data from multiple concurrent gpu_burn pods.
#
# Usage: bash k3s_multi_collect.sh <label> <pod1:gpucores> [pod2:gpucores] ...
#
# Examples:
#   bash k3s_multi_collect.sh test a:20 b:30
#   bash k3s_multi_collect.sh test a:40 b:40
#   bash k3s_multi_collect.sh test a:50 b:50    # over-committed
#
# Output (per pod):
#   /tmp/gpu-bench-ts/k3s/<label>_multi_<pod>_sm<gpucores>_gpuburn.log
#   /tmp/gpu-bench-ts/k3s/<label>_multi_smi.csv    (GPU-wide SM utilization)
#   /tmp/gpu-bench-ts/k3s/<label>_multi_pmon.csv   (per-process SM utilization)
set -e

DURATION=30
LABEL=${1:?"Usage: $0 <label> <pod1:gpucores> [pod2:gpucores] ..."}
shift
PODS=("$@")

if [ ${#PODS[@]} -lt 1 ]; then
  echo "ERROR: specify at least one pod:gpucores pair" >&2
  exit 1
fi

OUTDIR="/tmp/gpu-bench-ts/k3s"
mkdir -p "$OUTDIR"

# Parse pod specs
declare -a POD_NAMES POD_CORES
for spec in "${PODS[@]}"; do
  IFS=':' read -r name cores <<< "$spec"
  if [ -z "$name" ] || [ -z "$cores" ]; then
    echo "ERROR: invalid spec '$spec', expected name:gpucores" >&2
    exit 1
  fi
  POD_NAMES+=("$name")
  POD_CORES+=("$cores")
done

echo "=== Multi-pod benchmark: ${LABEL} ==="
echo "Pods: ${PODS[*]}"
echo "Duration: ${DURATION}s"
echo ""

# Clean up any existing pods
for name in "${POD_NAMES[@]}"; do
  kubectl delete pod "gpu-ts-${name}" --ignore-not-found --wait=true 2>/dev/null || true
done
sleep 3

# Create all pods
for i in "${!POD_NAMES[@]}"; do
  name="${POD_NAMES[$i]}"
  cores="${POD_CORES[$i]}"
  podname="gpu-ts-${name}"
  echo "Creating ${podname} (gpucores=${cores})..."

  if [ "$cores" -eq 0 ]; then
    cat <<EOF | kubectl apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${podname}
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: oguzpastirmaci/gpu-burn
      command: ["/bin/sh", "-c", "GPU_CORE_UTILIZATION_POLICY=FORCE ./gpu_burn $DURATION"]
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 1500
EOF
  else
    cat <<EOF | kubectl apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${podname}
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: oguzpastirmaci/gpu-burn
      command: ["/bin/sh", "-c", "GPU_CORE_UTILIZATION_POLICY=FORCE ./gpu_burn $DURATION"]
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 1500
          nvidia.com/gpucores: ${cores}
EOF
  fi
done

# Wait for all pods to be ready
echo ""
echo "Waiting for all pods to start..."
for name in "${POD_NAMES[@]}"; do
  kubectl wait --for=condition=Ready "pod/gpu-ts-${name}" --timeout=120s 2>/dev/null
  echo "  gpu-ts-${name}: Running"
done

# Start nvidia-smi GPU-wide monitoring (100ms interval)
SMILOG="${OUTDIR}/${LABEL}_multi_smi.csv"
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

# Start nvidia-smi pmon for per-process SM utilization
PMONLOG="${OUTDIR}/${LABEL}_multi_pmon.csv"
echo "timestamp,pid,sm,mem,command" > "$PMONLOG"
(
  # pmon -d 1 outputs one sample per second per process
  nvidia-smi pmon -d 1 -s u 2>/dev/null | while IFS= read -r line; do
    # Skip comment lines
    case "$line" in \#*) continue ;; esac
    # Parse: gpu pid type sm mem enc dec jpg ofa command
    read -r gpu pid type sm mem enc dec jpg ofa cmd <<< "$line"
    if [ -n "$pid" ] && [ "$pid" != "-" ] && [ "$sm" != "-" ]; then
      TS=$(date +%s.%N)
      echo "${TS},${pid},${sm},${mem},${cmd}"
    fi
  done >> "$PMONLOG"
) &
PMON_PID=$!

echo ""
echo "Monitoring started (smi PID=${SMI_PID}, pmon PID=${PMON_PID})"
echo "Waiting for gpu_burn to complete..."

# Wait for all pods to complete
for attempt in $(seq 1 120); do
  all_done=true
  for name in "${POD_NAMES[@]}"; do
    PHASE=$(kubectl get pod "gpu-ts-${name}" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$PHASE" != "Succeeded" ] && [ "$PHASE" != "Failed" ]; then
      all_done=false
      break
    fi
  done
  if $all_done; then
    break
  fi
  sleep 3
done

# Stop monitoring
kill $SMI_PID 2>/dev/null || true
kill $PMON_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true
wait $PMON_PID 2>/dev/null || true

# Collect logs and results
echo ""
echo "=== Results ==="
for i in "${!POD_NAMES[@]}"; do
  name="${POD_NAMES[$i]}"
  cores="${POD_CORES[$i]}"
  podname="gpu-ts-${name}"
  GPULOG="${OUTDIR}/${LABEL}_multi_${name}_sm${cores}_gpuburn.log"

  kubectl logs "$podname" > "$GPULOG" 2>/dev/null
  PROCD=$(grep -oP "proc'd: \K\d+" "$GPULOG" | tail -1)
  echo "  ${podname} (gpucores=${cores}): ${PROCD:-0} proc'd  → ${GPULOG}"
done
echo "  nvidia-smi: ${SMILOG}"
echo "  pmon:       ${PMONLOG}"

# Clean up pods
for name in "${POD_NAMES[@]}"; do
  kubectl delete pod "gpu-ts-${name}" --ignore-not-found 2>/dev/null || true
done

echo ""
echo "=== Done ==="
