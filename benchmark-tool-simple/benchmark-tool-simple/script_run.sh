#!/bin/bash
if [ "$#" -ne 2 ]; then
 echo "Usage: $0 <results_suffix> <port_num>"
 exit 1
fi


results_suffix=$1
port_num=$2

triples=(
 "1024 1024 1"
 "1024 1024 4"
 "1024 1024 8"
 "1024 1024 16"
 "1024 1024 32"
 "1024 1024 64"
 "1024 1024 128"
)

results_dir="$(pwd)/vllm-result/results-${results_suffix}_${port_num}"
if [ ! -d "$results_dir" ]; then
  mkdir -p "$results_dir"
fi


# Resolve model id from local server (override with PRETRAINED_ID or API_BASE if needed)
if [ -z "$PRETRAINED_ID" ]; then
  API_BASE="http://127.0.0.1:${port_num}"
  MODELS_ENDPOINT="$API_BASE/v1/models"

  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: 'jq' is required but not installed." >&2
    exit 1
  fi

  PRETRAINED_ID=$(curl -sf "$MODELS_ENDPOINT" | jq -r '.data[]? | select(.id != null) | .id' | head -n 1)

  if [ -z "$PRETRAINED_ID" ] || [ "$PRETRAINED_ID" = "null" ]; then
    echo "Error: Failed to parse PRETRAINED_ID from $MODELS_ENDPOINT" >&2
    exit 1
  fi
fi

VLLM_DIR="$(pwd)/vllm"
cd $VLLM_DIR

for triple in "${triples[@]}"; do
  set -- $triple
  input_tokens=$1
  output_tokens=$2
  parallel_requests=$3

  scenario_dir="$results_dir/$input_tokens.$output_tokens.$parallel_requests"
  if [ ! -d "$scenario_dir" ]; then
    mkdir -p "$scenario_dir"
  fi

  extra_monitor_args=""
  if [ -n "$MONITOR_SCOPE" ]; then
    extra_monitor_args="--monitor-scope $MONITOR_SCOPE"
  fi

    python benchmarks/benchmark_serving.py \
        --backend vllm \
        --model $PRETRAINED_ID \
        --dataset-name fixed \
        --random-input-len $input_tokens \
        --random-output-len $output_tokens \
        --max-concurrency $parallel_requests \
        --num-prompts $(($parallel_requests * 3)) \
        --result-dir "$scenario_dir" \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "25,50,75,90,95,99" \
        --host "127.0.0.1" \
        --port "$port_num" \
        --enable-device-monitor "npu" \
        --save-result \
        --save-detailed \

        $extra_monitor_args
done
cd -