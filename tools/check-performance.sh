#!/bin/sh
set -eu

frames=${PERF_FRAMES:-2000}
iterations=${PERF_ITERATIONS:-5}
result_dir=${PERF_RESULT_DIR:-target/performance}
guard=benchmarks/aac-lc-decode-guard.json
binary=target/release/examples/benchmark_decode

mkdir -p "$result_dir"
cargo build --release --locked -p fdk-aac-rust --no-default-features \
  --example benchmark_decode
"$binary" --json "$frames" "$iterations" > "$result_dir/aac-lc-decode.json"

json_number() {
  key=$1
  file=$2
  sed -n "s/.*\"$key\":[[:space:]]*\([0-9.][0-9.]*\).*/\1/p" "$file" | head -n 1
}

minimum_realtime=$(json_number minimum_median_realtime_factor "$guard")
maximum_first_frame=$(json_number maximum_median_first_frame_microseconds "$guard")
maximum_rss=$(json_number maximum_peak_rss_kib "$guard")
maximum_binary=$(json_number maximum_binary_bytes "$guard")
actual_realtime=$(json_number median_realtime_factor "$result_dir/aac-lc-decode.json")
actual_first_frame=$(json_number median_first_frame_microseconds "$result_dir/aac-lc-decode.json")
actual_rss=$(json_number peak_rss_kib "$result_dir/aac-lc-decode.json")
actual_binary=$(json_number binary_bytes "$result_dir/aac-lc-decode.json")

for value in \
  "$minimum_realtime" "$maximum_first_frame" "$maximum_rss" "$maximum_binary" \
  "$actual_realtime" "$actual_first_frame" "$actual_rss" "$actual_binary"
do
  test -n "$value" || {
    echo "Missing numeric performance measurement or guard" >&2
    exit 1
  }
done

awk -v actual="$actual_realtime" -v limit="$minimum_realtime" \
  'BEGIN { exit !(actual >= limit) }' || {
  echo "Median realtime factor $actual_realtime is below catastrophic guard $minimum_realtime" >&2
  exit 1
}
awk -v actual="$actual_first_frame" -v limit="$maximum_first_frame" \
  'BEGIN { exit !(actual <= limit) }' || {
  echo "Median first-frame latency $actual_first_frame us exceeds guard $maximum_first_frame us" >&2
  exit 1
}
awk -v actual="$actual_rss" -v limit="$maximum_rss" \
  'BEGIN { exit !(actual <= limit) }' || {
  echo "Peak RSS $actual_rss KiB exceeds guard $maximum_rss KiB" >&2
  exit 1
}
awk -v actual="$actual_binary" -v limit="$maximum_binary" \
  'BEGIN { exit !(actual <= limit) }' || {
  echo "Benchmark binary size $actual_binary bytes exceeds guard $maximum_binary bytes" >&2
  exit 1
}

{
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  rustc -Vv
  uname -a
  if [ -r /proc/cpuinfo ]; then
    sed -n 's/^model name[[:space:]]*: /cpu=/p' /proc/cpuinfo | head -n 1
  fi
} > "$result_dir/environment.txt"

cat "$result_dir/aac-lc-decode.json"
echo "Performance catastrophe guards passed."
