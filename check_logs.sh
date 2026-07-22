#!/bin/bash
# Check slurm array job logs for failures vs successes.
# Checks ALL job submissions for the pattern; for each task ID uses the most
# recent log (highest job ID), so resubmitted tasks that succeeded override
# earlier failures.
#
# Usage: bash check_logs.sh [log_dir] [job_name_pattern]
# Defaults: log_dir=slurm/logs, pattern=qe_gemma_shard

LOG_DIR="${1:-slurm/logs}"
PATTERN="${2:-qe_gemma_shard}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/$LOG_DIR"

SUCCESS_MARKER="Done. Output dir:"
ERR_PATTERN="CUDA error|CANCELLED AT|out of memory|slurmstepd: error"

if ! ls "$LOG_DIR"/${PATTERN}_*.err &>/dev/null; then
    echo "No log files found matching '${PATTERN}_*.err' in $LOG_DIR"
    exit 1
fi

# Build a map: task_id -> most recent err file (highest job_id wins)
declare -A latest_err  # task_id -> file path

for err_file in "$LOG_DIR"/${PATTERN}_*.err; do
    [ -f "$err_file" ] || continue
    base=$(basename "$err_file" .err)
    # base format: {PATTERN}_{JOB_ID}_{TASK_ID}
    job_id=$(echo "$base" | sed "s/${PATTERN}_//" | cut -d_ -f1)
    task_id=$(echo "$base" | sed "s/${PATTERN}_${job_id}_//")

    current="${latest_err[$task_id]}"
    if [ -z "$current" ]; then
        latest_err[$task_id]="$err_file"
    else
        current_job=$(basename "$current" .err | sed "s/${PATTERN}_//" | cut -d_ -f1)
        if [ "$job_id" -gt "$current_job" ]; then
            latest_err[$task_id]="$err_file"
        fi
    fi
done

declare -a failed=()
declare -a succeeded=()
declare -a unknown=()

for task_id in $(echo "${!latest_err[@]}" | tr ' ' '\n' | sort -n); do
    err_file="${latest_err[$task_id]}"
    job_id=$(basename "$err_file" .err | sed "s/${PATTERN}_//" | cut -d_ -f1)

    if grep -q "$SUCCESS_MARKER" "$err_file"; then
        succeeded+=("$task_id")
    elif grep -qE "$ERR_PATTERN" "$err_file"; then
        reason=$(grep -E "$ERR_PATTERN" "$err_file" | head -1 | sed 's/^[[:space:]]*//')
        failed+=("$task_id (job $job_id): $reason")
    else
        unknown+=("$task_id (job $job_id)")
    fi
done

total=$(( ${#succeeded[@]} + ${#failed[@]} + ${#unknown[@]} ))
echo "Checked ${total} unique task IDs across all ${PATTERN} submissions"
echo ""
echo "✓ Succeeded: ${#succeeded[@]}"
echo "✗ Failed:    ${#failed[@]}"
echo "? Unknown (still running or empty): ${#unknown[@]}"
echo ""

if [ ${#failed[@]} -gt 0 ]; then
    echo "=== FAILED ==="
    for entry in "${failed[@]}"; do
        echo "  task $entry"
    done
    echo ""
    echo "Array IDs to resubmit:"
    ids=$(printf '%s\n' "${failed[@]}" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//')
    echo "  sbatch --array=${ids} slurm/${PATTERN}.sh"
    echo ""
fi

if [ ${#unknown[@]} -gt 0 ]; then
    echo "=== UNKNOWN (no success marker, no recognised error) ==="
    for entry in "${unknown[@]}"; do
        task_id=$(echo "$entry" | awk '{print $1}')
        echo "  task $entry"
        tail -3 "${latest_err[$task_id]}" 2>/dev/null | grep -v "^$" | head -2 | sed 's/^/    /'
    done
fi
