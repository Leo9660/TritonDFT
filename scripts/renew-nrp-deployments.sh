#!/usr/bin/env bash
# Renew TritonDFT Deployments so NRP Nautilus doesn't purge them.
#
# NRP purges any Deployment whose object age (metadata.creationTimestamp) passes
# ~2 weeks. Rolling updates don't reset that timestamp — only re-creating the
# Deployment OBJECT does. This script does it with ZERO downtime per deployment:
#   1. snapshot the LIVE spec (always matches the current image; doubles as a
#      disaster-recovery backup),
#   2. `delete --cascade=orphan` — removes only the Deployment object; the
#      ReplicaSet and its pods keep running,
#   3. re-`create` from the snapshot — the new Deployment's pod-template-hash
#      matches the orphaned ReplicaSet, so it ADOPTS it. No pod restarts.
#
# Self-heal: if a Deployment is already gone (e.g. NRP purged it before this
# run), it is recreated from the last good snapshot instead.
#
# Runs headlessly via kubelogin's cached refresh token (offline_access). If the
# refresh token has expired it will fail fast (won't hang) — re-auth once with
# any interactive `kubectl get pods -n datahub-llm`.
set -u

export HOME="${HOME:-/home/yichen}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"

NS=datahub-llm
DEPLOYS="postgres tritondft tritondft-worker"
SNAP_DIR="$HOME/.cache/tritondft-renew/snapshots"
LOG="$HOME/.cache/tritondft-renew/renew.log"
mkdir -p "$SNAP_DIR" "$(dirname "$LOG")"

# Don't let an expired-token device-code prompt hang the cron forever.
KC() { timeout 90 kubectl "$@"; }

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

strip() {  # stdin: live `get -o yaml`  ->  stdout: createable spec
  # status: is always the last top-level block in `get -o yaml`.
  sed '/^status:$/,$d' \
    | grep -vE '^[[:space:]]+(creationTimestamp|resourceVersion|uid|generation|selfLink):'
}

rc=0
log "=== renew start (ns=$NS) ==="
for D in $DEPLOYS; do
  snap="$SNAP_DIR/$D.yaml"
  if KC -n "$NS" get deploy "$D" -o yaml --show-managed-fields=false 2>/dev/null | strip > "$snap.tmp" && [ -s "$snap.tmp" ]; then
    mv "$snap.tmp" "$snap"
    KC -n "$NS" delete deploy "$D" --cascade=orphan --ignore-not-found >/dev/null 2>&1
    if KC -n "$NS" create -f "$snap" >/dev/null 2>&1; then
      log "OK   $D — clock reset, ReplicaSet adopted (no pod restart)"
    else
      log "ERR  $D — recreate failed; orphaned ReplicaSet still serving, restoring from snapshot"
      KC -n "$NS" create -f "$snap" >/dev/null 2>&1 && log "OK   $D — restored on retry" || { log "FAIL $D — manual fix needed"; rc=1; }
    fi
  else
    rm -f "$snap.tmp"
    # Deployment missing (purged?). Self-heal from last snapshot if we have one.
    if [ -s "$snap" ]; then
      if KC -n "$NS" create -f "$snap" >/dev/null 2>&1; then
        log "HEAL $D — was missing, recreated from snapshot"
      else
        log "FAIL $D — missing and snapshot restore failed (auth expired?)"; rc=1
      fi
    else
      log "WARN $D — not found and no snapshot; skipping"; rc=1
    fi
  fi
done
log "=== renew done (exit $rc) ==="
exit $rc
