#!/usr/bin/env bash
# PreToolUse guard for Bash: block commands that reference NFS paths
# under /root/notebooks outside this project (LN_Reg2RG).
# Best-effort string match on the command text — not a sandbox guarantee.
# Relative paths (e.g. after `cd ..`) or obfuscated paths are not caught.

cmd=$(jq -r '.tool_input.command // empty')

if [ -z "$cmd" ]; then
  exit 0
fi

if echo "$cmd" | grep -qP '/root/notebooks/(?!automl/LN_Reg2RG\b|automl/\.git-credentials\b|automl/\.git-ssh\b|automl/env\b|groups/BME/reg2rg_nifti\b)'; then
  cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"This command references an NFS path under /root/notebooks outside the LN_Reg2RG project. Blocked by .claude/settings.json policy. If access is genuinely required, ask the user to explicitly update the permission rules first."}}
EOF
fi

exit 0
