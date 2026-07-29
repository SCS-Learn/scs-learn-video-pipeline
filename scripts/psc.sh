#!/usr/bin/env bash
# PSC (Bridges-2) access helper.
#
# No credential is stored anywhere by design. `psc.sh login` opens ONE
# interactive, MFA-authenticated master connection; every later command reuses
# that socket via SSH connection multiplexing and needs no authentication. When
# the socket expires (ControlPersist, default 8h) access ends on its own.
#
# That is why there is nothing for this to read out of 1Password: the only
# secret involved is your PSC password, which you type once, into ssh, yourself.
#
# Every non-interactive command runs with BatchMode=yes so that a dead socket
# fails immediately instead of silently blocking on a password prompt -- which
# matters when an agent is driving this.
#
# ---------------------------------------------------------------------------
# LOGIN NODE POLICY -- read before adding commands here.
#
#   "All production computing must be done on Bridges-2's compute nodes, NOT on
#    Bridges-2's login nodes."
#   "You cannot use Bridges-2's login nodes for your work."
#   "File transfers can no longer be initiated from the Bridges-2 login nodes."
#                                        -- PSC Bridges-2 User Guide
#
# So: `run` is for LIGHT MANAGEMENT ONLY -- ls, squeue, sinfo, module avail,
# pip list, cat, sacct. Never import torch/whisperx/insightface, never run
# ffmpeg, never run a broad `find`, never transcode or infer. For anything real
# use `sbatch` (batch) or `interact` (interactive compute node). For moving
# files use `sync`, which goes over the DTN, never a login node.
# ---------------------------------------------------------------------------
#
#   ./scripts/psc.sh install-config   # write the ~/.ssh/config stanzas (once)
#   ./scripts/psc.sh probe            # reachability, no auth needed
#   ./scripts/psc.sh login            # interactive: password, once
#   ssh -N psc-dtn                    # transfers: separate connection, no shell
#   ./scripts/psc.sh status           # is the shared session alive?
#   ./scripts/psc.sh run 'squeue -u $USER'      # LIGHT commands only
#   ./scripts/psc.sh sync             # rsync code up VIA THE DTN
#   ./scripts/psc.sh sbatch scripts/psc_face_anon.sbatch
#   ./scripts/psc.sh interact         # get a real compute node to work on
#   ./scripts/psc.sh logout           # close the shared session
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Read ONLY PSC_* keys out of .env, so unrelated secrets in that file never
# enter this process's environment.
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r k v; do
        v="${v%\"}"; v="${v#\"}"
        export "$k=$v"
    done < <(grep -E '^PSC_[A-Z0-9_]+=' "$ENV_FILE" 2>/dev/null || true)
fi

PSC_HOST="${PSC_HOST:-bridges2.psc.edu}"
PSC_USER="${PSC_USER:-}"
PSC_ALIAS="${PSC_SSH_ALIAS:-psc}"
PSC_DTN_HOST="${PSC_DTN_HOST:-data.bridges2.psc.edu}"
PSC_DTN_ALIAS="${PSC_ALIAS}-dtn"
PSC_REMOTE_REPO="${PSC_REMOTE_REPO:-}"
PSC_CONDA_ENV="${PSC_CONDA_ENV:-scs-learn}"
SSH_CONFIG="$HOME/.ssh/config"

die() { echo "psc.sh: $*" >&2; exit 1; }

require_user() {
    [ -n "$PSC_USER" ] || die "PSC_USER is not set. Add it to $ENV_FILE"
}

# ssh with the multiplexed socket, never prompting.
sshq() { ssh -o BatchMode=yes "$PSC_ALIAS" "$@"; }

cmd_install_config() {
    require_user
    mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
    touch "$SSH_CONFIG"; chmod 600 "$SSH_CONFIG"
    if grep -qE "^[[:space:]]*Host[[:space:]]+.*\b${PSC_ALIAS}\b" "$SSH_CONFIG" 2>/dev/null; then
        echo "A 'Host $PSC_ALIAS' stanza already exists in $SSH_CONFIG - leaving it alone."
        echo "Current stanza:"
        awk -v a="$PSC_ALIAS" '
            $1=="Host" { inb=0; for(i=2;i<=NF;i++) if($i==a) inb=1 }
            inb { print "    " $0 }' "$SSH_CONFIG"
        return 0
    fi
    cat >> "$SSH_CONFIG" <<EOF

# --- scs-learn-video-pipeline: PSC Bridges-2 (added by scripts/psc.sh) ---
# ControlMaster lets one interactive MFA login be reused by later
# non-interactive commands, so no credential has to be stored.
Host $PSC_ALIAS
    HostName $PSC_HOST
    User $PSC_USER
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
EOF
    cat >> "$SSH_CONFIG" <<EOF

# Data Transfer Node. PSC forbids initiating file transfers from the login
# nodes, so scripts/psc.sh sync uses this host instead.
Host $PSC_DTN_ALIAS
    HostName $PSC_DTN_HOST
    User $PSC_USER
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 8h
    ServerAliveInterval 60
EOF
    echo "Appended 'Host $PSC_ALIAS' and 'Host $PSC_DTN_ALIAS' to $SSH_CONFIG:"
    tail -25 "$SSH_CONFIG"
}

cmd_probe() {
    echo "=== resolve $PSC_HOST ==="
    if command -v dig >/dev/null 2>&1; then
        dig +short "$PSC_HOST" | head -5
    else
        python3 -c "import socket,sys; print('\n'.join(sorted({r[4][0] for r in socket.getaddrinfo(sys.argv[1],22)})))" "$PSC_HOST"
    fi
    echo
    echo "=== TCP :22 reachable? ==="
    if command -v nc >/dev/null 2>&1; then
        nc -z -G 8 "$PSC_HOST" 22 && echo "port 22 OPEN" || echo "port 22 unreachable"
    fi
    echo
    echo "=== SSH service banner / host keys (no auth) ==="
    ssh-keyscan -T 8 -t rsa,ed25519 "$PSC_HOST" 2>/dev/null \
        | sed 's/^/  /' || echo "  ssh-keyscan failed"
    echo
    echo "=== which auth methods does the server offer? ==="
    # Deliberately fails; the interest is the 'continue with' line it prints.
    ssh -o BatchMode=yes -o PreferredAuthentications=none \
        -o StrictHostKeyChecking=accept-new \
        "${PSC_USER:-nobody}@$PSC_HOST" true 2>&1 | grep -iE 'continue with|permission denied|authentication' | sed 's/^/  /'
    echo
    echo "  note: 'publickey' in that list means key-based auth is available at"
    echo "  the SSH layer. No 'keyboard-interactive' is offered, which is the"
    echo "  usual carrier for Duo -- so Duo likely rides on 'password'. If so, a"
    echo "  registered key would authenticate with no MFA prompt at all."
    return 0        # the probe above fails on purpose; that is not our failure
}

cmd_login() {
    require_user
    grep -qE "^[[:space:]]*Host[[:space:]]+.*\b${PSC_ALIAS}\b" "$SSH_CONFIG" 2>/dev/null \
        || die "no 'Host $PSC_ALIAS' in $SSH_CONFIG - run: $0 install-config"
    if ssh -O check "$PSC_ALIAS" >/dev/null 2>&1; then
        echo "Session already live. Nothing to do (see: $0 status)."
        return 0
    fi

    # ssh reads a password from the controlling terminal, so this needs a real
    # TTY. Do NOT use `ssh -M -N -f` here: -f implies -n, which "prevents
    # reading from stdin", so ssh can never ask for the password and simply
    # burns its three attempts and exits -- which looks exactly like a wrong
    # password, with no prompt line in the output to give the game away.
    if [ ! -t 0 ] || [ ! -t 1 ]; then
        cat >&2 <<EOF
psc.sh: no TTY available, so ssh cannot prompt for your PSC password here.

Run this in a real terminal (Terminal.app / iTerm), not through an agent or a
pipe:

    cd "$REPO_ROOT" && ssh $PSC_ALIAS

Log in, then type 'exit'. ControlMaster+ControlPersist keep the shared session
alive ~8h after you exit, so '$0 status', 'run' and 'sync' will then work.
EOF
        exit 1
    fi

    echo "Opening the shared master session to $PSC_USER@$PSC_HOST."
    echo "Use your PSC *Kerberos* password (reset at https://apr.psc.edu) --"
    echo "it is not your CMU Andrew/SSO password."
    echo "Log in, then type 'exit'; the session persists ~8h after you leave."
    echo
    # Foreground + interactive: ControlMaster auto promotes this to the master,
    # and ControlPersist keeps the socket after this client exits.
    ssh "$PSC_ALIAS"
    echo
    if ssh -O check "$PSC_ALIAS" >/dev/null 2>&1; then
        echo "Session is live. It persists ~8h idle; close early with: $0 logout"
    else
        die "no shared session afterwards - login did not succeed"
    fi
}

cmd_status() {
    if ssh -O check "$PSC_ALIAS" 2>/dev/null; then
        echo "-> shared session LIVE"
        echo -n "-> remote whoami/host: "
        sshq 'echo "$(whoami)@$(hostname)"' 2>/dev/null || echo "(command failed)"
    else
        echo "-> no shared session. Run: $0 login"
        return 1
    fi
}

cmd_run() {
    [ $# -gt 0 ] || die "usage: $0 run '<remote command>'"
    sshq "$@"
}

cmd_sync() {
    require_user
    [ -n "$PSC_REMOTE_REPO" ] || die "PSC_REMOTE_REPO is not set in $ENV_FILE"
    # PSC: "File transfers can no longer be initiated from the Bridges-2 login
    # nodes." Everything here goes over the Data Transfer Node instead.
    if ! ssh -O check "$PSC_DTN_ALIAS" >/dev/null 2>&1; then
        die "no session to the DTN ($PSC_DTN_HOST).
Transfers must not go through a login node, so the DTN needs its own connection.
Run this once in a real terminal:

    ssh -N $PSC_DTN_ALIAS

Enter your password; it will then sit there with no prompt -- that is correct,
-N requests no remote command. Background it (ctrl-Z then bg) or open another
tab, and leave it up. The socket persists ~8h.

Do NOT expect a shell: the DTN refuses one with 'Login denied: Only file
transfers are allowed on this account'. Authentication still succeeds and the
master socket is still created, so scp/sftp/rsync work over it -- but an
interactive login will always look like it failed."
    fi
    echo "rsync -> $PSC_DTN_ALIAS:$PSC_REMOTE_REPO  (via DTN, not a login node)"
    # macOS ships openrsync / rsync 2.6.9, which has no --info=; -v --stats works
    # on both that and modern rsync 3.x.
    local vflag="-v --stats"
    rsync --info=stats1 --version >/dev/null 2>&1 && vflag="--info=stats1,progress2"
    # Code and scripts only. Never the lecture media (hundreds of MB), never
    # .env, never the venv.
    rsync -az $vflag \
        -e "ssh -o BatchMode=yes" \
        --exclude '.git/' --exclude '.venv/' --exclude 'data/' \
        --exclude '.env' --exclude '__pycache__/' --exclude '*.pyc' \
        --exclude '.DS_Store' --exclude 'manifest*.json' \
        "$REPO_ROOT/" "$PSC_DTN_ALIAS:$PSC_REMOTE_REPO/"
}

cmd_sbatch() {
    [ $# -gt 0 ] || die "usage: $0 sbatch <script.sbatch> [sbatch args...]"
    local script="$1"; shift
    [ -f "$REPO_ROOT/$script" ] || [ -f "$script" ] || die "no such script: $script"
    echo "Submitting $script (runs on compute nodes, not a login node)"
    sshq "cd '$PSC_REMOTE_REPO' && mkdir -p logs && sbatch $* '$script'"
}

cmd_interact() {
    # The only sanctioned way to work interactively: SLURM hands you a compute
    # node. Needs a TTY, so this must be run from a real terminal.
    if [ ! -t 0 ] || [ ! -t 1 ]; then
        die "interact needs a TTY. Run in a real terminal:
    ssh $PSC_ALIAS -t 'interact -p RM-shared -n 8 -t 1:00:00'
or for a GPU (NOT v100 for face_anon - sm_70 unsupported):
    ssh $PSC_ALIAS -t 'interact -p GPU-shared --gpus=l40s-48:1 -t 1:00:00'"
    fi
    local part="${1:-RM-shared}"; shift || true
    echo "Requesting an interactive compute node on $part ..."
    ssh "$PSC_ALIAS" -t "interact -p $part $*"
}

cmd_info() {
    echo "PSC_HOST         = $PSC_HOST"
    echo "PSC_USER         = ${PSC_USER:-<unset>}"
    echo "PSC_SSH_ALIAS    = $PSC_ALIAS"
    echo "PSC_REMOTE_REPO  = ${PSC_REMOTE_REPO:-<unset>}"
    echo "PSC_CONDA_ENV    = $PSC_CONDA_ENV"
    echo "PSC_DTN_HOST     = $PSC_DTN_HOST (transfers; login nodes forbidden)"
echo "ssh config       = $SSH_CONFIG"
    echo "control socket   = $(ls ~/.ssh/cm-* 2>/dev/null | tr '\n' ' ' || echo none)"
}

cmd_logout() { ssh -O exit "$PSC_ALIAS" 2>&1 || echo "no session to close"; }

# Convenience wrappers, all over the shared socket.
cmd_sq()    { sshq 'squeue -u $USER -o "%.10i %.12P %.14j %.8T %.10M %.6D %R"'; }
cmd_quota() { sshq 'projects 2>/dev/null || echo "(projects command unavailable)"'; }

case "${1:-}" in
    install-config) shift; cmd_install_config "$@";;
    probe)          shift; cmd_probe "$@";;
    login)          shift; cmd_login "$@";;
    status)         shift; cmd_status "$@";;
    run)            shift; cmd_run "$@";;
    sync)           shift; cmd_sync "$@";;
    sbatch)         shift; cmd_sbatch "$@";;
    interact)       shift; cmd_interact "$@";;
    logout)         shift; cmd_logout "$@";;
    info)           shift; cmd_info "$@";;
    sq)             shift; cmd_sq "$@";;
    quota)          shift; cmd_quota "$@";;
    *) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1;;
esac
