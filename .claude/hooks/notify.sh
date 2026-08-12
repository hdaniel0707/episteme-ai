#!/usr/bin/env bash
# Claude Code Stop hook: completion chime + desktop notification + ntfy push to phone.
#
# Reads the hook payload (JSON) on stdin. Every step is best-effort: the script
# always exits 0 so a missing player or a network hiccup never fails the turn.
#
# Configured by notify.env beside this script (gitignored — see notify.env.example),
# overridable by the ambient environment:
#   NTFY_TOPIC   ntfy topic to publish to. No default: the topic name is the only
#                thing gating who can read these pushes, so it must not live in a
#                tracked file. Unset simply disables the phone push.
#   NTFY_SERVER  ntfy server base URL (default https://ntfy.sh)

set -uo pipefail

# Ambient env wins over the file, so a one-off `NTFY_TOPIC=... ./notify.sh` works.
ENV_FILE="$(dirname "$0")/notify.env"
if [ -r "$ENV_FILE" ]; then
    _topic="${NTFY_TOPIC:-}" _server="${NTFY_SERVER:-}"
    # shellcheck disable=SC1090
    . "$ENV_FILE" || true
    [ -n "$_topic" ] && NTFY_TOPIC="$_topic"
    [ -n "$_server" ] && NTFY_SERVER="$_server"
    unset _topic _server
fi

NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
MAX_MSG_CHARS=280

payload="$(cat 2>/dev/null || true)"

field() { printf '%s' "$payload" | jq -r "$1 // empty" 2>/dev/null; }

cwd="$(field '.cwd')"
[ -n "$cwd" ] || cwd="$PWD"
project="$(basename "$cwd")"
transcript="$(field '.transcript_path')"
event="$(field '.hook_event_name')"
# Notification carries a notification_type discriminating ~10 unrelated events.
ntype="$(field '.notification_type')"

# Keep a short trail so an unexplained push can be traced back to its payload.
LOG="$(dirname "$0")/notify.log"
{
    printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$event" "${ntype:--}" \
        "$(printf '%s' "$payload" | jq -c . 2>/dev/null | cut -c1-400)"
    # Must not be the group's last status: a missing log would fail the `&&`.
    [ -r "$LOG" ] && tail -n 199 "$LOG"
    :
} >"$LOG.tmp" 2>/dev/null && mv -f "$LOG.tmp" "$LOG" 2>/dev/null || true

# Tail of what Claude last said, so the phone push is actually informative.
tail_text() {
    [ -n "$transcript" ] && [ -r "$transcript" ] || return 0
    jq -r 'select(.type=="assistant") | .message.content[]?
           | select(.type=="text") | .text' "$transcript" 2>/dev/null |
        grep -v '^[[:space:]]*$' | tail -n 4 | tr '\n\r\t' '   ' |
        cut -c1-"$MAX_MSG_CHARS"
}

# The Notification event is not one event. It multiplexes everything the CLI
# would show in the OS notification centre, tagged by notification_type:
#
#   permission_prompt / input_needed / dialog_open / goal_proposal
#   agent_needs_input / worker_permission_prompt / push_notification
#       -> genuinely blocked on the user. Note that permission_prompt does NOT
#          mean a permission request: it is the CLI's fallback type for any
#          dialog lacking an explicit override (`hXE[kind] ?? "permission
#          prompt"`), so AskUserQuestion arrives here too, carrying the generic
#          message "Claude needs your permission". Deduped below.
#   idle_prompt
#       -> the input box sat untouched for messageIdleNotifThresholdMs (60s).
#          Nothing is blocked; it fires *after* Stop already pushed "done", so
#          it is a duplicate that reads like a permission request. Drop it.
#   auth_success / elicitation_* / computer_use_*
#       -> informational, no action pending. Drop.
case "$ntype" in
idle_prompt | auth_success | elicitation_* | computer_use_*)
    exit 0
    ;;
esac

# Dedupe the AskUserQuestion / ExitPlanMode double-push. PreToolUse fires first
# and carries the actual question, then the dialog's own permission_prompt
# arrives seconds later with generic wording. Keep the informative one: mark the
# session on PreToolUse, and suppress a permission_prompt that lands right after.
session="$(field '.session_id')"
MARK="$(dirname "$0")/.notify-dialog-${session:-none}"
if [ "$event" = "Notification" ] && [ "$ntype" = "permission_prompt" ]; then
    if [ -f "$MARK" ]; then
        now=$(date +%s)
        then=$(stat -c %Y "$MARK" 2>/dev/null || echo 0)
        [ $((now - then)) -lt 60 ] && exit 0
    fi
elif [ "$event" = "PreToolUse" ]; then
    : >"$MARK" 2>/dev/null || true
fi

# Two flavours: "Claude wants something from you" vs "Claude is done".
#
#   Notification  - a blocking dialog, per the filter above.
#   PreToolUse    - matched to the blocking dialog tools (AskUserQuestion,
#                   ExitPlanMode). Fires the instant the dialog appears, which
#                   Notification does not reliably do: an AskUserQuestion answered
#                   inside the idle window produced no Notification event at all.
#   Stop          - the turn has actually ended.
if [ "$event" = "Notification" ] && [ "$ntype" = "agent_completed" ]; then
    summary="$(field '.message')"
    [ -n "$summary" ] || summary="A subagent finished in $project."
    label="done"
    tags="robot,white_check_mark"
    priority="default"
elif [ "$event" = "Notification" ] || [ "$event" = "PreToolUse" ]; then
    summary="$(field '.message')"
    # AskUserQuestion: push the actual question. ExitPlanMode: flag the approval.
    [ -n "$summary" ] || summary="$(field '.tool_input.questions[0].question')"
    if [ -z "$summary" ] && [ -n "$(field '.tool_input.plan')" ]; then
        summary="Claude wants approval for a plan."
    fi
    [ -n "$summary" ] || summary="Claude Code is waiting for you in $project."
    label="needs you"
    tags="raising_hand,question"
    priority="high"
else
    summary="$(tail_text)"
    [ -n "$summary" ] || summary="Claude Code finished in $project."
    label="done"
    tags="robot,white_check_mark"
    priority="default"
fi

# ntfy headers must stay single-line; the body may be UTF-8.
title="$(printf 'Claude Code %s: %s' "$label" "$project" | tr -d '[:cntrl:]' | cut -c1-80)"

# --- 1a. Completion sound ---------------------------------------------------
# Inside the sbx sandbox there is no /dev/snd, no D-Bus and no $DISPLAY, so the
# only thing that can actually make noise is a BEL written to the controlling
# terminal — the host terminal emulator rings it. The player branches below are
# for when this script runs on the host directly.
chime() {
    local wav
    for wav in /usr/share/sounds/freedesktop/stereo/complete.oga \
        /usr/share/sounds/freedesktop/stereo/message.oga; do
        [ -r "$wav" ] || continue
        if command -v paplay >/dev/null 2>&1; then
            paplay "$wav" 2>/dev/null && return 0
        elif command -v ffplay >/dev/null 2>&1; then
            ffplay -nodisp -autoexit -loglevel quiet "$wav" 2>/dev/null && return 0
        fi
    done
    if command -v canberra-gtk-play >/dev/null 2>&1; then
        canberra-gtk-play -i complete 2>/dev/null && return 0
    fi
    # Keep the BEL off stdout: hook stdout is parsed by Claude Code. The braces
    # put the redirect itself inside the silenced scope — with `printf ... 2>/dev/null`
    # the shell still reports a missing /dev/tty on stderr.
    { printf '\a' >/dev/tty; } 2>/dev/null || true
}
chime

# --- 1b. Desktop notification ----------------------------------------------
if command -v notify-send >/dev/null 2>&1; then
    urgency=normal
    icon=dialog-information
    if [ "$label" = "needs you" ]; then
        urgency=critical # stays on screen until dismissed
        icon=dialog-question
    fi
    notify-send -a "Claude Code" -u "$urgency" -i "$icon" \
        "$title" "$summary" 2>/dev/null || true
fi

# --- 2. Push to phone via ntfy ---------------------------------------------
# No topic configured (fresh clone with no notify.env): the local signals above
# still fired, so just skip the push rather than POSTing to $NTFY_SERVER/.
[ -n "$NTFY_TOPIC" ] || exit 0

curl -fsS -m 10 \
    -H "Title: $title" \
    -H "Tags: $tags" \
    -H "Priority: $priority" \
    -d "$summary" \
    "$NTFY_SERVER/$NTFY_TOPIC" >/dev/null 2>&1 || true

exit 0
