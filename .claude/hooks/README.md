# Claude Code notification hook

`notify.sh` is registered on three events and branches on `hook_event_name`:

| Event | Fires when | Push looks like |
|---|---|---|
| `Stop` | the turn has ended | "Claude Code done: episteme-ai", normal priority, 🤖✅ |
| `Notification` | Claude wants input *mid-turn* — a permission prompt, or the input box sitting idle | "Claude Code needs you: episteme-ai", **high** priority, 🙋❓ |
| `PreToolUse` | an `AskUserQuestion` / `ExitPlanMode` dialog opens, carrying the actual question | same as above, deduped against the `Notification` that follows |

Caveat: with `defaultMode: bypassPermissions` the permission-prompt trigger almost
never fires, since tools do not stop to ask. The dialog and idle triggers still do.

## Setup

```bash
cp .claude/hooks/notify.env.example .claude/hooks/notify.env
$EDITOR .claude/hooks/notify.env     # set NTFY_TOPIC
```

Then subscribe on your phone with the ntfy app (Add subscription → the same topic
name), or in a browser at `https://ntfy.sh/<topic>`.

`notify.env` is gitignored and is the **only** place the topic lives. Keep it that
way: ntfy's public server has no per-topic auth, so the topic name *is* the
credential — anyone who learns it can read every push (which include the tail of
what Claude last said) and send their own. Pick something unguessable, and rotate
by editing `notify.env`. With no `notify.env` present the hook still chimes and
raises a desktop notification; it just skips the phone push.

`notify.log` keeps a 200-line trail of raw hook payloads — session IDs, transcript
paths, prompt text — so an unexplained push can be traced back. It is gitignored
for that reason. So is the `.notify-dialog-<session>` dedupe marker.

**Sound / desktop:** the sbx sandbox has no `/dev/snd`, no D-Bus and no `$DISPLAY`,
so inside it the only audible signal is a BEL to the controlling terminal. The
`paplay` / `ffplay` / `canberra-gtk-play` / `notify-send` branches take over
automatically if the script is ever run on the host.

## Where the hook is registered

`.claude/settings.json` — deliberately **not** `~/.claude/settings.json`. Only the
project directory is bind-mounted from the host; everything else in the sbx sandbox
(including `~/.claude`) is wiped when the sandbox is recreated. Keeping the
registration here means it survives with no maintenance.

`sbx` does not seed a new sandbox's `~/.claude` from the host copy — the two files
are unrelated — so the host global settings are not a route to persistence either.

Because it lives in the project, this hook also applies when you run Claude Code
**natively on the host** in this directory, where `notify-send` and `paplay` really
exist. The command uses `${CLAUDE_PROJECT_DIR:-.}` so it resolves on both sides.

Do not also add the hook to `~/.claude/settings.json` or to
`.claude/settings.local.json`: hooks from different settings files are merged, not
overridden, so it would fire twice per turn.

Note that a tracked `.claude/settings.json` means anyone who clones this repo and
runs Claude Code here gets the hook proposed to them. Claude Code asks before
running a hook it has not seen, so it cannot fire silently — but it is why nothing
secret belongs in this file.

## Testing without waiting for a turn

```bash
echo '{"cwd":"'"$PWD"'","hook_event_name":"Stop"}' | .claude/hooks/notify.sh
```
