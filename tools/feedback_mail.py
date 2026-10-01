"""Feedback delivery: take what the feedback form collected and send it by email.

WHY THIS EXISTS
---------------
The feedback button saved an entry to `localStorage` and to `feedback.json` beside
the server, then said "Feedback saved". Nothing left the machine. On a shared
deployment that means the form is a local notepad: the person running the pipeline
never learns what the person using it thought, and the person using it believes
they have sent something. Both halves of that are wrong, and the second is worse.

WHAT IT NEEDS
-------------
An SMTP account, configured in `.env.local` (the same file `start.bat` already
reads). Nothing is hard-coded here: a credential in the repository is a credential
leaked, and this file is public.

    TF_FEEDBACK_MAIL_HOST=smtp.163.com
    TF_FEEDBACK_MAIL_PORT=465
    TF_FEEDBACK_MAIL_USER=<the sending account>
    TF_FEEDBACK_MAIL_PASS=<its SMTP authorisation code>
    TF_FEEDBACK_MAIL_TO=18806370529@163.com

163.com issues a *client authorisation code* for SMTP, separate from the account
password; the account password is rejected. Set `TF_FEEDBACK_MAIL_PASS` to that
code.

`TF_FEEDBACK_MAIL_MODE` is one of:
    auto   (default) TLS on 465, STARTTLS on 587/25
    ssl    force implicit TLS
    starttls force STARTTLS
    plain  no encryption — only for a relay on localhost

WHEN IT IS NOT CONFIGURED
-------------------------
`configured()` is False and `send()` returns `{"sent": False, "reason": ...}`. It
does not raise and it does not pretend. The caller is expected to tell the reader
which of the two happened, because "saved on this machine" and "sent to the
maintainer" are different promises.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# The destination the form's feedback is addressed to by default. Overridable with
# TF_FEEDBACK_MAIL_TO; present as a constant so a deployment that sets nothing but
# credentials still delivers somewhere the maintainer reads.
DEFAULT_TO = "18806370529@163.com"
DEFAULT_HOST = "smtp.163.com"
DEFAULT_PORT = 465

# Ports where an implicit-TLS handshake is the expected default.
_TLS_PORTS = {465, 994, 995, 993}


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return ""


def load_env_file(path=None) -> int:
    """Load `.env.local` into the environment. Returns how many keys were set.

    Its own reader rather than an import from `tools/_env.py`: this module is
    imported by the server, which is started from several working directories, and
    a relative import of a tools file is the kind of thing that works on the
    developer's machine and nowhere else.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / ".env.local"
    path = Path(path)
    if not path.is_file():
        return 0
    applied = 0
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue
        os.environ[key] = val
        applied += 1
    return applied


def config() -> Dict:
    """The resolved mail settings. Never raises."""
    load_env_file()
    host = _env("TF_FEEDBACK_MAIL_HOST") or DEFAULT_HOST
    try:
        port = int(_env("TF_FEEDBACK_MAIL_PORT") or DEFAULT_PORT)
    except ValueError:
        port = DEFAULT_PORT
    mode = (_env("TF_FEEDBACK_MAIL_MODE") or "auto").lower()
    if mode == "auto":
        mode = "ssl" if port in _TLS_PORTS else "starttls"
    return {
        "host": host,
        "port": port,
        "user": _env("TF_FEEDBACK_MAIL_USER"),
        "password": _env("TF_FEEDBACK_MAIL_PASS"),
        "to": _env("TF_FEEDBACK_MAIL_TO") or DEFAULT_TO,
        "sender": _env("TF_FEEDBACK_MAIL_FROM"),
        "mode": mode,
        "timeout": float(_env("TF_FEEDBACK_MAIL_TIMEOUT") or 20),
    }


def configured() -> bool:
    """Can this send? Credentials and a destination, nothing more."""
    c = config()
    return bool(c["host"] and c["user"] and c["password"] and c["to"])


def missing_settings() -> List[str]:
    """Which of the required keys are absent, for a message that can be acted on."""
    c = config()
    absent = []
    if not c["user"]:
        absent.append("TF_FEEDBACK_MAIL_USER")
    if not c["password"]:
        absent.append("TF_FEEDBACK_MAIL_PASS")
    if not c["to"]:
        absent.append("TF_FEEDBACK_MAIL_TO")
    return absent


# ── rendering ────────────────────────────────────────────────────────────────

def _stars(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    n = max(0, min(5, n))
    return "*" * n + "." * (5 - n)


def subject_for(entry: Dict) -> str:
    """One line that is useful in an inbox list, from what the form collected."""
    rating = entry.get("rating")
    tags = entry.get("tags") or []
    bits = []
    if rating:
        bits.append("%d/5" % int(rating))
    if tags:
        bits.append(", ".join(str(t) for t in tags[:3]))
    n = entry.get("sequenceLength")
    if n:
        bits.append("%s nt" % n)
    return "[TorusFold feedback] " + (" · ".join(bits) if bits else "no rating")


def render(entry: Dict) -> str:
    """The feedback as plain text: what was said, then what it was said about."""
    lines = []
    lines.append("TorusFold feedback")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Rating      : %s  (%s/5)" % (_stars(entry.get("rating")),
                                               entry.get("rating")))
    tags = entry.get("tags") or []
    lines.append("Tags        : %s" % (", ".join(str(t) for t in tags) if tags else "(none)"))
    lines.append("Submitted   : %s" % entry.get("timestamp", "(unknown)"))
    lines.append("Job         : %s" % (entry.get("jobId") or "(none)"))
    lines.append("Seq length  : %s" % (entry.get("sequenceLength") or "(unknown)"))
    lines.append("")
    lines.append("Comments")
    lines.append("-" * 60)
    comments = (entry.get("comments") or "").strip()
    lines.append(comments if comments else "(none written)")
    lines.append("")

    summary = entry.get("result_summary") or {}
    if summary:
        lines.append("Result at the time of writing")
        lines.append("-" * 60)
        for k, v in summary.items():
            lines.append("%-22s %s" % (k + ":", v))
        lines.append("")

    # Provenance: which build and which machine, so a report can be placed without
    # a round trip. Read from what the browser sent, never invented here.
    ctx = entry.get("context") or {}
    if ctx:
        lines.append("Context")
        lines.append("-" * 60)
        for k in sorted(ctx):
            lines.append("%-22s %s" % (k + ":", ctx[k]))
        lines.append("")

    lines.append("-- ")
    lines.append("Sent by the TorusFold-Hybrid feedback form. Reply to this message "
                 "to reach the reporter if they left an address.")
    return "\n".join(lines)


# ── sending ──────────────────────────────────────────────────────────────────

def send(entry: Dict, to: Optional[str] = None) -> Dict:
    """Email one feedback entry. Returns a dict; never raises.

    Returns `{"sent": True, ...}` only when the server accepted the message. Any
    other outcome carries `"sent": False` and a `reason` the interface can show.
    """
    c = config()
    dest = (to or c["to"]).strip()
    if not configured():
        return {"sent": False,
                "reason": "email is not configured",
                "missing": missing_settings(),
                "hint": "Set TF_FEEDBACK_MAIL_USER and TF_FEEDBACK_MAIL_PASS in "
                        ".env.local (and TF_FEEDBACK_MAIL_TO if the default is not "
                        "the address you want)."}
    sender = c["sender"] or c["user"]

    msg = EmailMessage()
    msg["Subject"] = subject_for(entry)
    msg["From"] = sender
    msg["To"] = dest
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="torusfold")
    # The reporter's own address is not collected by the form, so there is no
    # Reply-To to set. The body says so rather than the header implying one.
    msg.set_content(render(entry))

    try:
        ctx = ssl.create_default_context()
        if c["mode"] == "ssl":
            with smtplib.SMTP_SSL(c["host"], c["port"], timeout=c["timeout"],
                                  context=ctx) as s:
                s.login(c["user"], c["password"])
                s.send_message(msg)
        elif c["mode"] == "plain":
            with smtplib.SMTP(c["host"], c["port"], timeout=c["timeout"]) as s:
                s.login(c["user"], c["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(c["host"], c["port"], timeout=c["timeout"]) as s:
                s.ehlo()
                s.starttls(context=ctx)
                s.ehlo()
                s.login(c["user"], c["password"])
                s.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        # The commonest mistake with 163.com is using the account password. Say so
        # here, where the failure is known, rather than leaving the reader to find
        # it in a manual.
        return {"sent": False, "reason": "the mail server rejected the credentials",
                "detail": str(e),
                "hint": "163.com requires a client authorisation code for SMTP, not "
                        "the account password. Set TF_FEEDBACK_MAIL_PASS to that code."}
    except (smtplib.SMTPException, OSError, ssl.SSLError) as e:
        return {"sent": False, "reason": "could not reach the mail server",
                "detail": "%s: %s" % (type(e).__name__, e),
                "hint": "Check TF_FEEDBACK_MAIL_HOST/PORT/MODE. Port 465 needs "
                        "implicit TLS, which `mode=auto` selects for it."}

    return {"sent": True, "to": dest, "via": "%s:%s (%s)" % (c["host"], c["port"],
                                                             c["mode"])}


def status_line() -> Tuple[bool, str]:
    """(configured, one line describing it) — for a startup report."""
    c = config()
    if configured():
        return True, "feedback email: %s:%s -> %s (%s)" % (
            c["host"], c["port"], c["to"], c["mode"])
    return False, ("feedback email: not configured (missing %s); feedback is saved "
                   "locally only" % ", ".join(missing_settings() or ["settings"]))
