"""wsl.exe writes its own diagnostics as UTF-16LE on top of the command's UTF-8.

MEASURED, because none of this is guessable. On this machine:

    $ python -c "import subprocess; print(subprocess.run(
          ['wsl.exe','-e','bash','-lc','ls -d /home/ocr/tools/rsRNASP1'],
          capture_output=True).stdout)"
    b'/home/ocr/tools/rsRNASP1\\nw\\x00s\\x00l\\x00:\\x00 \\x00<hKm0R \\x00l\\x00o\\x00...'

The second half is

    wsl: 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。

and it is UTF-16LE. Two failures came out of that, and this file pins both:

1. `ValueError: embedded null character`. Decoding the whole buffer as UTF-8 does NOT
   raise -- 0x00 is valid UTF-8 -- so NULs came through as NULs. `_run_bash` output is
   used to build WSL paths, and a path containing a NUL makes CreateProcess raise. The
   rsRNASP1 row read "check failed: embedded null character" for a probe that had been
   passing. So: no NUL may survive, ever.

2. The banner merging into the value. Deleting NULs is not sufficient -- the decoded
   banner carries newlines of its own, so a line filter leaves fragments behind and the
   report said `rsRNASP1 not found at N/ec localhost ...`. So the cut is at the
   banner's own signature rather than at a line boundary.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from torusfold.scheme2 import rsrnasp_quality as rs  # noqa: E402
import _pyrosetta as pj  # noqa: E402

# The literal bytes wsl.exe appends here, rebuilt rather than pasted so the test says
# what it is testing.
BANNER_TEXT = ("wsl: 检测到 localhost 代理配置，但未镜像到 WSL。"
               "NAT 模式下的 WSL 不支持 localhost 代理。\n")
BANNER = BANNER_TEXT.encode("utf-16-le")

PAYLOAD = "/home/ocr/tools/rsRNASP1\n"


def test_the_banner_is_really_utf16le():
    """If this ever stops being true, the rest of the file is testing nothing."""
    assert BANNER[:10] == b"w\x00s\x00l\x00:\x00 \x00"
    assert b"\x00" in BANNER


def test_no_nul_survives_the_decode():
    """The property the CreateProcess crash was about."""
    assert "\x00" not in rs._decode_wsl_output(PAYLOAD.encode("utf-8") + BANNER)
    # CRLF before the banner, which is what a redirected console stream can produce.
    assert "\x00" not in rs._decode_wsl_output(b"/home/ocr/tools/rsRNASP1\r\n" + BANNER)


def test_the_value_is_clean_when_the_banner_follows():
    out = rs._decode_wsl_output(PAYLOAD.encode("utf-8") + BANNER)
    assert out == "/home/ocr/tools/rsRNASP1"


def test_the_banner_alone_decodes_to_nothing():
    assert rs._decode_wsl_output(BANNER).strip() == ""


def test_plain_utf8_passes_through_unchanged():
    """The fix must not damage ordinary output, including non-ASCII."""
    for text in ("Ubuntu-24.04\n", "Python 3.12.3\n", "café\n", "中文路径\n",
                 "-3146.58\n"):
        assert rs._decode_wsl_output(text.encode("utf-8")) == text.rstrip("\n") or \
               rs._decode_wsl_output(text.encode("utf-8")) == text


def test_multiple_lines_are_kept():
    out = rs._decode_wsl_output(b"line one\nline two\n")
    assert out.splitlines() == ["line one", "line two"]


def test_invalid_bytes_do_not_raise():
    """Never raise here: a decode error inside a probe reads as 'tool not installed'."""
    out = rs._decode_wsl_output(b"prefix\xff\xfe\x9csuffix")
    assert isinstance(out, str)


def test_an_empty_buffer_is_an_empty_string():
    assert rs._decode_wsl_output(b"") == ""


def test_the_tools_probe_uses_the_same_decoder():
    """`tools/_pyrosetta.py` runs wsl.exe too and hit the same banner."""
    assert pj._decode_wsl_output(PAYLOAD.encode("utf-8") + BANNER) == \
        "/home/ocr/tools/rsRNASP1"
    assert "\x00" not in pj._decode_wsl_output(PAYLOAD.encode("utf-8") + BANNER)


def _wsl_available() -> bool:
    try:
        return subprocess.run(["wsl.exe", "--status"], capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _wsl_available(), reason="no wsl.exe on this machine")
def test_a_real_wsl_call_now_returns_a_usable_path():
    """The end-to-end regression: this used to come back with the banner glued on, and
    the probe built from it raised ValueError inside a later wsl.exe call."""
    ok, out = rs._run_bash("ls -d /home/ocr/tools/rsRNASP1", timeout=60)
    assert "\x00" not in out
    assert "wsl:" not in out
    if ok:
        assert out.splitlines()[-1].strip() == "/home/ocr/tools/rsRNASP1"


@pytest.mark.skipif(not _wsl_available(), reason="no wsl.exe on this machine")
def test_the_probe_runs_without_raising():
    """Whatever the answer is, it must be a dict and not an exception."""
    rs._probe_cache = None
    probe = rs._probe()
    rs._probe_cache = None
    assert isinstance(probe, dict)
    assert "available" in probe
    assert "\x00" not in str(probe)
    assert "wsl:" not in str(probe)
