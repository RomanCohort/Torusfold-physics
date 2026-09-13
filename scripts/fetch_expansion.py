"""Download the representative structures and convert them into the format the pipeline reads.

Format constraints, read off the existing database rather than guessed:

  * _cgdata/rsRNASP/Training_set/*.pdb is legacy fixed-column PDB. The reader in
    scripts/cg_occupancy_reference_scheme.py slices line[12:16], line[17:20], line[21],
    line[22:27], line[30:38] -- i.e. strict columns, not whitespace splitting. So the output
    has to be written to spec, not merely be parseable.
  * Residue names are the bare one-letter codes (A U G C), matching those files.

RCSB serves mmCIF; legacy .pdb is unavailable for the newer/larger entries (7QVP, 9TZE, 4V4N
all 404 on the .pdb endpoint). gemmi (0.7.5, in the comfyui env) does the conversion.

Downloads are grouped by entry: several representatives can name entities of the same entry,
and that entry should only cross the wire once. The mmCIF is deleted immediately after the
needed entity is extracted -- it is 12-47 MB for a ribosome and is not kept.

Two traps, both found by checking the output rather than trusting it, and both silent:

  * gemmi's Table hands back the RAW text of an atom_site loop, not the interpreted value, so
    label_atom_id arrives as '"O5\\''" with the CIF quoting still attached. Every sugar atom
    (O5', C5', C4', O4', C3', O3', C2', O2', C1') contains a single quote and so is quoted, so
    all of them would have shifted one column and stopped matching -- the CG model would have
    silently lost its C4' bead, which is two of its three interactions. See unq().
  * An entity is routinely deposited as several copies in the asymmetric unit (4WRA_26 is
    chains Z and BC). Taking them all multiplies the residue count and overflows the legacy
    five-column serial field, desynchronising the strict reader. Only the largest chain is kept.

Provenance of the manifest this consumes, so the set can be reproduced rather than trusted:

  1. enumerate every RNA polymer entity from the RCSB search API and pull its sequence;
  2. exact-sequence dedup, then greedy clustering at ~90% identity (Mash-style k-mer distance,
     which is an approximation to percent identity and saturates below ~85% -- it cannot
     resolve fold families, only near-duplicates);
  3. one representative per cluster, longest first, mapped back to a PDB entity id.

Steps 1-3 ran outside this repository (scratch scripts) and their intermediate files are not
kept; the counts are recorded in the commit message. What this script guarantees is the last
step: entity id in, pipeline-format PDB out.

Run in an env with gemmi (this machine: comfyui; the CPU env does not have it):
    python scripts/fetch_expansion.py --manifest manifest.json --pools lsu large mid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import gemmi

CIF_URL = "https://files.rcsb.org/download/{}.cif"
REPO = Path(__file__).resolve().parent.parent
OUT_DEFAULT = REPO / "_cgdata" / "expansion"
TMP = Path(os.environ.get("TORUSFOLD_CIFCACHE") or Path(tempfile.gettempdir()) / "_cifcache")

COMP_MAP = {"RA": "A", "RU": "U", "RG": "G", "RC": "C", "T": "U", "DU": "U", "DI": "I"}


def unq(s) -> str:
    """Strip CIF quoting from one value.

    gemmi's Table hands back the raw text of the loop, not the interpreted value, so
    label_atom_id comes through as '"O5\\''" -- with the quotes that the CIF writer added
    because the name contains a single quote. Every sugar atom (O5', C5', C4', O4', C3',
    O3', C2', O2', C1') is affected, so leaving them in shifts the atom field one column and
    the strict reader sees the wrong atom name.
    """
    s = str(s)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def download(entry: str, dest: Path) -> bool:
    """Fetch one mmCIF. Returns False on 404 (no such deposition)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    req = urllib.request.Request(CIF_URL.format(entry),
                                headers={"User-Agent": "torusfold-expansion"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise
    return False


def extract_entity(cif_path: Path, entity_num: str) -> list:
    """Pull the ATOM rows of one entity out of an mmCIF. Returns list of (atom, comp, seq, x, y, z, asym)."""
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()
    try:
        tab = block.find_mmcif_category("_atom_site.")
    except Exception:
        return []
    tags = list(tab.tags)
    if not tags:
        return []
    idx = {}
    for want in ("label_entity_id", "label_comp_id", "label_atom_id",
                 "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "label_asym_id"):
        for full in (f"_atom_site.{want}", want):
            if full in tags:
                idx[want] = tags.index(full)
                break
    if "label_entity_id" not in idx:
        return []
    out = []
    for row in tab:
        if str(row[idx["label_entity_id"]]) != entity_num:
            continue
        comp = unq(row[idx["label_comp_id"]])
        atom = unq(row[idx["label_atom_id"]])
        try:
            x = float(row[idx["Cartn_x"]])
            y = float(row[idx["Cartn_y"]])
            z = float(row[idx["Cartn_z"]])
        except (ValueError, KeyError):
            continue
        try:
            asym = unq(row[idx["label_asym_id"]]) if "label_asym_id" in idx else ""
        except Exception:
            asym = ""
        out.append((atom, comp, unq(row[idx["label_seq_id"]]), x, y, z, asym))
    # An entity is routinely deposited as several copies in the asymmetric unit (4WRA_26 is
    # chains Z and BC, 62707 atoms each). Taking them all multiplies the residue count and
    # overflows the legacy serial field. Keep the largest single chain.
    if not out:
        return out
    by_asym = defaultdict(list)
    for r in out:
        by_asym[r[6]].append(r)
    best = max(by_asym.values(), key=len)
    return best


CG_ATOMS = {"P", "C4'", "N9", "N1"}  # the three beads of the cgRNASP model


def write_pdb(rows, path: Path):
    """Write legacy fixed-column PDB. Returns (residue count, mode).

    Columns follow the existing Training_set files exactly:
      [0:6] record  [6:11] serial  [12:16] atom  [17:20] resname
      [21] chain    [22:26] resseq  [30:38] x   [38:46] y  [46:54] z

    Legacy PDB serials are five columns wide, so anything past 99,999 atoms would push the
    coordinate columns out of alignment and desynchronise the strict reader. Past that ceiling
    the file is written with only the three atoms the CG model consumes -- which is all any
    reader in this repository takes from these files. mode is "full" or "cg3".
    """
    mode = "full"
    if len(rows) > 99999:
        rows = [r for r in rows if r[0] in CG_ATOMS]
        mode = "cg3"
        if len(rows) > 99999:
            return 0, "overflow"
    by_res = {}
    for atom, comp, seq, x, y, z, asym in rows:
        by_res.setdefault((asym, seq), []).append((atom, comp, x, y, z))
    if not by_res:
        return 0, "empty"
    lines, serial = [], 1
    for (asym, seq), atoms in by_res.items():
        comp = COMP_MAP.get(atoms[0][1], atoms[0][1])
        try:
            rseq = int(seq)
        except ValueError:
            rseq = 1
        for atom, _c, x, y, z in atoms:
            aname = atom[:4]
            # PDB puts sub-4-character names starting at column 14, not 13: " P  " for the
            # phosphate, " O5'" for the backbone oxygen, but "OP1" keeps column 13. The
            # existing Training_set files follow this, and a strict reader takes the element
            # from columns 13-14, so getting it wrong would misread phosphorus as "P" alone.
            if len(aname) < 4 and not aname[:1].isdigit():
                aname = " " + aname
            lines.append(
                f"ATOM  {serial:5d} {aname:<4s} {comp:>3s} A{rseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
            serial += 1
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(lines)
    return len(by_res), mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=None,
                    help="subset of small mid large lsu (default: all)")
    ap.add_argument("--manifest", default=r"C:\tmp\manifest.json")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--min-len", type=int, default=0,
                    help="skip representatives shorter than this")
    ap.add_argument("--limit", type=int, default=0, help="stop after N entries (debug)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.manifest))
    if args.pools:
        rows = [r for r in rows if r["pool"] in args.pools]
    jobs = defaultdict(list)
    for r in rows:
        if r["rep_len"] < args.min_len or not r["entity_ids"]:
            continue
        eid = r["entity_ids"][0]
        entry, _, num = eid.partition("_")
        if entry and num:
            jobs[entry].append((eid, num, r))
    print(f"{len(jobs)} entries to fetch, {sum(len(v) for v in jobs.values())} representatives")

    done = skipped = failed = 0
    log = []
    for i, (entry, items) in enumerate(sorted(jobs.items()), 1):
        cif = TMP / f"{entry}.cif"
        try:
            if not download(entry, cif):
                for eid, _n, _r in items:
                    log.append((eid, "no-mmcif", "", 0))
                failed += len(items)
                print(f"[{i}/{len(jobs)}] {entry:6s} NO mmCIF")
                continue
            for eid, num, r in items:
                atoms = extract_entity(cif, num)
                if not atoms:
                    log.append((eid, "entity-empty", "", 0))
                    failed += 1
                    continue
                target = out_dir / f"{eid}.pdb"
                nres, mode = write_pdb(atoms, target)
                if mode == "overflow":
                    log.append((eid, "overflow", "", 0))
                    failed += 1
                    continue
                log.append((eid, f"ok:{mode}", str(target), nres))
                done += 1
            print(f"[{i}/{len(jobs)}] {entry:6s} ok  ({cif.stat().st_size / 1048576:.1f} MB)")
        finally:
            try:
                os.remove(cif)
            except OSError:
                pass
        if args.limit and i >= args.limit:
            break

    (out_dir / "_fetch_log.tsv").write_text(
        "entity\tstatus\tpath\tresidues\n" +
        "\n".join(f"{a}\t{b}\t{c}\t{d}" for a, b, c, d in log) + "\n", encoding="utf-8")
    print(f"\nwritten {done}   failed {failed}   skipped {skipped}")
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()
