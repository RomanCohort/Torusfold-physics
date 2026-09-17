"""The chunked clash metric must be exactly the metric it replaced, only lighter.

Measured, 2026-09-16: torch.cdist(beads, beads) per sampled frame allocated an N x N float64
matrix -- 617 MB for a 2929-residue chain, twice with the diagonal copy, 2500 times per chain. A
single worker was holding 25 GB of commit when the machine hit its limit and Windows popped
"virtual memory insufficient"; three workers died inside VCRUNTIME140 at 23:04:12, Pool replaced
them without a word, and the round hung at 98 percent for 29.7 h.

The replacement is the same O(N^2) work with nothing bigger than chunk x N alive at once. These
tests hold it to bit-equality with the implementation it replaced, because "a lighter diagnostic"
is only allowed to be a diagnostic if it still measures the same thing.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ibi_core as IC                 # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402


def _naive(pos):
    """The implementation the chunked one replaced, kept here as the reference."""
    beads = pos.reshape(pos.shape[0], -1, 3)
    dd = torch.cdist(beads, beads)
    dd = dd + torch.eye(dd.shape[-1], device=dd.device) * 10.0
    return float(dd.min()), int((dd < C.CLASH_DIST).sum()), int((dd < C.CLASH_SIGMA).sum())


def _same_metric(a, b, rel=0.0):
    """Same three numbers, with rel=0 demanding bit-equality of the distance.

    NOT every comparison here can demand that, and the reason is cdist's, not this metric's: the
    naive path hands the WHOLE batch to torch.cdist, which takes a batched kernel, while the
    chunked path calls it once per replica. In float32 the two round differently at about 1e-5
    relative (measured: 0.399993896484375 against 0.39999985694885254 on the same geometry). The
    metric a caller reads is unchanged -- and a 1e-4 window still catches every real mistake, since
    a cross-replica pair is 100 nm out and an off-by-one chunk is a whole bead out, not 1e-5.
    Nothing consumes clash_min as a criterion (it is printed, and diagnose_chain_meltdown.py reads
    its minimum), so this is a diagnostic tolerance and not a physics one.
    """
    if rel == 0.0:
        return a == b
    return (abs(a[0] - b[0]) <= rel * max(abs(a[0]), abs(b[0]), 1e-30)
            and a[1] == b[1] and a[2] == b[2])


def test_chunked_matches_the_n_by_n_matrix():
    for nrep, n, seed in [(1, 137, 1), (1, 512, 2), (1, 513, 3), (1, 2, 7)]:
        g = torch.Generator().manual_seed(seed)
        pos = torch.rand(nrep, n, 3, generator=g) * 3.0
        assert _same_metric(IC.closest_bead_pair(pos), _naive(pos)), (nrep, n)
    # multi-replica batches: same metric, to the last digits cdist's kernel choice allows
    for nrep, n, seed in [(3, 200, 4), (2, 1000, 5)]:
        g = torch.Generator().manual_seed(seed)
        pos = torch.rand(nrep, n, 3, generator=g) * 3.0
        assert _same_metric(IC.closest_bead_pair(pos), _naive(pos), rel=1e-4), (nrep, n)


def test_replicas_stay_independent():
    """Two replicas far apart: a batch must not invent contacts between them.

    The naive path cdist's each replica separately, and so must this one -- otherwise every
    multi-replica caller (ibi_round0's 8, REMD's 24) would report the distance between two
    different trajectories as the closest bead pair.
    """
    a = torch.zeros(1, 10, 3)
    a[0, :, 0] = torch.arange(10.0) * 0.4
    b = a.clone()
    b[0, :, 0] += 100.0
    pair = torch.cat([a, b], dim=0)
    assert _same_metric(IC.closest_bead_pair(pair), IC.closest_bead_pair(a), rel=1e-4), (
        "the two replicas are 100 nm apart; their cross distances must not be counted")


def test_a_lone_bead_reports_no_pair_instead_of_the_diagonal_pad():
    """The old implementation returned 10.0 here -- the pad it wrote to hide the diagonal."""
    one = torch.rand(1, 1, 3, generator=torch.Generator().manual_seed(12))
    assert _naive(one)[0] == 10.0, "this is the artefact the chunked version refuses to report"
    assert IC.closest_bead_pair(one)[0] == float("inf")


def test_the_chunk_is_a_bound_not_a_rounding():
    """n = chunk + 1 exercises the short final block, which is where an off-by-one would show."""
    chunk = 8
    g = torch.Generator().manual_seed(11)
    pos = torch.rand(1, chunk + 1, 3, generator=g) * 2.0
    assert _same_metric(IC.closest_bead_pair(pos, chunk=chunk), _naive(pos), rel=1e-4)
