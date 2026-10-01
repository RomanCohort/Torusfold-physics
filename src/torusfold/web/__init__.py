"""TorusFold web assets — hand-written SPA (no build step).

Served by ``serve.py``: ``GET /`` returns ``index.html``; ``GET /web/{name}``
returns any file in this directory (``app.js``, ``style.css``, ``panels.css``,
the ``modules/`` scripts, and the vendored ``3Dmol-min.js``).

Structures are drawn with 3Dmol.js, wrapped by ``circrna_viewer_3dmol.js``. The
data contract matches the backend: a ``PDB_DATA`` string plus an ``FP`` JSON
object (per_residue arrays + scalar singletons + coloring_schemes list).

The previous Mol* implementation (``circrna_viewer.js`` plus the 4.8 MB
``molstar.js`` and ``molstar.css``) has been removed; it is in the git history if
it is ever needed.
"""
