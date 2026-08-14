"""TorusFold web assets — hand-written SPA (no build step).

Served by ``server/api.py``: ``GET /`` returns ``index.html``; ``GET
/web/{name}`` returns ``app.js`` / ``style.css`` / ``circrna_viewer.js``.

The Mol* viewer logic in ``circrna_viewer.js`` is extracted from the IGEM
``html_renderer.py`` HTML_TEMPLATE (Mol* 5.10.1, five compatibility fixes
preserved). The data contract matches: ``PDB_DATA`` string + ``FP`` JSON
object (per_residue arrays + scalar singletons + coloring_schemes list).
"""
