# Working with this shared snapshot

This folder is an independent copy of CAP's current implementation, not a symlink or import into the author's workstation. Treat edits here as edits to the shared version; later CAP changes will not appear automatically. Agree which repository owns a change before copying it back.

The cleanup preserves all Python modules, tests and existing examples. It adds a current sensor recipe, selected human-facing evidence, focused documentation and package-data declarations for static calibration. Existing sibling files in `humanoid-astra/` are untouched. `SOURCE_MANIFEST.json` maps copied files to original paths and SHA-256 hashes; changed documentation/metadata and the GPU-normalized recipe are identified separately.

Keep controller changes small and readable. Update `g1cap/README.md` when the reading order or runtime ownership changes, `docs/code-walkthrough.md` when execution changes, and the appropriate `g1cap/tool_docs/` contract when agent-visible behavior changes. Keep benchmark solutions and evaluator truth out of those agent-facing documents.

Run `python3 -m unittest discover -s tests -v` from this root before sharing changes. Record failures/skips and native environment details. Keep disposable probes in `temp/` and raw results in `runs/`; retain a compact verified lesson with links to the original evidence when conducting experiments. Every new physical trial needs a labeled full video, including failures. Preserve one episode across interactive revisions.

This snapshot includes no new licensing grant. Confirm project ownership and third-party asset/model redistribution terms before publishing outside the collaboration. Static robot calibration was derived from the installed Arena model; its provenance is described in `g1cap/assets/README.md`. Upstream source/models and original full experiment archives remain external dependencies.
