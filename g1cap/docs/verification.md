# Verification of the shared package

Checked on 2026-09-18, from this standalone project root.

| Check | Result |
| --- | --- |
| Source fidelity | All copied Python implementation, test and example files are byte-identical to CAP; 261 copied/provenance entries, 256 unchanged (four human-facing READMEs rewritten and one sensor recipe GPU normalized) |
| Full unit/contract suite | Python 3.12: 599 tests discovered, 540 passed, 59 skipped; no failures/errors |
| Worker execution | Native macOS worker sandboxes enabled for the test invocation; no worker sandbox bypass |
| Mock demo | Expected `policy_exited_early` then `success`; no generation/episode errors; synthetic only |
| CLI | Both main and interactive `--help` commands exit successfully |
| Wheel | Built with Python 3.10/setuptools 80.7.1; archive contains every Python module, tool contract and all three calibration assets |
| Documentation | All local Markdown file links resolve within the shared folder |
| Videos | Original hashes match; H.264, 1,631 frames/32.62 s and 364 frames/7.28 s respectively |

Commands used for the main checks:

```sh
python3.12 -m unittest discover -s tests -v
python3 -m g1cap demo --out /absolute/path/to/new-mock-output
python3.10 -m pip wheel --no-deps --no-build-isolation --wheel-dir /absolute/path/to/wheels .
```

The default system Python 3.14 lacks NumPy and produced two sensor-test import errors even after worker sandbox permission was available. Initial execution inside the outer application sandbox also blocked worker/socket checks. Neither result is hidden: the successful suite used the existing scientific Python 3.12 environment with NumPy 2.3.4 and SciPy 1.16.2, and permission to create native worker sandboxes. Install `.[analysis]` in your own environment before full discovery.

Skipped tests require optional/native libraries or platform-specific resources. No new GPU simulation, upstream installation, physical robot run or model generation was performed. Passing these tests does not qualify native runtime compatibility, full sensor-based transfer or hardware deployment.

Raw logs and the packaging verification script remain in the original CAP archive under `runs/sharing-package-001/` and `temp/sharing-package/`. The shared folder contains a portable source hash manifest, not the full historical experiment archive.
