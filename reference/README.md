# reference/ — oracle, golden fixtures, acceptance tests

Read-only for agents (a hook blocks edits). The human regenerates fixtures only after accepting a
decision in `docs/DECISIONS.md`.

| Path | What it is |
| --- | --- |
| `qet_ref.py` | Literal, slow, stdlib-only implementation of QET (canonical form, contracts, Compile, Reduce, Render, lattice, every configuration's edge weights, sub-lattice, toy table trainer). It defines the expected behaviour; production code must never import it. |
| `make_fixtures.py` | Writes `fixtures/*.json` from the oracle. `--check` exits 1 if a fixture on disk is stale. |
| `fixtures/` | Golden inputs and outputs (see the `description` field in each file). |
| `test_oracle.py` | Self-tests of the oracle: the method page's worked numbers and the identities of SPEC §9 on hundreds of random DAGs. |
| `acceptance/test_core_fixtures.py` | Tests the real implementation through the facade `qet.core.api` (SPEC §10). Milestone E0 is done when these pass. |

```bash
python reference/make_fixtures.py --check   # fixtures match the oracle
pytest reference -q                          # oracle self-tests + acceptance tests of src/qet
```

The fixtures pin D-001 (tool-first action strings), D-003 (English template text) and the rest of SPEC.md.
To change one of those, accept a new decision, edit `qet_ref.py`, run `make_fixtures.py`, and commit
the oracle, fixtures and decision together.
