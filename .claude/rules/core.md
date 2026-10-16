---
paths:
  - "src/qet/core/**"
  - "src/qet/train/**"
  - "src/qet/infer/**"
  - "tests/core/**"
  - "tests/train/**"
---
# Core, training and inference code

- Behaviour is defined by docs/SPEC.md and pinned by reference/fixtures/. reference/qet_ref.py shows the intended
  logic in plain Python; match it, never import it from src/.
- Change math only with a failing test first. Lattice quantities stay float64.
- Before finishing, re-check: prompts depend only on (task, H); one renderer for training and inference; one
  conflict function for Compile and Reduce; log π = token sum including the end token; no renormalization;
  pass 1 in eval mode without gradients; weights detached in pass 2; no update between passes; the true
  objective is logged; configurations differ only in weights and start states.
- Property tests keep their sample sizes and tolerances.
