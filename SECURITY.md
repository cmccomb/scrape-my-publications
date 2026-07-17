# Security policy

Please report suspected vulnerabilities privately through GitHub's security
advisory interface rather than opening a public issue.

## Model-loading boundary

The SPECTER2 adapter currently requires `adapters==1.3.0`, which in turn
requires the `transformers` 4.57 line. Generic advisories exist for untrusted
checkpoint, trainer, X-CLIP, LightGlue, and remote-attention loading paths in
that dependency family. This project does not use those paths.

The scraper loads only the AllenAI SPECTER2 base model and adapter, both pinned
to immutable Hugging Face commit SHAs in `main.py`. Remote model code is
disabled. Do not make the model identifiers or revisions user-configurable
without revisiting this boundary and the current dependency audit.

CI continues to audit the complete installed environment and has narrow,
documented exceptions for `PYSEC-2025-217`, `PYSEC-2026-2288`,
`PYSEC-2026-2289`, and `PYSEC-2026-2290`. Those exceptions cover the unused
paths above and should be removed as soon as the adapter stack supports a fixed
`transformers` release. This boundary was last reviewed on 2026-07-17.
