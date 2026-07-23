# Security policy

Please report suspected vulnerabilities privately through GitHub's security
advisory interface rather than opening a public issue.

## Model-loading boundary

The scraper loads only the AllenAI SPECTER2 base model, pinned to an immutable
Hugging Face commit SHA in `main.py`. Remote model code is disabled. Do not
make the model identifier or revision user-configurable without revisiting
this boundary and the current dependency audit.

The separate AdapterHub runtime is intentionally not installed. Its current
release requires the vulnerable Transformers 4.57 line. The repository instead
uses AllenAI's standard SPECTER2 base checkpoint with a patched Transformers 5
release. Any future embedding-model migration must rebuild the complete
dataset before incremental refreshes resume so incompatible vector spaces are
never mixed.

CI audits the complete installed environment without vulnerability
exceptions. This boundary was last reviewed on 2026-07-23.
