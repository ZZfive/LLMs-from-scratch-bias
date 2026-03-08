# Upstream Sync Automation

This repository includes a GitHub Actions workflow that tracks changes from
`rasbt/LLMs-from-scratch`, translates supported documentation files, and opens
or updates a pull request with the results.

## What It Does

- Checks upstream every three days and also supports manual runs.
- Tracks `README.md`, other Markdown files, and Jupyter notebooks.
- Rewrites only Markdown content and notebook markdown cells.
- Avoids overwriting files that were changed manually after the last automated sync.
- Creates or updates a fixed automation branch: `bot/upstream-sync`.
- Opens or updates a failure issue when the translation provider fails.

## Required Repository Configuration

Configure these values in GitHub before enabling the workflow:

- `secrets.TRANSLATION_API_KEY`
- `secrets.TRANSLATION_BASE_URL`
- `vars.TRANSLATION_MODEL`

Optional:

- `vars.TRANSLATION_TIMEOUT_SECONDS`
- `vars.MAX_CHANGED_FILES_PER_RUN`

The translation endpoint must be compatible with the OpenAI chat completions
API format.

## First Run

The initial scheduled run bootstraps the upstream SHA baseline without
rewriting existing files. This avoids unexpectedly taking over files that were
maintained manually.

If you want automation to take over the existing translated Markdown and
notebooks, trigger the workflow manually with:

- `force_full_resync=true`

That manual run rebuilds all supported upstream files and records them in
`.sync/managed_manifest.json`.

## Failure Notifications

If translation fails because of quota limits, authentication problems, rate
limits, missing models, service errors, or timeouts, the workflow:

- leaves `.sync/state.json` unchanged
- creates or updates the issue `bot: translation sync failure`
- comments on the open sync pull request when one exists

Use the generated issue or PR comment as the primary alert surface.
