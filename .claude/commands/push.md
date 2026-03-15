---
description: Commit with conventional commit message and push. Optionally run tests first.
---

First, look at the staged/unstaged changes to judge whether tests are needed:
- Run `git diff --stat HEAD` to see what files changed

If the changes touch any Python logic files (data/, features/, models/, evaluation/, signals/, scripts/, config.py, gen.py, backtest.py, collect.py, etc.), check whether tests have recently passed before deciding to run them:

```bash
STAMP=/tmp/bpa_last_test_run
if [ -f "$STAMP" ]; then
  LAST=$(cat "$STAMP")
  NOW=$(date +%s)
  AGE=$((NOW - LAST))
  if [ $AGE -lt 300 ]; then
    echo "Tests passed ${AGE}s ago — skipping re-run"
    SKIP_TESTS=1
  fi
fi
```

If `SKIP_TESTS` is not set, run tests:
1. Run `cd /home/ixn/Documents/code/crypto/bitcoin-price-analysis && .venv/bin/python -m pytest tests/ -v`
   - If tests fail: show a summary of failures, STOP — do NOT commit or push, do NOT attempt to fix errors automatically
   - If tests pass: record the timestamp and continue:
     ```bash
     date +%s > /tmp/bpa_last_test_run
     ```

If the changes are trivial (docs, comments, config templates, markdown, command files, .gitignore, etc.), skip tests and continue directly.

Then:
2. **Check if the architecture diagram needs updating.** The diagram lives between `<!-- ARCHITECTURE_DIAGRAM_START -->` and `<!-- ARCHITECTURE_DIAGRAM_END -->` in `README.md`. Regenerate it ONLY if the diff includes any of these structural changes:
   - A module directory was added or removed (new/deleted directory under `data/`, `features/`, `models/`, `evaluation/`, `signals/`, `scripts/`)
   - A new data source or model type was added
   - The pipeline flow changed significantly (new stage, removed stage, reordered stages)

   If none of these apply, skip this step entirely. When regenerating, replace the content between the markers with an updated Mermaid diagram that reflects the current architecture. Stage the changed `README.md`.

3. Run `git add .` to stage all changes
4. Run `git status` and `git diff --staged` to review what will be committed
5. Commit with a conventional commit message (feat:, fix:, refactor:, chore:, docs:, test:, etc.)
6. Update `CHANGELOG.md` with a new dated section for today (or append to today's section if one exists). Keep entries **extremely concise** — one short line per item, no essays. Group under Added/Changed/Fixed/Removed as appropriate. Only include items that matter for research workflow (model changes, feature engineering, evaluation pipeline, data sources, config thresholds, signal contract). Skip pure test-only or trivial doc changes. Then `git add CHANGELOG.md` and amend the commit.
7. Push to the remote branch
