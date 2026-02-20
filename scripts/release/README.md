# Release Scripts

Utilities for release engineering (`v1.0.0-rc1` -> `v1.0.0` flow).

## Commands

Run from repository root:

1. Release gate (lint/type/test/build/check):
```bash
python scripts/release/release.py gate --clean-dist
```

2. Build only:
```bash
python scripts/release/release.py build --clean-dist
```

3. Twine check only:
```bash
python scripts/release/release.py check
```

4. Generate release notes draft:
```bash
python scripts/release/release.py notes --version v1.0.0-rc1
```

5. Update package version:
```bash
python scripts/release/release.py set-version --version 1.0.0rc1
```

6. Create tag (and push optionally):
```bash
python scripts/release/release.py tag --version v1.0.0-rc1 --push
```

Use `--force` to recreate existing tags and `--allow-dirty` to bypass clean-worktree check.
