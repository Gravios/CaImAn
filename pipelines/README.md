# Pipeline Framework

The `pipelines/` directory contains the Gravios fork's session management
tools.  Each calcium imaging recording session gets its own pair of files
(a `.py` script and a `.json` config) derived from these templates.

## Files

| File | Purpose |
|---|---|
| `template_pipeline.py` | Generic pipeline script — copy and rename per session |
| `template_pipeline.json` | Fully-commented parameter template |
| `new_session.py` | CLI: create session files, run MC, estimate parameters |

## Quick start

```bash
# Create session files, run MC, estimate parameters
python pipelines/new_session.py \
    stroh-ej-20140708-TL2 \
    /data/src/stroh-ej/RawDataSel_AD_Project/G1_B6J/08072014/ \
    -y --run-mc --estimate-params

# Review and run
python /data/src/stroh-ej/.../stroh-ej-20140708-TL2_pipeline.py
```

## `new_session.py` flags

| Flag | Description |
|---|---|
| `--run-mc` | Run GPU rigid MC; analyse shifts → update `max_shifts`; delete mmap after use |
| `--estimate-params` | Estimate gSig/thresholds from MC mmap; write into JSON |
| `--n-frames N` | Frames to subsample for estimation (default 500) |
| `-y` / `--force` | Overwrite existing files (batch-safe) |
| `--dry-run` | Preview without writing |
| `--gSig N` | Override gSig (sets gSiz automatically) |
| `--fr HZ` | Frame rate |
| `--method-init` | `corr_pnr` or `greedy_roi` |

## Batch processing

```bash
for dir in /data/src/stroh-ej/RawDataSel_AD_Project/G2_5FAD/*/; do
    python pipelines/new_session.py "$(basename $dir)_TL002" "$dir" \
        -y --run-mc --estimate-params
done
```

See `docs/source/session_management.rst` for the complete reference.
