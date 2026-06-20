"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# manifest.py — freeze to versioned JSONL; the loop consumes the FROZEN file, never re-samples
def freeze(ops, n_per_op=12, path="manifest_v1.jsonl"):
    _assert_split_disjoint()
    rows = []
    for op in ops:
        for i in range(n_per_op):
            rows.append(sample_task(op, i, "train"))
            rows.append(sample_task(op, i, "test"))
    _atomic_write(path, rows)                    # write .tmp then os.replace (atomic, no half-read)
