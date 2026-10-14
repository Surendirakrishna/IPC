# Source data and known limitations

## Snapshot

| File (`data/source/`) | SHA-256 | Notes |
| --- | --- | --- |
| `ipc_classification_index.jsonl` | `3467306e31c5e2df3b5d32d6722c949f55d692f0bd7f76dc9b249689744dcd09` | 80,145 unique records, one per line |
| `ipc_schema_tree.json` | `81aaefb9c5386799814b3f5f71cbd2df9692eb268a4aca68a55c2f75cf9aa990` | Original wording; 80,734 occurrences (589 repeated symbols) |
| `ipc_maingroup_duplicates.json` | `a591e5b6637c539db7bdffde8d7db5e53844ab175be423bb09f53e2499f6fa49` | 589 duplicate main-group entries |

The brief named the tree file `ipc_schema_tree(2).json`. The supplied file is named
`ipc_schema_tree.json`, and its hash matches the expected value. You can change the filename with
`SOURCE_TREE_FILE`.

`ipc validate-sources` measured the following (manifest: `data/manifests/imports/validation_20260101.json`):

- 8 sections, 132 classes, 655 subclasses, 7,668 main groups and 71,682 subgroups.
- `taxonomy_version` 20260101.
- 15,035 reference edges, none pointing to a missing code.
- No duplicate canonical codes and no display-code collisions.
- Parents, ancestors and children are consistent.

These counts are compared only when all three hashes match
`data/manifests/expected_snapshot_20260101.json`. They are not universal IPC rules.

## Limitations (reported by validation, not repaired)

1. **Flattened subgroups.** Every subgroup sits at depth 5, directly under its main group. The
   inputs do not contain the deeper dot hierarchy, and nothing reconstructs it. A parent path
   therefore stops at the main group.
2. **"CPC Scheme" root title.** The tree root is `{"symbol": "IPC", "title": "International Patent
   Classification (CPC Scheme)", "version": "20260101"}`. This is exposed as a source-quality warning
   by `/v1/taxonomy/status` and on the `/status` page. The snapshot is **not** certified as a complete
   official IPC release.
3. **Cleaned and original titles differ.** Only 7,688 records carry `title_raw`. The importer walks
   the original tree and stores, per code:
   - `original_title`: the wording from the matching occurrence;
   - `original_title_source`: which occurrence it came from and why.

   Every occurrence is also kept in `ipc_source_occurrences`.
4. **Duplicate main groups** are separate source contexts of one symbol. There is one canonical
   record, with `titles`, `canonical_title` and `context_heading` kept. Both tree occurrences are
   preserved. Canonical occurrence selection is deterministic: normalized title match, then prefix
   match, then the last occurrence.
5. **References are edges without conditions.** `cross_references` and `precedence_references`
   become `ipc_reference_edges`. Each edge keeps the verbatim parenthetical clause in which the
   target is cited, when one exists. Conditions and exceptions are never inferred, and a referenced
   code is never assigned just because the edge exists.
6. **No classification notes or priority rules** are in the inputs, such as the H04L 9/06–9/32
   last-place rule. The pipeline reports a rule check as unresolved and moves same-main-group
   competitors to `alternatives`. It does not invent rule semantics.
7. **Code formats are preserved.** Internal zero-padded codes (`B60L0053120000`) and display codes
   (`B60L 53/12`) come from the source as a lookup pair. No padding conversion is derived.
