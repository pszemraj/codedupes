# Clone fixture notes

The duplicated code in this project is deliberate. Do not run a deduplication refactor over the fixture before evaluating a detector.

The canonical labels live in `clone-ground-truth.json`:

- `exact`: byte-identical source regions in two modules.
- `edit_distance`: structurally close bubble renderers with insertions, deletions, and token substitutions.
- `semantic`: two source-dissimilar wrapping algorithms with tested equivalent behavior over the fixture corpus.

Markers are the stable identifiers. Line numbers are included for tools that need them, but markers should be preferred when the source is edited.
