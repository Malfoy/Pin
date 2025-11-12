# kmer_indexer

Rust utility that builds a minimizer dictionary from FASTA sequences using
SIMD multi-minimizers, then re-queries the same input to ensure every k-mer is
covered by at least one stored minimizer.

## Usage

```bash
# build + query freshly generated index
cargo run --release -- \
    --kmer 31 \
    --minimizer 17 \
    --seeds 4 \
    --build training.fa \
    --query validation.fa \
    --save-index k31_m17.idx

# query using a previously saved index
cargo run --release -- \
    --kmer 31 \
    --minimizer 17 \
    --seeds 4 \
    --load-index k31_m17.idx \
    --query validation.fa
```

- `--kmer/-k`: length of each k-mer (`K`).
- `--minimizer/-m`: length of minimizers (`M`) used during indexing.
- `--seeds/-n`: number of hash seeds (SIMD lanes) to use. The current binary
  supports values from 1 to 16.
- The final positional argument is the FASTA file to index.

Constraints:

- `K >= M`.
- `K - M + 1` must be odd (SIMD minimizer iterators require an odd window).

On success the CLI prints aggregate statistics about the indexing run and
reports that validation passed. If any k-mer from the FASTA file cannot be
covered by at least one stored minimizer, the tool exits with a descriptive
error that shows the offending sequence and k-mer offset.

## Implementation Notes

- FASTA parsing is streaming, so files are read twice (once for insertion,
  once for validation) without keeping all records in memory, while batching
  chunks of sequences so they can be processed in parallel via Rayon.
- The minimizer dictionary is a 1,000-way sharded `AHashSet` protected by
  `parking_lot::Mutex` locks to minimize contention while allowing updates from
  many threads.
- Each sequence is first scanned with the “all minimizers” iterator to find
  which k-mers are already covered by the current dictionary; the sticky
  iterator is only invoked to index the remaining uncovered kmers, so no new
  entries are inserted unless they are strictly needed.
- Serialized indexes store the full sharded hash-set plus `K`, `M`, and seed
  metadata in a `bincode` blob compressed with `zstd -1`, enabling
  `--load-index`/`--save-index` workflows without rebuilding from FASTA while
  keeping files small.
- The CLI reports construction time, query/validation time, and also runs an
  additional 1 Mb random-sequence lookup, printing how many k-mers hit the
  index alongside the query time for that synthetic workload.
- `multiminimizers` (the SIMD minimizer implementation from
  https://github.com/lrobidou/multiminimizers) is vendored under
  `external/multiminimizers` and used directly as a path dependency.
- Stored dictionary keys are raw (non-hashed) minimizer encodings, which keeps
  membership tests exact while benefiting from `ahash` performance.
