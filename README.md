# Pin

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
