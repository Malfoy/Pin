use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ahash::AHashSet;
use anyhow::{Context, Result, anyhow, bail, ensure};
use bincode::{deserialize_from, serialize_into};
use clap::Parser;
use flate2::read::MultiGzDecoder;
use multiminimizers::{
    compute_all_superkmers_linear_streaming, compute_superkmers_linear_streaming,
};
use num_format::{Locale, ToFormattedString};
use parking_lot::Mutex;
use rand::Rng;
use rayon::{ThreadPoolBuilder, prelude::*};
use serde::{Deserialize, Serialize};
use zstd::stream::{read::Decoder as ZstdDecoder, write::Encoder as ZstdEncoder};

mod ocm;

use ocm::Config as OcmConfig;

const CANONICAL: bool = true;
const SUPPORTED_MAX_SEEDS: usize = 16;
const PARALLEL_CHUNK_SIZE: usize = 256;
const PARTITION_COUNT: usize = 10000;
const SHARD_INITIAL_CAPACITY: usize = 256;
const RANDOM_QUERY_LENGTH: usize = 1_000_000;
const ZSTD_LEVEL: i32 = -1;

#[derive(Parser, Debug)]
#[command(
    name = "kmer_indexer",
    about = "Indexes k-mers by their minimizers and validates coverage"
)]
struct Args {
    /// Length of the k-mers.
    #[arg(short = 'k', long = "kmer", value_name = "K")]
    k: usize,

    /// Length of minimizers (m-mers).
    #[arg(short = 'm', long = "minimizer", value_name = "M")]
    m: usize,

    /// Number of distinct seeds used by the SIMD minimizer iterator.
    #[arg(short = 'n', long = "seeds", value_name = "N")]
    seeds: usize,

    /// FASTA file whose sequences should be inserted into the index.
    #[arg(long = "build", value_name = "FASTA")]
    build_fasta: Option<PathBuf>,

    /// FASTA file to query against the index (fails if a k-mer is missing).
    #[arg(long = "query", value_name = "FASTA")]
    query_fasta: Option<PathBuf>,

    /// Load an existing serialized index from disk before performing operations.
    #[arg(long = "load-index", value_name = "FILE")]
    load_index: Option<PathBuf>,

    /// Save the final index to disk after building.
    #[arg(long = "save-index", value_name = "FILE")]
    save_index: Option<PathBuf>,

    /// Number of worker threads to use (defaults to logical CPU count).
    #[arg(long = "threads", value_name = "THREADS")]
    threads: Option<usize>,

    /// Use Open-Close Mod minimizers instead of SIMD sticky minimizers.
    #[arg(long = "OCM", alias = "ocm")]
    ocm: bool,
}

#[derive(Default, Clone, Copy)]
struct BuildStats {
    sequences_indexed: usize,
    sequences_skipped: usize,
    kmers_seen: u64,
    minimizers_emitted: u64,
    minimizers_new: u64,
}

impl BuildStats {
    fn merge(&mut self, other: BuildStats) {
        self.sequences_indexed += other.sequences_indexed;
        self.sequences_skipped += other.sequences_skipped;
        self.kmers_seen += other.kmers_seen;
        self.minimizers_emitted += other.minimizers_emitted;
        self.minimizers_new += other.minimizers_new;
    }
}

#[derive(Default, Clone, Copy)]
struct ValidationStats {
    sequences_validated: usize,
    kmers_checked: u64,
}

impl ValidationStats {
    fn merge(&mut self, other: ValidationStats) {
        self.sequences_validated += other.sequences_validated;
        self.kmers_checked += other.kmers_checked;
    }
}

struct SequenceRecord {
    header: String,
    sequence: Vec<u8>,
}

struct PartitionedMinimizerSet {
    shards: Vec<Mutex<AHashSet<u64>>>,
}

#[derive(Serialize, Deserialize)]
struct SerializedIndex {
    shard_count: usize,
    shards: Vec<Vec<u64>>,
    k: usize,
    m: usize,
    seeds: usize,
}

impl PartitionedMinimizerSet {
    fn new(shard_count: usize, shard_capacity: usize) -> Self {
        assert!(shard_count > 0, "shard_count must be non-zero");
        let shards = (0..shard_count)
            .map(|_| Mutex::new(AHashSet::with_capacity(shard_capacity)))
            .collect();
        Self { shards }
    }

    fn from_shards(shard_entries: Vec<Vec<u64>>) -> Self {
        let shards = shard_entries
            .into_iter()
            .map(|entries| {
                let mut set = AHashSet::with_capacity(entries.len());
                set.extend(entries);
                Mutex::new(set)
            })
            .collect();
        Self { shards }
    }

    fn shard_index(&self, value: u64) -> usize {
        (value % self.shards.len() as u64) as usize
    }

    fn insert(&self, value: u64) -> bool {
        let idx = self.shard_index(value);
        let mut guard = self.shards[idx].lock();
        guard.insert(value)
    }

    fn contains(&self, value: u64) -> bool {
        let idx = self.shard_index(value);
        let guard = self.shards[idx].lock();
        guard.contains(&value)
    }

    fn snapshot(&self) -> Vec<Vec<u64>> {
        self.shards
            .iter()
            .map(|mutex| mutex.lock().iter().copied().collect())
            .collect()
    }

    fn total_entries(&self) -> usize {
        self.shards.iter().map(|mutex| mutex.lock().len()).sum()
    }
}

impl Default for PartitionedMinimizerSet {
    fn default() -> Self {
        Self::new(PARTITION_COUNT, SHARD_INITIAL_CAPACITY)
    }
}

fn save_index_to_file(
    path: &Path,
    dictionary: &PartitionedMinimizerSet,
    k: usize,
    m: usize,
    seeds: usize,
) -> Result<()> {
    let serialized = SerializedIndex {
        shard_count: dictionary.shards.len(),
        shards: dictionary.snapshot(),
        k,
        m,
        seeds,
    };

    let file = File::create(path)
        .with_context(|| format!("Failed to create index file '{}'", path.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = ZstdEncoder::new(writer, ZSTD_LEVEL)
        .with_context(|| format!("Failed to initialize zstd encoder for '{}'", path.display()))?;
    serialize_into(&mut encoder, &serialized).with_context(|| {
        format!(
            "Failed to serialize compressed index to '{}'",
            path.display()
        )
    })?;
    encoder
        .finish()
        .with_context(|| format!("Failed to finalize index file '{}'", path.display()))?;
    Ok(())
}

fn load_index_from_file(
    path: &Path,
    expected_k: usize,
    expected_m: usize,
    expected_seeds: usize,
) -> Result<PartitionedMinimizerSet> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open index file '{}'", path.display()))?;
    let reader = BufReader::new(file);
    let mut decoder = ZstdDecoder::new(reader)
        .with_context(|| format!("Failed to initialize zstd decoder for '{}'", path.display()))?;
    let snapshot: SerializedIndex = deserialize_from(&mut decoder)
        .with_context(|| format!("Failed to deserialize index from '{}'", path.display()))?;

    ensure!(
        snapshot.k == expected_k,
        "Loaded index was built with K = {}, but CLI requested K = {}",
        snapshot.k,
        expected_k
    );
    ensure!(
        snapshot.m == expected_m,
        "Loaded index was built with M = {}, but CLI requested M = {}",
        snapshot.m,
        expected_m
    );
    ensure!(
        snapshot.seeds == expected_seeds,
        "Loaded index was built with {} seeds, but CLI requested {}",
        snapshot.seeds,
        expected_seeds
    );
    ensure!(
        snapshot.shard_count == snapshot.shards.len(),
        "Corrupted index: shard count metadata does not match shard data length"
    );

    Ok(PartitionedMinimizerSet::from_shards(snapshot.shards))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.k > 0, "K must be greater than zero");
    ensure!(args.m > 0, "M must be greater than zero");
    ensure!(
        args.k >= args.m,
        "K (k-mer length) must be greater than or equal to M (minimizer length)"
    );
    ensure!(args.seeds >= 1, "The number of seeds must be at least 1");
    ensure!(
        args.seeds <= SUPPORTED_MAX_SEEDS,
        "At most {SUPPORTED_MAX_SEEDS} seeds are supported in this binary"
    );

    let width = args.k - args.m + 1;
    ensure!(
        width % 2 == 1,
        "k - m + 1 must be odd (SIMD minimizers require an odd window); got {width}"
    );

    ensure!(
        args.build_fasta.is_some() || args.load_index.is_some(),
        "Provide --build to create an index, --load-index to reuse an existing one, or both."
    );

    let default_threads = num_cpus::get();
    let requested_threads = args.threads.unwrap_or(default_threads);
    ThreadPoolBuilder::new()
        .num_threads(requested_threads)
        .build_global()
        .map_err(|err| anyhow!("Failed to configure Rayon thread pool: {err}"))?;
    if let Some(path) = &args.build_fasta {
        ensure_path_exists(path, "build FASTA")?;
    }
    if let Some(path) = &args.query_fasta {
        ensure_path_exists(path, "query FASTA")?;
    }
    if let Some(path) = &args.load_index {
        ensure_path_exists(path, "index file")?;
    }

    match args.seeds {
        1 => run::<1>(&args),
        2 => run::<2>(&args),
        3 => run::<3>(&args),
        4 => run::<4>(&args),
        5 => run::<5>(&args),
        6 => run::<6>(&args),
        7 => run::<7>(&args),
        8 => run::<8>(&args),
        9 => run::<9>(&args),
        10 => run::<10>(&args),
        11 => run::<11>(&args),
        12 => run::<12>(&args),
        13 => run::<13>(&args),
        14 => run::<14>(&args),
        15 => run::<15>(&args),
        16 => run::<16>(&args),
        _ => unreachable!("seeds upper-bound is enforced above"),
    }
}

fn run<const N: usize>(args: &Args) -> Result<()> {
    let initial_dictionary = if let Some(path) = &args.load_index {
        load_index_from_file(path, args.k, args.m, args.seeds)?
    } else {
        PartitionedMinimizerSet::default()
    };
    let dictionary = Arc::new(initial_dictionary);
    let ocm_config = if args.ocm {
        Some(Arc::new(OcmConfig::new(args.k, args.m, N)?))
    } else {
        None
    };

    if let Some(build_path) = &args.build_fasta {
        let build_start = Instant::now();
        let build_stats =
            insert_sequences::<N>(build_path, args.k, args.m, &dictionary, ocm_config.clone())?;
        let build_duration = build_start.elapsed();

        println!(
            "Indexed {} sequences ({} kmers considered).",
            fmt_num(build_stats.sequences_indexed),
            fmt_num(build_stats.kmers_seen)
        );
        if build_stats.sequences_skipped > 0 {
            println!(
                "Skipped {} sequences shorter than K.",
                fmt_num(build_stats.sequences_skipped)
            );
        }
        println!(
            "Dictionary now holds {} minimizers.",
            fmt_num(dictionary.as_ref().total_entries())
        );
        println!("Construction time: {build_duration:?}");
        if args.ocm {
            println!(
                "OCM emitted {} minimizers ({} unique)",
                fmt_num(build_stats.minimizers_emitted),
                fmt_num(build_stats.minimizers_new)
            );
        }
    }

    if let Some(save_path) = &args.save_index {
        save_index_to_file(save_path, dictionary.as_ref(), args.k, args.m, args.seeds)?;
        println!(
            "Saved index with {} minimizers to {}",
            fmt_num(dictionary.as_ref().total_entries()),
            save_path.display()
        );
    }

    if let Some(query_path) = &args.query_fasta {
        let validation_start = Instant::now();
        let validation_stats =
            validate_sequences::<N>(query_path, args.k, args.m, &dictionary, ocm_config.clone())?;
        let validation_duration = validation_start.elapsed();
        println!(
            "Validated {} sequences covering {} kmers: all accounted for.",
            fmt_num(validation_stats.sequences_validated),
            fmt_num(validation_stats.kmers_checked)
        );
        println!("Query time: {validation_duration:?}");
    }

    if dictionary.as_ref().total_entries() > 0 {
        let random_sequence = generate_random_sequence(RANDOM_QUERY_LENGTH);
        let total_random_kmers = if random_sequence.len() >= args.k {
            (random_sequence.len() - args.k + 1) as u64
        } else {
            0
        };
        let random_query_start = Instant::now();
        let random_hits = count_kmers_in_dictionary::<N>(
            &random_sequence,
            args.k,
            args.m,
            dictionary.as_ref(),
            ocm_config.as_deref(),
        );
        let random_query_duration = random_query_start.elapsed();
        println!(
            "Random query ({} bp): {} / {} kmers hit in {:?}",
            fmt_num(random_sequence.len()),
            fmt_num(random_hits),
            fmt_num(total_random_kmers),
            random_query_duration
        );
    }
    Ok(())
}

fn insert_sequences<const N: usize>(
    fasta: &Path,
    k: usize,
    m: usize,
    dictionary: &Arc<PartitionedMinimizerSet>,
    ocm_config: Option<Arc<OcmConfig>>,
) -> Result<BuildStats> {
    let mut stats = BuildStats::default();
    stream_fasta_in_chunks(fasta, PARALLEL_CHUNK_SIZE, |chunk| {
        let chunk_stats = process_insert_chunk::<N>(chunk, dictionary, k, m, ocm_config.clone());
        stats.merge(chunk_stats);
        Ok(())
    })?;
    Ok(stats)
}

fn process_insert_chunk<const N: usize>(
    chunk: Vec<SequenceRecord>,
    dictionary: &Arc<PartitionedMinimizerSet>,
    k: usize,
    m: usize,
    ocm_config: Option<Arc<OcmConfig>>,
) -> BuildStats {
    chunk
        .into_par_iter()
        .map(|record| {
            process_single_insert::<N>(&record.sequence, k, m, dictionary, ocm_config.as_deref())
        })
        .reduce(BuildStats::default, |mut acc, item| {
            acc.merge(item);
            acc
        })
}

fn process_single_insert<const N: usize>(
    sequence: &[u8],
    k: usize,
    m: usize,
    dictionary: &PartitionedMinimizerSet,
    ocm_config: Option<&OcmConfig>,
) -> BuildStats {
    if sequence.len() < k {
        return BuildStats {
            sequences_skipped: 1,
            ..BuildStats::default()
        };
    }

    let mut stats = BuildStats {
        sequences_indexed: 1,
        sequences_skipped: 0,
        kmers_seen: (sequence.len() - k + 1) as u64,
        minimizers_emitted: 0,
        minimizers_new: 0,
    };

    let mut covered = vec![false; sequence.len() - k + 1];
    let mut cached_ocm_selections = None;
    if let Some(cfg) = ocm_config {
        let selections =
            ocm::compute_selections(sequence, cfg).expect("OCM minimizer selection failed");
        #[cfg(debug_assertions)]
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static SEQ_COUNTER: AtomicUsize = AtomicUsize::new(0);
            let idx = SEQ_COUNTER.fetch_add(1, Ordering::Relaxed);
            if idx < 1 {
                eprintln!(
                    "OCM seq {} len {} selections {}",
                    idx,
                    sequence.len(),
                    selections.len()
                );
            }
        }
        ocm::apply_dictionary_hits(sequence, cfg, dictionary, &mut covered, &selections);
        cached_ocm_selections = Some(selections);
    } else {
        mark_covered_kmers::<N>(sequence, k, m, dictionary, &mut covered, None);
    }

    if covered.iter().all(|&hit| hit) {
        return stats;
    }

    if let Some(cfg) = ocm_config {
        let selections = cached_ocm_selections.unwrap_or_else(|| {
            ocm::compute_selections(sequence, cfg).expect("OCM minimizer selection failed")
        });
        for sel in selections {
            let (start, end) = match ocm::selection_kmer_range(&sel, covered.len()) {
                Some(range) => range,
                None => continue,
            };
            if start >= end {
                continue;
            }
            if covered[start..end].iter().all(|&hit| hit) {
                continue;
            }
            let min_start = sel.minimizer_pos;
            let min_end = min_start + cfg.minimizer_len();
            if min_end > sequence.len() {
                continue;
            }
            stats.minimizers_emitted += 1;
            let encoded = ocm::encode_minimizer(&sequence[min_start..min_end]);
            if dictionary.insert(encoded) {
                stats.minimizers_new += 1;
            }
            covered[start..end].fill(true);
        }
    } else if let Some(iter) = compute_superkmers_linear_streaming::<N, CANONICAL>(sequence, k, m) {
        for sk in iter {
            if sk.superkmer.len() < k {
                continue;
            }
            let start = sk.superkmer.start();
            if start >= covered.len() {
                continue;
            }
            let raw_end = sk.superkmer.end();
            if raw_end < k {
                continue;
            }
            let end_exclusive = (raw_end - k + 1).min(covered.len());
            if start >= end_exclusive {
                continue;
            }
            if covered[start..end_exclusive].iter().all(|&hit| hit) {
                continue;
            }
            stats.minimizers_emitted += 1;
            if dictionary.insert(sk.get_minimizer_no_hashed()) {
                stats.minimizers_new += 1;
            }
            covered[start..end_exclusive].fill(true);
        }
    }

    if let Some((idx, _)) = covered.iter().enumerate().find(|(_, hit)| !**hit) {
        panic!(
            "Insertion failed to cover k-mer {} in sequence of length {}",
            idx,
            sequence.len()
        );
    }

    stats
}

fn validate_sequences<const N: usize>(
    fasta: &Path,
    k: usize,
    m: usize,
    dictionary: &Arc<PartitionedMinimizerSet>,
    ocm_config: Option<Arc<OcmConfig>>,
) -> Result<ValidationStats> {
    let mut stats = ValidationStats::default();
    stream_fasta_in_chunks(fasta, PARALLEL_CHUNK_SIZE, |chunk| {
        let chunk_stats =
            process_validation_chunk::<N>(chunk, dictionary, k, m, ocm_config.clone())?;
        stats.merge(chunk_stats);
        Ok(())
    })?;
    Ok(stats)
}

fn process_validation_chunk<const N: usize>(
    chunk: Vec<SequenceRecord>,
    dictionary: &Arc<PartitionedMinimizerSet>,
    k: usize,
    m: usize,
    ocm_config: Option<Arc<OcmConfig>>,
) -> Result<ValidationStats> {
    chunk
        .into_par_iter()
        .try_fold(ValidationStats::default, |mut acc, record| {
            let seq_stats = process_single_validation::<N>(
                &record.header,
                &record.sequence,
                k,
                m,
                dictionary,
                ocm_config.as_deref(),
            )?;
            acc.merge(seq_stats);
            Ok(acc)
        })
        .try_reduce(ValidationStats::default, |mut acc, item| {
            acc.merge(item);
            Ok(acc)
        })
}

fn process_single_validation<const N: usize>(
    header: &str,
    sequence: &[u8],
    k: usize,
    m: usize,
    dictionary: &PartitionedMinimizerSet,
    ocm_config: Option<&OcmConfig>,
) -> Result<ValidationStats> {
    if sequence.len() < k {
        return Ok(ValidationStats::default());
    }

    let stats = ValidationStats {
        sequences_validated: 1,
        kmers_checked: (sequence.len() - k + 1) as u64,
    };

    let mut covered = vec![false; sequence.len() - k + 1];
    mark_covered_kmers::<N>(sequence, k, m, dictionary, &mut covered, ocm_config);

    if let Some((idx, _)) = covered.iter().enumerate().find(|(_, hit)| !**hit) {
        bail!(
            "Validation failed: sequence '{}' is missing coverage at k-mer position {}",
            header,
            idx
        );
    }

    Ok(stats)
}

fn mark_covered_kmers<const N: usize>(
    sequence: &[u8],
    k: usize,
    m: usize,
    dictionary: &PartitionedMinimizerSet,
    covered: &mut [bool],
    ocm_config: Option<&OcmConfig>,
) {
    if let Some(cfg) = ocm_config {
        let selections =
            ocm::compute_selections(sequence, cfg).expect("OCM minimizer selection failed");
        ocm::apply_dictionary_hits(sequence, cfg, dictionary, covered, &selections);
        return;
    }
    if let Some(iter) = compute_all_superkmers_linear_streaming::<N, CANONICAL>(sequence, k, m) {
        for sk in iter {
            if sk.superkmer.len() < k {
                continue;
            }
            if dictionary.contains(sk.get_minimizer_no_hashed()) {
                let start = sk.superkmer.start();
                if start >= covered.len() {
                    continue;
                }
                let raw_end = sk.superkmer.end();
                if raw_end < k {
                    continue;
                }
                let end_exclusive = (raw_end - k + 1).min(covered.len());
                if start < end_exclusive {
                    covered[start..end_exclusive].fill(true);
                }
            }
        }
    }
}

fn stream_fasta_in_chunks<F>(path: &Path, chunk_size: usize, mut on_chunk: F) -> Result<()>
where
    F: FnMut(Vec<SequenceRecord>) -> Result<()>,
{
    let mut chunk = Vec::with_capacity(chunk_size);
    stream_fasta_records(path, |header, seq| {
        chunk.push(SequenceRecord {
            header,
            sequence: seq,
        });
        if chunk.len() >= chunk_size {
            let batch = std::mem::take(&mut chunk);
            on_chunk(batch)?;
        }
        Ok(())
    })?;
    if !chunk.is_empty() {
        on_chunk(chunk)?;
    }
    Ok(())
}

fn stream_fasta_records<F>(path: &Path, mut on_record: F) -> Result<()>
where
    F: FnMut(String, Vec<u8>) -> Result<()>,
{
    let mut reader = open_sequence_reader(path)?;
    let mut header: Option<String> = None;
    let mut sequence: Vec<u8> = Vec::new();
    let mut line = String::new();
    let mut line_number = 0usize;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        line_number += 1;
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() || trimmed.starts_with(';') || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('>') {
            if let Some(prev_header) = header.take() {
                let seq = std::mem::take(&mut sequence);
                on_record(prev_header, seq)?;
            }
            header = Some(rest.trim().to_owned());
            continue;
        }

        ensure!(
            header.is_some(),
            "FASTA sequence line encountered before header at line {}",
            line_number
        );
        append_sanitized_line(trimmed, &mut sequence);
    }

    if let Some(prev_header) = header {
        on_record(prev_header, sequence)?;
    }

    Ok(())
}

fn ensure_path_exists(path: &Path, label: &str) -> Result<()> {
    std::fs::metadata(path)
        .with_context(|| format!("Unable to read {} '{}'", label, path.display()))?;
    Ok(())
}

fn open_sequence_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("Failed to open FASTA file {:?}", path))?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let reader: Box<dyn BufRead> = match ext.as_str() {
        "gz" | "gzip" => {
            let decoder = MultiGzDecoder::new(file);
            Box::new(BufReader::new(decoder))
        }
        "zst" | "zstd" => {
            let decoder = ZstdDecoder::new(file)
                .with_context(|| format!("Failed to decode zstd file {:?}", path))?;
            Box::new(BufReader::new(decoder))
        }
        _ => Box::new(BufReader::new(file)),
    };
    Ok(reader)
}

fn count_kmers_in_dictionary<const N: usize>(
    sequence: &[u8],
    k: usize,
    m: usize,
    dictionary: &PartitionedMinimizerSet,
    ocm_config: Option<&OcmConfig>,
) -> u64 {
    if sequence.len() < k {
        return 0;
    }
    let mut covered = vec![false; sequence.len() - k + 1];
    mark_covered_kmers::<N>(sequence, k, m, dictionary, &mut covered, ocm_config);
    covered.into_iter().filter(|hit| *hit).count() as u64
}

fn generate_random_sequence(len: usize) -> Vec<u8> {
    const ALPHABET: [u8; 4] = [b'A', b'C', b'G', b'T'];
    let mut rng = rand::rng();
    (0..len)
        .map(|_| {
            let idx = rng.random_range(0..ALPHABET.len());
            ALPHABET[idx]
        })
        .collect()
}

fn append_sanitized_line(line: &str, buffer: &mut Vec<u8>) {
    buffer.reserve(line.len());
    for byte in line.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        buffer.push(sanitize_base(byte));
    }
}

fn sanitize_base(byte: u8) -> u8 {
    let upper = byte.to_ascii_uppercase();
    match upper {
        b'A' | b'C' | b'G' | b'T' => upper,
        b'U' => b'T',
        b'N' => b'A',
        _ => b'A',
    }
}

fn fmt_num<T>(value: T) -> String
where
    T: ToFormattedString,
{
    value.to_formatted_string(&Locale::en)
}
