//! Zstandard frame compressor.
//!
//! Produces valid zstd frames decompressible by any standard decoder.
//! Uses greedy hash-based matching (equivalent to zstd level 1).

use super::constants::*;

/// Compress data into a zstd frame.
///
/// `level` controls the compression strategy:
/// - 0: no compression (raw blocks, fastest)
/// - 1-2: greedy matching (fast, hash_log=14)
/// - 3-5: lazy matching (better ratio, hash_log=15)
/// - 6-8: lazy matching + deeper search (hash_log=16)
/// - 9-11: lazy matching + deepest search (hash_log=17)
///
/// Returns a valid zstd frame decompressible by any conformant decoder.
pub fn compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    write_frame_header(&mut out, data.len() as u64);

    if data.is_empty() {
        write_raw_block(&mut out, &[], true);
        return out;
    }

    let blocks: Vec<&[u8]> = data.chunks(ZSTD_BLOCKSIZE_MAX).collect();
    let n_blocks = blocks.len();

    if level <= 0 {
        // Level 0: raw blocks only
        for (i, block) in blocks.iter().enumerate() {
            write_raw_block(&mut out, block, i == n_blocks - 1);
        }
        return out;
    }

    // Level 1+: compress blocks (optionally in parallel)
    let params = MatchParams::from_level(level);

    #[cfg(feature = "parallel")]
    let compressed_blocks: Vec<Option<Vec<u8>>> = {
        use rayon::prelude::*;
        blocks.par_iter()
            .map(|block| compress_block(block, &params))
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let compressed_blocks: Vec<Option<Vec<u8>>> = blocks.iter()
        .map(|block| compress_block(block, &params))
        .collect();

    for (i, compressed) in compressed_blocks.iter().enumerate() {
        let is_last = i == n_blocks - 1;
        match compressed {
            Some(c) if c.len() < blocks[i].len() => {
                write_compressed_block(&mut out, c, is_last);
            }
            _ => write_raw_block(&mut out, blocks[i], is_last),
        }
    }

    out
}

/// Convenience wrapper.
pub fn compress_to_vec(data: &[u8]) -> Vec<u8> {
    compress(data, 1)
}

// =========================================================================
// Frame header
// =========================================================================

fn write_frame_header(out: &mut Vec<u8>, content_size: u64) {
    // Magic number (LE)
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

    // Frame_Header_Descriptor:
    // bit 7-6: Frame_Content_Size_flag (determines FCS field size)
    // bit 5:   Single_Segment_flag (1 = no Window_Descriptor)
    // bit 4:   unused
    // bit 3:   reserved
    // bit 2:   Content_Checksum_flag (0 = no checksum)
    // bit 1-0: Dictionary_ID_flag (0 = no dict)

    let (fcs_flag, fcs_bytes) = if content_size <= 255 {
        (0u8, 1) // 1 byte FCS (but flag 0 means 0 bytes normally...)
    } else if content_size <= 65535 + 256 {
        (1u8, 2) // 2 bytes
    } else if content_size <= u32::MAX as u64 {
        (2u8, 4)
    } else {
        (3u8, 8)
    };

    // For single-segment, FCS_flag=0 means 1 byte (when single_segment=1)
    let single_segment = 1u8; // always single-segment for simplicity
    let descriptor = (fcs_flag << 6) | (single_segment << 5);
    out.push(descriptor);

    // No Window_Descriptor (single_segment = 1)

    // Frame_Content_Size
    match fcs_bytes {
        1 => out.push(content_size as u8),
        2 => out.extend_from_slice(&((content_size - 256) as u16).to_le_bytes()),
        4 => out.extend_from_slice(&(content_size as u32).to_le_bytes()),
        8 => out.extend_from_slice(&content_size.to_le_bytes()),
        _ => {}
    }
}

// =========================================================================
// Block writing
// =========================================================================

fn write_raw_block(out: &mut Vec<u8>, data: &[u8], is_last: bool) {
    let header = (is_last as u32)
        | ((BLOCK_TYPE_RAW as u32) << 1)
        | ((data.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.extend_from_slice(data);
}

fn write_compressed_block(out: &mut Vec<u8>, compressed: &[u8], is_last: bool) {
    let header = (is_last as u32)
        | ((BLOCK_TYPE_COMPRESSED as u32) << 1)
        | ((compressed.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.extend_from_slice(compressed);
}

// =========================================================================
// Block compression (greedy matching + raw literals + predefined FSE)
// =========================================================================

/// A sequence: (literal_length, offset_value, match_length).
/// `off` is the raw back-reference distance.
/// After repeat offset resolution, it becomes an "offset value" for encoding.
struct Sequence {
    ll: u32,
    off: u32,  // raw back-reference distance
    ml: u32,   // actual match length (>= ZSTD_MINMATCH)
}

/// Offset value after repeat-offset resolution.
/// In zstd, offset_value 1/2/3 = repeat offsets, >3 = new offset + 3.
struct EncodedSequence {
    ll: u32,
    of_value: u32, // offset value for encoding (1..3 = repcode, >3 = new)
    ml: u32,
}

/// Match finder parameters, derived from compression level.
struct MatchParams {
    hash_log: u32,
    lazy_depth: u32,   // 0=greedy, 1=lazy, 2=lazy2
    search_depth: u32, // hash chain search depth
}

impl MatchParams {
    fn from_level(level: i32) -> Self {
        match level {
            0..=2 => Self { hash_log: 14, lazy_depth: 0, search_depth: 4 },
            3..=5 => Self { hash_log: 15, lazy_depth: 1, search_depth: 16 },
            6..=8 => Self { hash_log: 16, lazy_depth: 1, search_depth: 64 },
            _     => Self { hash_log: 17, lazy_depth: 1, search_depth: 128 },
        }
    }
}

fn compress_block(data: &[u8], params: &MatchParams) -> Option<Vec<u8>> {
    let sequences = find_matches(data, params);

    if sequences.is_empty() {
        return None;
    }

    // === 1. Resolve repeat offsets per zstd spec ===
    let encoded_seqs = resolve_repeat_offsets(&sequences);

    // === 2. Collect literals ===
    let mut literals = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    for seq in &sequences {
        literals.extend_from_slice(&data[pos..pos + seq.ll as usize]);
        pos += seq.ll as usize + seq.ml as usize;
    }
    literals.extend_from_slice(&data[pos..]);

    // === 3. Encode block ===
    let mut block = Vec::with_capacity(data.len());

    // Literals section: try Huffman, fall back to raw
    let mut used_huf = false;
    if literals.len() >= 64 {
        if let Some(huf) = encode_literals_huffman(&literals) {
            if huf.len() < literals.len() {
                block.extend_from_slice(&huf);
                used_huf = true;
            }
        }
    }
    if !used_huf {
        encode_literals_raw(&mut block, &literals);
    }

    // Sequences section
    encode_sequences_section(&mut block, &encoded_seqs);

    Some(block)
}

/// Resolve repeat offsets per zstd spec (RFC 8878 §3.1.2.5).
///
/// Encoder chooses offset_value for each sequence:
/// - If raw_offset matches a repeat offset → use repcode (1/2/3)
/// - Otherwise → use raw_offset + 3
///
/// After each sequence, the repeat offset table is updated:
/// - New offset → shift: rep = [new, old_rep0, old_rep1]
/// - Repeat 1 → no change
/// - Repeat 2 → rotate: rep = [rep1, rep0, rep2]
/// - Repeat 3 → rotate: rep = [rep2, rep0, rep1]
fn resolve_repeat_offsets(sequences: &[Sequence]) -> Vec<EncodedSequence> {
    let mut rep = [1u32, 4, 8]; // initial repeat offsets
    let mut out = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let raw_off = seq.off;
        let of_value;

        if seq.ll > 0 {
            // Normal case (ll > 0)
            if raw_off == rep[0] {
                of_value = 1;
                // rep unchanged
            } else if raw_off == rep[1] {
                of_value = 2;
                // rotate: [rep1, rep0, rep2]
                rep = [rep[1], rep[0], rep[2]];
            } else if raw_off == rep[2] {
                of_value = 3;
                // rotate: [rep2, rep0, rep1]
                rep = [rep[2], rep[0], rep[1]];
            } else {
                of_value = raw_off + 3;
                // shift: [new, old0, old1]
                rep = [raw_off, rep[0], rep[1]];
            }
        } else {
            // ll == 0: offsets are shifted by 1
            // of_value 1 → rep[1], of_value 2 → rep[2], of_value 3 → rep[0]-1
            if raw_off == rep[1] {
                of_value = 1;
                rep = [rep[1], rep[0], rep[2]];
            } else if raw_off == rep[2] {
                of_value = 2;
                rep = [rep[2], rep[0], rep[1]];
            } else if raw_off == rep[0].wrapping_sub(1) && rep[0] > 1 {
                of_value = 3;
                rep = [rep[0] - 1, rep[0], rep[1]];
            } else {
                of_value = raw_off + 3;
                rep = [raw_off, rep[0], rep[1]];
            }
        }

        out.push(EncodedSequence { ll: seq.ll, of_value, ml: seq.ml });
    }

    out
}

/// Hash chain match finder with configurable lazy depth.
///
/// - `lazy_depth=0`: greedy — take first match (level 1-2)
/// - `lazy_depth=1`: lazy — check next position for better match (level 3+)
fn find_matches(data: &[u8], params: &MatchParams) -> Vec<Sequence> {
    if data.len() < ZSTD_MINMATCH + 1 {
        return vec![];
    }

    let hash_size = 1usize << params.hash_log;
    let hash_mask = (hash_size - 1) as u32;
    let mut hash_table = vec![0u32; hash_size];
    let mut chain = vec![0u32; data.len()];
    let mut sequences = Vec::new();
    let mut anchor = 0usize;
    let mut ip = 0usize;

    while ip + ZSTD_MINMATCH < data.len() {
        // Try to find a match at current position
        let best = find_best_at(data, ip, &hash_table, &chain, hash_mask, params.search_depth);

        if let Some((offset, match_len)) = best {
            let mut final_off = offset;
            let mut final_len = match_len;
            let mut final_ip = ip;

            // Lazy matching: check if next position gives a better match
            if params.lazy_depth >= 1 && ip + 1 + ZSTD_MINMATCH < data.len() {
                insert_hash(&mut hash_table, &mut chain, data, ip, hash_mask);
                if let Some((off2, len2)) = find_best_at(data, ip + 1, &hash_table, &chain, hash_mask, params.search_depth) {
                    if len2 > final_len + 1 {
                        final_off = off2;
                        final_len = len2;
                        final_ip = ip + 1;
                    }
                }
            }

            let ll = (final_ip - anchor) as u32;
            sequences.push(Sequence { ll, off: final_off as u32, ml: final_len as u32 });

            // Insert all positions within the match for future matching
            for p in ip..std::cmp::min(final_ip + final_len, data.len().saturating_sub(ZSTD_MINMATCH)) {
                insert_hash(&mut hash_table, &mut chain, data, p, hash_mask);
            }

            ip = final_ip + final_len;
            anchor = ip;
        } else {
            insert_hash(&mut hash_table, &mut chain, data, ip, hash_mask);
            ip += 1;
        }
    }

    sequences
}

/// Insert position into hash chain.
#[inline]
fn insert_hash(hash_table: &mut [u32], chain: &mut [u32], data: &[u8], pos: usize, mask: u32) {
    if pos + 4 > data.len() { return; }
    let h = hash4(&data[pos..], mask);
    chain[pos] = hash_table[h];
    hash_table[h] = pos as u32;
}

/// Find the best match at `pos` by walking the hash chain.
fn find_best_at(
    data: &[u8], pos: usize,
    hash_table: &[u32], chain: &[u32],
    mask: u32, max_depth: u32,
) -> Option<(usize, usize)> {
    if pos + ZSTD_MINMATCH > data.len() { return None; }
    let h = hash4(&data[pos..], mask);
    let mut candidate = hash_table[h] as usize;
    let mut best_len = ZSTD_MINMATCH - 1;
    let mut best_off = 0;

    for _ in 0..max_depth {
        if candidate >= pos || pos - candidate > (1 << 24) { break; }
        if candidate + ZSTD_MINMATCH > data.len() { break; }

        // Quick 4-byte check
        if data[candidate..candidate + 4] == data[pos..pos + 4] {
            let ml = common_prefix_len(&data[candidate..], &data[pos..]);
            if ml > best_len {
                best_len = ml;
                best_off = pos - candidate;
            }
        }

        let next = chain[candidate] as usize;
        if next >= candidate { break; }
        candidate = next;
    }

    if best_len >= ZSTD_MINMATCH { Some((best_off, best_len)) } else { None }
}

/// Fast common prefix length using 8-byte chunks.
#[inline]
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let max = std::cmp::min(a.len(), b.len());
    let mut i = 0;
    while i + 8 <= max {
        let va = u64::from_le_bytes(a[i..i+8].try_into().unwrap());
        let vb = u64::from_le_bytes(b[i..i+8].try_into().unwrap());
        if va != vb {
            return i + ((va ^ vb).trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < max && a[i] == b[i] { i += 1; }
    i
}

/// 4-byte multiplicative hash, result masked to table size.
#[inline]
fn hash4(data: &[u8], mask: u32) -> usize {
    let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    (v.wrapping_mul(0x9E3779B1) as usize) & (mask as usize)
}

// =========================================================================
// Huffman literal compression
// =========================================================================

/// Build Huffman codes, encode tree + streams. Returns None if Huffman doesn't help.
fn encode_literals_huffman(literals: &[u8]) -> Option<Vec<u8>> {
    // Count frequencies
    let mut counts = [0u32; 256];
    let mut max_sym = 0u8;
    for &b in literals {
        counts[b as usize] += 1;
        if b > max_sym { max_sym = b; }
    }
    let n_used = counts.iter().filter(|&&c| c > 0).count();
    if n_used < 2 { return None; }

    // Build length-limited Huffman (max 11 bits)
    let (codes, max_bits) = build_huffman_codes(&counts, max_sym as usize)?;

    // Encode tree description (weights packed as 4-bit pairs)
    let tree_desc = encode_huffman_tree(&codes, max_bits, max_sym as usize);
    if tree_desc.is_empty() { return None; }

    // Encode streams: single stream for < 1KB, 4 streams for >= 1KB
    let use_4 = literals.len() >= 1024;
    let streams = if use_4 {
        encode_huf_4streams(literals, &codes)
    } else {
        encode_huf_1stream(literals, &codes)
    };

    let regen = literals.len();
    let comp = tree_desc.len() + streams.len();
    let lh_size = 3 + (regen >= 1024) as usize + (regen >= 16384) as usize;

    let mut out = Vec::with_capacity(lh_size + comp);
    let htype = LIT_TYPE_COMPRESSED as u32;

    match lh_size {
        3 => {
            // bit[1:0]=type(2), bit[2]=streams_flag, bit[3]=0, bit[13:4]=regen, bit[23:14]=comp
            let sf = if use_4 { 1u32 } else { 0u32 };
            let lhc = htype | (sf << 2) | ((regen as u32) << 4) | ((comp as u32) << 14);
            out.extend_from_slice(&lhc.to_le_bytes()[..3]);
        }
        4 => {
            let lhc = htype | (2u32 << 2) | ((regen as u32) << 4) | ((comp as u32) << 18);
            out.extend_from_slice(&lhc.to_le_bytes()[..4]);
        }
        _ => {
            let lhc = htype | (3u32 << 2) | ((regen as u32) << 4) | ((comp as u32) << 22);
            out.extend_from_slice(&lhc.to_le_bytes()[..4]);
            out.push((comp >> 10) as u8);
        }
    }

    out.extend_from_slice(&tree_desc);
    out.extend_from_slice(&streams);
    Some(out)
}

/// Build canonical Huffman codes, max 11 bits. Returns None if can't build valid codes.
fn build_huffman_codes(counts: &[u32; 256], max_sym: usize) -> Option<([(u32, u8); 256], u8)> {
    const MAX_BITS: u8 = 11;
    let mut syms: Vec<(u32, u8)> = (0..=max_sym)
        .filter(|&s| counts[s] > 0)
        .map(|s| (counts[s], s as u8))
        .collect();
    syms.sort();
    let n = syms.len();
    if n < 2 { return None; }

    // Assign lengths from Shannon entropy, clamp to [1, MAX_BITS]
    let total: f64 = syms.iter().map(|&(c, _)| c as f64).sum();
    let mut lengths = [0u8; 256];
    for &(freq, sym) in &syms {
        let p = freq as f64 / total;
        lengths[sym as usize] = (-p.log2()).ceil().clamp(1.0, MAX_BITS as f64) as u8;
    }

    // Fix Kraft inequality iteratively
    for _ in 0..1000 {
        let kraft: i64 = (0..256).filter(|&s| lengths[s] > 0)
            .map(|s| 1i64 << (MAX_BITS - lengths[s])).sum();
        let target = 1i64 << MAX_BITS;
        if kraft == target { break; }
        if kraft > target {
            // Over-full: lengthen least frequent symbol
            for &(_, sym) in &syms {
                if lengths[sym as usize] < MAX_BITS {
                    lengths[sym as usize] += 1;
                    break;
                }
            }
        } else {
            // Under-full: shorten most frequent symbol
            for &(_, sym) in syms.iter().rev() {
                if lengths[sym as usize] > 1 {
                    lengths[sym as usize] -= 1;
                    break;
                }
            }
        }
    }

    // Verify Kraft
    let kraft: u64 = (0..256).filter(|&s| lengths[s] > 0)
        .map(|s| 1u64 << (MAX_BITS - lengths[s])).sum();
    if !kraft.is_power_of_two() { return None; }

    let max_bits = *lengths.iter().max().unwrap_or(&0);
    let mut bl_count = [0u32; 16];
    for &l in &lengths { if l > 0 { bl_count[l as usize] += 1; } }

    let mut next_code = [0u32; 16];
    for bits in 1..=max_bits as usize {
        next_code[bits] = (next_code[bits - 1] + bl_count[bits - 1]) << 1;
    }

    let mut codes = [(0u32, 0u8); 256];
    for s in 0..256 {
        if lengths[s] > 0 {
            codes[s] = (next_code[lengths[s] as usize], lengths[s]);
            next_code[lengths[s] as usize] += 1;
        }
    }

    Some((codes, max_bits))
}

fn encode_huffman_tree(codes: &[(u32, u8); 256], max_bits: u8, max_sym: usize) -> Vec<u8> {
    if max_bits == 0 { return vec![]; }
    let mut weights: Vec<u8> = (0..=max_sym)
        .map(|s| if codes[s].1 > 0 { max_bits + 1 - codes[s].1 } else { 0 })
        .collect();
    while weights.last() == Some(&0) && weights.len() > 1 { weights.pop(); }
    if !weights.is_empty() { weights.pop(); } // last weight is implicit
    if weights.is_empty() || weights.len() > 255 { return vec![]; }

    // Check all weights fit in 4 bits
    if weights.iter().any(|&w| w > 12) { return vec![]; }

    let num = weights.len();

    if num <= 128 {
        // Direct mode: header = num + 127, packed 4-bit pairs
        let mut desc = Vec::with_capacity(1 + num.div_ceil(2));
        desc.push((num as u8) + 127);
        for pair in weights.chunks(2) {
            let w0 = pair[0];
            let w1 = if pair.len() > 1 { pair[1] } else { 0 };
            desc.push((w0 << 4) | (w1 & 0x0F));
        }
        desc
    } else {
        // >128 weights: FSE-compressed 2-stream interleaved encoding
        match encode_weights_fse(&weights) {
            Some(compressed) if compressed.len() < 127 && compressed.len() < num.div_ceil(2) => {
                // Verify: try decoding the FSE weights to check roundtrip
                let header_byte = compressed.len() as u8;
                let verify = crate::zstd::decode::decode_huf_weights_from_fse(&compressed, header_byte);
                match verify {
                    Ok(decoded_weights) if decoded_weights == weights => {
                        let mut desc = Vec::with_capacity(1 + compressed.len());
                        desc.push(header_byte);
                        desc.extend_from_slice(&compressed);
                        desc
                    }
                    _ => vec![], // FSE roundtrip mismatch, fall back to raw
                }
            }
            _ => vec![],
        }
    }
}

/// FSE-compress weights using 2-stream interleaved encoding.
///
/// Decoder reads: sentinel → init_state(dec1) → init_state(dec2) →
/// loop { dec1.decode + update, dec2.decode + update } until exhausted.
///
/// Encoder writes backward: data pairs → state2 → state1 → sentinel.
/// After bw.finish() (reverse + sentinel), decoder reads correctly.
fn encode_weights_fse(weights: &[u8]) -> Option<Vec<u8>> {
    let mut counts = [0u32; 13];
    let mut max_w = 0u8;
    for &w in weights { counts[w as usize] += 1; if w > max_w { max_w = w; } }
    if max_w == 0 { return None; }

    let table_log = 6u32;
    let table_size = 1u32 << table_log;
    let total = weights.len() as u32;

    // Normalize
    let mut norm = [0i16; 13];
    let mut dist = 0u32;
    for s in 0..=max_w as usize {
        if counts[s] == 0 { continue; }
        norm[s] = std::cmp::max(1, (counts[s] as u64 * table_size as u64 / total as u64) as i16);
        dist += norm[s] as u32;
    }
    while dist > table_size {
        for s in 0..=max_w as usize { if norm[s] > 1 { norm[s] -= 1; dist -= 1; break; } }
    }
    while dist < table_size {
        let best = (0..=max_w as usize).max_by_key(|&s| counts[s]).unwrap_or(0);
        norm[best] += 1; dist += 1;
    }

    let fse = super::fse::FseCTable::build(&norm, max_w as usize, table_log);

    // --- FSE table header (must match decode.rs read_probabilities exactly) ---
    // Decoder: accuracy_log = 5 + get_bits(4)
    //          loop: max_remaining = prob_sum - counter + 1
    //                bits_to_read = highest_bit_set(max_remaining)
    //                read bits_to_read, apply low_threshold logic
    //                prob = value - 1
    let mut hdr = Vec::with_capacity(16);
    let mut bb: u64 = (table_log - 5) as u64;
    let mut bp = 4u32;
    let prob_sum = table_size;
    let mut counter = 0u32;

    for s in 0..=max_w as usize {
        if counter >= prob_sum { break; }
        let prob = norm[s] as i32;
        let value = (prob + 1) as u32; // prob=-1 → value=0, prob=0 → value=1, etc.

        let max_remaining = prob_sum - counter + 1;
        let bits_to_read = 32 - max_remaining.leading_zeros(); // = highest_bit_set(max_remaining)

        let low_threshold = ((1u32 << bits_to_read) - 1) - max_remaining;
        let mask = (1u32 << (bits_to_read - 1)) - 1;

        if value < low_threshold {
            // Case 1: decoder reads (btr-1) bits, gets value directly
            bb |= (value as u64) << bp;
            bp += bits_to_read - 1;
        } else if value <= mask {
            // Case 3: decoder reads btr bits, unchecked = value, value <= mask, value >= low_threshold
            bb |= (value as u64) << bp;
            bp += bits_to_read;
        } else {
            // Case 2: decoder reads btr bits, unchecked > mask, value = unchecked - low_threshold
            let encoded = value + low_threshold;
            bb |= (encoded as u64) << bp;
            bp += bits_to_read;
        }

        while bp >= 8 { hdr.push(bb as u8); bb >>= 8; bp -= 8; }

        if prob > 0 { counter += prob as u32; }
        else if prob == -1 { counter += 1; }
        // prob == 0 doesn't contribute to counter
    }
    if bp > 0 { hdr.push(bb as u8); }

    // --- 2-stream interleaved backward bitstream ---
    let _n = weights.len();
    let mut bw = super::bitstream::BackwardBitWriter::new();

    // Decoder reads:
    //   1. sentinel (1-bit + padding zeros)
    //   2. dec1.init_state (table_log bits)
    //   3. dec2.init_state (table_log bits)
    //   4. loop: dec1.decode_symbol, dec1.update_state(num_bits),
    //            dec2.decode_symbol, dec2.update_state(num_bits), ...
    //
    // decode_symbol reads nothing (just returns state.symbol).
    // update_state reads state.num_bits bits → new_state = base_line + bits_read.
    //
    // Encoder must write in reverse:
    //   First: update_state bits for early symbols
    //   ...
    //   Last: init states → sentinel
    //
    // The FSE CTable.encode_symbol(state, sym) returns:
    //   (bits_out, num_bits, new_state)
    // where bits_out are the low bits of the OLD state.
    // This corresponds to what update_state reads.

    // Split: stream1 = w[0],w[2],w[4],... stream2 = w[1],w[3],w[5],...
    let stream1: Vec<u8> = weights.iter().step_by(2).copied().collect();
    let stream2: Vec<u8> = weights.iter().skip(1).step_by(2).copied().collect();

    // Init states from last symbols of each stream
    let mut st1 = fse.init_state(*stream1.last().unwrap() as usize);
    let mut st2 = fse.init_state(*stream2.last().unwrap() as usize);

    // Encode in reverse order, alternating stream2 then stream1
    // (because decoder reads stream1 first after init)
    let len1 = stream1.len();
    let len2 = stream2.len();

    // Process from second-to-last down to first
    // Decoder processes: s1[0], s2[0], s1[1], s2[1], ...
    // Last s1 and s2 are init'd, not update'd.
    // So encode s1[len1-2]..s1[0] and s2[len2-2]..s2[0]

    // Interleave: for each pair index going backward
    let max_idx = std::cmp::max(len1, len2);
    for i in (0..max_idx - 1).rev() {
        // stream2 update comes before stream1 in bitstream
        // (decoder reads s1 update first, then s2 update)
        // backward → write s2 first (will be read second)
        if i < len2 - 1 {
            let (bits, nb, ns) = fse.encode_symbol(st2, stream2[i] as usize);
            bw.add_bits(bits as u64, nb);
            bw.flush_bits();
            st2 = ns;
        }
        if i < len1 - 1 {
            let (bits, nb, ns) = fse.encode_symbol(st1, stream1[i] as usize);
            bw.add_bits(bits as u64, nb);
            bw.flush_bits();
            st1 = ns;
        }
    }

    // Write init states: dec2 state then dec1 state
    // (dec1 is read first from bitstream front → written last)
    bw.add_bits(st2 as u64, table_log);
    bw.flush_bits();
    bw.add_bits(st1 as u64, table_log);
    bw.flush_bits();

    let bitstream = bw.finish();
    let mut out = hdr;
    out.extend_from_slice(&bitstream);
    Some(out)
}

/// Encode one Huffman stream (symbols in reverse, padded with sentinel bit).
fn encode_huf_1stream(data: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
    let mut bw = super::bitstream::BackwardBitWriter::new();
    // Encode symbols in reverse (backward bitstream convention)
    for &sym in data.iter().rev() {
        let (code, nb) = codes[sym as usize];
        if nb == 0 { continue; }
        bw.add_bits(code as u64, nb as u32);
        bw.flush_bits();
    }
    bw.finish() // adds sentinel, no reverse needed
}

fn encode_huf_4streams(data: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
    let q = data.len().div_ceil(4);
    let ends = [q, std::cmp::min(q*2, data.len()), std::cmp::min(q*3, data.len()), data.len()];
    let starts = [0, q, ends[1], ends[2]];

    let c: Vec<Vec<u8>> = (0..4).map(|i| encode_huf_1stream(&data[starts[i]..ends[i]], codes)).collect();

    let mut out = Vec::with_capacity(6 + c.iter().map(|v| v.len()).sum::<usize>());
    // Jump table: sizes of first 3 streams (u16 LE each)
    for i in 0..3 { out.extend_from_slice(&(c[i].len() as u16).to_le_bytes()); }
    for stream in &c { out.extend_from_slice(stream); }
    out
}

// =========================================================================
// Literals section encoding (Raw mode)
// =========================================================================

fn encode_literals_raw(out: &mut Vec<u8>, literals: &[u8]) {
    let size = literals.len();

    if size <= 31 {
        // 1-byte header: type=0 (raw), size in 5 bits
        out.push(LIT_TYPE_RAW | ((size as u8) << 3));
    } else if size <= 4095 {
        // 2-byte header
        let h = (LIT_TYPE_RAW as u16) | (1 << 2) | ((size as u16) << 4);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        // 3-byte header
        let h = (LIT_TYPE_RAW as u32) | (3 << 2) | ((size as u32) << 4);
        out.extend_from_slice(&h.to_le_bytes()[..3]);
    }

    out.extend_from_slice(literals);
}

// =========================================================================
// Sequences section encoding using exact C-compatible FSE tables
// =========================================================================

fn encode_sequences_section(out: &mut Vec<u8>, sequences: &[EncodedSequence]) {
    let nb_seq = sequences.len();

    // Number of sequences header
    if nb_seq < 128 {
        out.push(nb_seq as u8);
    } else if nb_seq < 0x7F00 {
        out.push(((nb_seq >> 8) as u8) + 128);
        out.push(nb_seq as u8);
    } else {
        out.push(255);
        out.extend_from_slice(&((nb_seq - 0x7F00) as u16).to_le_bytes());
    }

    if nb_seq == 0 { return; }

    // Symbol compression modes: all predefined
    out.push(0x00);

    // Build predefined FSE tables (exact C-compatible implementation)
    let ll_table = super::fse::FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
    let ml_table = super::fse::FseCTable::build(&ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);
    let of_table = super::fse::FseCTable::build(
        &OF_DEFAULT_NORM, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG,
    );

    // Convert sequences to codes + extra bit values
    let mut ll_codes_v = Vec::with_capacity(nb_seq);
    let mut ml_codes_v = Vec::with_capacity(nb_seq);
    let mut off_codes_v = Vec::with_capacity(nb_seq);
    let mut ll_values = Vec::with_capacity(nb_seq);
    let mut ml_values = Vec::with_capacity(nb_seq);
    let mut off_values = Vec::with_capacity(nb_seq);

    for seq in sequences {
        let llc = ll_code(seq.ll);
        let ml_base = seq.ml - ZSTD_MINMATCH as u32;
        let mlc = ml_code(ml_base);
        // of_value already has repeat offset resolution applied
        let ofc = off_code(seq.of_value);

        ll_codes_v.push(llc);
        ml_codes_v.push(mlc);
        off_codes_v.push(ofc);
        ll_values.push(seq.ll - LL_BASE[llc as usize]);
        ml_values.push(seq.ml - ML_BASE[mlc as usize]);
        off_values.push(if ofc > 0 { seq.of_value - (1u32 << ofc) } else { 0 });
    }

    // Encode with the exact C-compatible FSE sequence encoder
    let bitstream = super::fse::encode_sequences(
        &ll_table, &of_table, &ml_table,
        &ll_codes_v, &off_codes_v, &ml_codes_v,
        &ll_values, &ml_values, &off_values,
    );
    out.extend_from_slice(&bitstream);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_empty() {
        let compressed = compress(&[], 1);
        assert!(compressed.len() >= 5); // magic + header + empty block
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    #[test]
    fn compress_small() {
        let data = b"hello world";
        let compressed = compress(data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert!(compressed.len() > 5);
    }

    #[test]
    fn compress_repetitive() {
        let data = vec![42u8; 4096];
        let compressed = compress(&data, 1);
        // Valid zstd frame (raw blocks are larger than input due to framing)
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    #[test]
    fn compress_real_data() {
        let data: Vec<u8> = (0..1024u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    /// Golden test: roundtrip through our compressor → our decompressor.
    #[test]
    fn roundtrip_self_contained() {
        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros", vec![0u8; 4096]),
            ("sequential", (0..4096u32).flat_map(|i| i.to_le_bytes()).collect()),
            ("f32_data", (0..256u32).flat_map(|i| (i as f32 * 1.5).to_le_bytes()).collect()),
            ("repetitive", b"hello world! ".repeat(100)),
            ("small", b"abc".to_vec()),
        ];

        for (name, data) in &test_cases {
            let compressed = compress(data, 1);
            let decompressed = crate::zstd::decompress(&compressed)
                .unwrap_or_else(|e| panic!("{}: decompress failed: {}", name, e));

            assert_eq!(decompressed.len(), data.len(), "{}: length mismatch", name);
            assert_eq!(&decompressed, data, "{}: data mismatch", name);
        }
    }
}

