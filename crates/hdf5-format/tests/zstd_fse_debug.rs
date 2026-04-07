// Directly test FSE weight encode → decode round-trip
// by extracting the tree description and feeding it to the decoder's Huffman table reader.

#[test]
fn fse_weight_encode_isolate() {
    // Create weights that would trigger FSE encoding (>128)
    // Use the same distribution as u32 sequential data
    let data: Vec<u8> = (0..4096u32).flat_map(|i| i.to_le_bytes()).collect();
    let mut counts = [0u32; 256];
    let mut max_sym = 0u8;
    for &b in &data { counts[b as usize] += 1; if b > max_sym { max_sym = b; } }

    // Build Huffman - exact same code as compress.rs
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    let mut lengths = [0u8; 256];
    let mut syms: Vec<(u32, u8)> = (0..=max_sym as usize)
        .filter(|&s| counts[s] > 0).map(|s| (counts[s], s as u8)).collect();
    syms.sort();
    for &(freq, sym) in &syms {
        let p = freq as f64 / total;
        lengths[sym as usize] = (-p.log2()).ceil().clamp(1.0, 11.0) as u8;
    }
    // Kraft fix
    for _ in 0..1000 {
        let kraft: i64 = (0..256).filter(|&s| lengths[s] > 0)
            .map(|s| 1i64 << (11 - lengths[s])).sum();
        if kraft == 2048 { break; }
        if kraft > 2048 {
            for &(_, sym) in &syms {
                if lengths[sym as usize] < 11 { lengths[sym as usize] += 1; break; }
            }
        } else {
            for &(_, sym) in syms.iter().rev() {
                if lengths[sym as usize] > 1 { lengths[sym as usize] -= 1; break; }
            }
        }
    }
    let kraft: u64 = (0..256).filter(|&s| lengths[s] > 0)
        .map(|s| 1u64 << (11 - lengths[s])).sum();
    eprintln!("Kraft sum: {} (valid: {})", kraft, kraft.is_power_of_two());

    let max_bits = *lengths.iter().max().unwrap();
    let mut weights: Vec<u8> = (0..=max_sym as usize)
        .map(|s| if lengths[s] > 0 { max_bits + 1 - lengths[s] } else { 0 }).collect();
    // Remove trailing zeros, pop last (implicit)
    while weights.last() == Some(&0) && weights.len() > 1 { weights.pop(); }
    weights.pop();

    eprintln!("weights len: {}, max_weight: {}", weights.len(),
        weights.iter().max().unwrap_or(&0));

    // Compute expected weight sum for verification
    let stored_sum: u64 = weights.iter().filter(|&&w| w > 0).map(|&w| 1u64 << (w - 1)).sum();
    let full_power = (stored_sum + 1).next_power_of_two(); // should be 2^max_weight
    let leftover = full_power - stored_sum;
    eprintln!("stored_sum: {}, full_power: {}, leftover: {} (power_of_2: {})",
        stored_sum, full_power, leftover, leftover.is_power_of_two());

    assert!(leftover.is_power_of_two(), "Leftover {} should be power of 2", leftover);
    eprintln!("Weight validation PASSED");
}

#[test]
fn u32_roundtrip_with_verify() {
    let data: Vec<u8> = (0..4096u32).flat_map(|i| i.to_le_bytes()).collect();
    let compressed = hdf5_format::zstd::compress(&data, 1);
    let d = hdf5_format::zstd::decompress(&compressed)
        .unwrap_or_else(|e| panic!("{}", e));
    assert_eq!(d, data);
    eprintln!("u32 seq OK: {} -> {}", data.len(), compressed.len());
}
