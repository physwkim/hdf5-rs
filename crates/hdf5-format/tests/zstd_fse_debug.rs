#[test]
fn tiny_fse_weight_test() {
    // 6 weights, all same (value 2) → should be trivially compressible
    // But > 128 is needed to trigger FSE. Use 130 uniform weights.
    let data: Vec<u8> = (0..50)
        .flat_map(|_| (0..130u8).collect::<Vec<_>>())
        .collect();
    let c = hdf5_format::zstd::compress(&data, 1);
    let d = hdf5_format::zstd::decompress(&c).unwrap();
    assert_eq!(d, data);
    eprintln!(
        "130-sym uniform: {} -> {} ({:.1}%)",
        data.len(),
        c.len(),
        (1.0 - c.len() as f64 / data.len() as f64) * 100.0
    );

    // Non-uniform 200 symbols with skewed distribution
    let mut data2 = Vec::new();
    for _ in 0..100 {
        for b in 0..200u8 {
            data2.push(b);
        }
    }
    // Add extra of first 10 symbols
    for _ in 0..1000 {
        for b in 0..10u8 {
            data2.push(b);
        }
    }
    let c2 = hdf5_format::zstd::compress(&data2, 1);
    let d2 = hdf5_format::zstd::decompress(&c2).unwrap();
    assert_eq!(d2, data2);
    eprintln!(
        "200-sym skewed: {} -> {} ({:.1}%)",
        data2.len(),
        c2.len(),
        (1.0 - c2.len() as f64 / data2.len() as f64) * 100.0
    );

    // 256 symbols, heavily skewed
    let mut data3 = Vec::new();
    for _ in 0..50 {
        for b in 0u16..256 {
            data3.push(b as u8);
        }
    }
    for _ in 0..5000 {
        data3.push(0);
        data3.push(1);
    } // heavy skew
    let c3 = hdf5_format::zstd::compress(&data3, 1);
    let d3 = hdf5_format::zstd::decompress(&c3).unwrap();
    assert_eq!(d3, data3);
    eprintln!(
        "256-sym skewed: {} -> {} ({:.1}%)",
        data3.len(),
        c3.len(),
        (1.0 - c3.len() as f64 / data3.len() as f64) * 100.0
    );
}
