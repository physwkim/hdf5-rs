//! Filter pipeline message (type 0x0B).
//!
//! Describes a pipeline of data filters (compression, checksumming, etc.)
//! applied to chunk data.
//!
//! Version 2 binary layout:
//! ```text
//! Byte 0: version = 2
//! Byte 1: number of filters
//! For each filter:
//!   filter_id:     u16 LE
//!   [if filter_id >= 256: name_length: u16 LE, name: NUL-padded string]
//!   flags:         u16 LE
//!   num_cd_values: u16 LE
//!   cd_values:     num_cd_values * u32 LE
//! ```

use crate::{FormatError, FormatResult};

/// Well-known filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

/// A single filter in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// Filter identifier (1 = deflate, 2 = shuffle, etc.).
    pub id: u16,
    /// Filter flags. Bit 0: filter is optional (0 = mandatory).
    pub flags: u16,
    /// Client data values (filter-specific parameters).
    pub cd_values: Vec<u32>,
}

/// A pipeline of data filters applied to chunk data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterPipeline {
    /// Ordered list of filters in the pipeline.
    pub filters: Vec<Filter>,
}

impl FilterPipeline {
    /// Create a pipeline with a single deflate (gzip) filter.
    ///
    /// `level` is the compression level (0-9). A level of 0 means no
    /// compression, 9 is maximum compression. The HDF5 default is 6.
    pub fn deflate(level: u32) -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_DEFLATE,
                flags: 0, // mandatory
                cd_values: vec![level],
            }],
        }
    }

    /// Create an empty pipeline (no filters).
    pub fn none() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Encode as a version-2 filter pipeline message.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);

        // Version
        buf.push(2);
        // Number of filters
        buf.push(self.filters.len() as u8);

        for f in &self.filters {
            // Filter ID
            buf.extend_from_slice(&f.id.to_le_bytes());

            // For filter IDs >= 256 (user-defined), a name string follows.
            // Predefined filters (< 256) have no name.
            if f.id >= 256 {
                // Name length = 0 (no name for now)
                buf.extend_from_slice(&0u16.to_le_bytes());
            }

            // Flags
            buf.extend_from_slice(&f.flags.to_le_bytes());

            // Number of client data values
            buf.extend_from_slice(&(f.cd_values.len() as u16).to_le_bytes());

            // Client data values
            for &cd in &f.cd_values {
                buf.extend_from_slice(&cd.to_le_bytes());
            }
        }

        buf
    }

    /// Decode a version-2 filter pipeline message.
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != 2 {
            return Err(FormatError::InvalidVersion(version));
        }

        let nfilters = buf[1] as usize;
        let mut pos = 2;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            // filter_id
            if buf.len() < pos + 2 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 2,
                    available: buf.len(),
                });
            }
            let id = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;

            // For user-defined filters (id >= 256), read and skip the name.
            if id >= 256 {
                if buf.len() < pos + 2 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 2,
                        available: buf.len(),
                    });
                }
                let name_len = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                // Name is padded to a multiple of 8 bytes
                let padded_len = (name_len + 7) & !7;
                if buf.len() < pos + padded_len {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + padded_len,
                        available: buf.len(),
                    });
                }
                pos += padded_len;
            }

            // flags + num_cd_values
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let flags = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let num_cd = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
            pos += 2;

            // cd_values
            if buf.len() < pos + num_cd * 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + num_cd * 4,
                    available: buf.len(),
                });
            }
            let mut cd_values = Vec::with_capacity(num_cd);
            for _ in 0..num_cd {
                let v = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
                pos += 4;
                cd_values.push(v);
            }

            filters.push(Filter {
                id,
                flags,
                cd_values,
            });
        }

        Ok((Self { filters }, pos))
    }
}

/// Apply filter pipeline to compress raw chunk data.
///
/// Returns the compressed data. If no filters are configured, returns the
/// input unchanged.
pub fn apply_filters(pipeline: &FilterPipeline, data: &[u8]) -> FormatResult<Vec<u8>> {
    let mut buf = data.to_vec();
    for filter in &pipeline.filters {
        buf = apply_single_filter(filter, &buf, true)?;
    }
    Ok(buf)
}

/// Reverse filter pipeline to decompress raw chunk data.
///
/// Filters are applied in reverse order. Returns the decompressed data.
pub fn reverse_filters(pipeline: &FilterPipeline, data: &[u8]) -> FormatResult<Vec<u8>> {
    let mut buf = data.to_vec();
    for filter in pipeline.filters.iter().rev() {
        buf = apply_single_filter(filter, &buf, false)?;
    }
    Ok(buf)
}

fn apply_single_filter(filter: &Filter, data: &[u8], compress: bool) -> FormatResult<Vec<u8>> {
    match filter.id {
        #[cfg(feature = "deflate")]
        FILTER_DEFLATE => {
            if compress {
                use flate2::write::ZlibEncoder;
                use flate2::Compression;
                use std::io::Write;

                let level = filter.cd_values.first().copied().unwrap_or(6);
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
                encoder.write_all(data).map_err(|e| {
                    FormatError::InvalidData(format!("deflate compress error: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    FormatError::InvalidData(format!("deflate finish error: {}", e))
                })
            } else {
                use flate2::read::ZlibDecoder;
                use std::io::Read;

                let mut decoder = ZlibDecoder::new(data);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out).map_err(|e| {
                    FormatError::InvalidData(format!("deflate decompress error: {}", e))
                })?;
                Ok(out)
            }
        }
        #[cfg(not(feature = "deflate"))]
        FILTER_DEFLATE => {
            Err(FormatError::UnsupportedFeature(
                "deflate filter requires the 'deflate' feature".into(),
            ))
        }
        other => Err(FormatError::UnsupportedFeature(format!(
            "filter id {}",
            other
        ))),
    }
}

/// Compress multiple chunks in parallel using rayon.
///
/// Each chunk is independently compressed through the filter pipeline.
/// If compression of a chunk fails, the original (uncompressed) data is used.
#[cfg(feature = "parallel")]
pub fn apply_filters_parallel(pipeline: &FilterPipeline, chunks: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    chunks
        .par_iter()
        .map(|chunk| apply_filters(pipeline, chunk).unwrap_or_else(|_| chunk.clone()))
        .collect()
}

/// Decompress multiple chunks in parallel using rayon.
///
/// Each chunk is independently decompressed through the reversed filter pipeline.
/// If decompression of a chunk fails, the original data is used.
#[cfg(feature = "parallel")]
pub fn reverse_filters_parallel(pipeline: &FilterPipeline, chunks: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    chunks
        .par_iter()
        .map(|chunk| reverse_filters(pipeline, chunk).unwrap_or_else(|_| chunk.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_deflate() {
        let pipeline = FilterPipeline::deflate(6);
        let encoded = pipeline.encode();

        assert_eq!(encoded[0], 2); // version
        assert_eq!(encoded[1], 1); // 1 filter

        let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, pipeline);
        assert_eq!(decoded.filters[0].id, FILTER_DEFLATE);
        assert_eq!(decoded.filters[0].cd_values, vec![6]);
    }

    #[test]
    fn encode_decode_empty() {
        let pipeline = FilterPipeline::none();
        let encoded = pipeline.encode();
        assert_eq!(encoded.len(), 2);
        let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(decoded, pipeline);
    }

    #[test]
    fn encode_decode_multiple_filters() {
        let pipeline = FilterPipeline {
            filters: vec![
                Filter { id: FILTER_SHUFFLE, flags: 0, cd_values: vec![] },
                Filter { id: FILTER_DEFLATE, flags: 0, cd_values: vec![4] },
            ],
        };
        let encoded = pipeline.encode();
        let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.filters.len(), 2);
        assert_eq!(decoded.filters[0].id, FILTER_SHUFFLE);
        assert_eq!(decoded.filters[1].id, FILTER_DEFLATE);
        assert_eq!(decoded.filters[1].cd_values, vec![4]);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [1u8, 0]; // version 1
        let err = FilterPipeline::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(1)));
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [2u8]; // missing nfilters
        let err = FilterPipeline::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline::deflate(6);
        let original = vec![42u8; 1024];

        let compressed = apply_filters(&pipeline, &original).unwrap();
        // Compressed should be smaller than original for repeated data.
        assert!(compressed.len() < original.len());

        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_level_zero() {
        let pipeline = FilterPipeline::deflate(0);
        let original = b"hello world, this is a test of level 0 deflate";

        let compressed = apply_filters(&pipeline, original).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_level_nine() {
        let pipeline = FilterPipeline::deflate(9);
        let original: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();

        let compressed = apply_filters(&pipeline, &original).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[cfg(all(feature = "deflate", feature = "parallel"))]
    #[test]
    fn parallel_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline::deflate(6);
        let chunks: Vec<Vec<u8>> = (0..8)
            .map(|i| vec![(i as u8).wrapping_mul(42); 1024])
            .collect();

        let compressed = apply_filters_parallel(&pipeline, &chunks);
        assert_eq!(compressed.len(), 8);
        // Each compressed chunk should be smaller (repeated data compresses well)
        for c in &compressed {
            assert!(c.len() < 1024);
        }

        let decompressed = reverse_filters_parallel(&pipeline, &compressed);
        assert_eq!(decompressed.len(), 8);
        for (original, decoded) in chunks.iter().zip(decompressed.iter()) {
            assert_eq!(original, decoded);
        }
    }
}
