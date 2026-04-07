//! Bit-level stream writers for zstd encoding.
//!
//! `BitWriter` — forward bitstream (Huffman literals).
//! `BackwardBitWriter` — backward bitstream (FSE sequences, FSE weights).
//!
//! The backward bitstream matches C zstd's `BIT_CStream_t` exactly:
//! - Bits accumulate LSB-first in a 64-bit register
//! - `flush_bits()` writes full bytes to output (forward/LE)
//! - `finish()` adds sentinel 1-bit, flushes, returns bytes
//! - Decoder reads this from the END toward the BEGINNING

/// Forward bitstream writer (Huffman literal streams).
pub struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32,
    current: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(256),
            bit_pos: 0,
            current: 0,
        }
    }

    pub fn write_bits(&mut self, value: u64, nbits: u32) {
        let mut val = value;
        let mut bits = nbits;
        while bits > 0 {
            let space = 8 - self.bit_pos;
            let take = std::cmp::min(space, bits);
            let mask = (1u64 << take) - 1;
            self.current |= ((val & mask) as u8) << self.bit_pos;
            val >>= take;
            bits -= take;
            self.bit_pos += take;
            if self.bit_pos == 8 {
                self.buf.push(self.current);
                self.current = 0;
                self.bit_pos = 0;
            }
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buf.push(self.current);
        }
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len() * 8 + self.bit_pos as usize
    }
}

/// Backward bitstream writer matching C zstd's BIT_CStream_t.
///
/// Bits accumulate LSB-first in a 64-bit container. `flush_bits()` writes
/// complete bytes to the output buffer in LE order. The decoder reads
/// from the END of this buffer (BitReaderReversed).
///
/// Key: bytes are written FORWARD. No reverse needed. The decoder
/// naturally reads backward from the last byte.
pub struct BackwardBitWriter {
    container: u64,
    bit_pos: u32,
    buf: Vec<u8>,
}

impl Default for BackwardBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardBitWriter {
    pub fn new() -> Self {
        Self {
            container: 0,
            bit_pos: 0,
            buf: Vec::with_capacity(256),
        }
    }

    /// Add `nbits` from the low bits of `value` to the container.
    /// Matches: `BIT_addBits(bitC, value, nbBits)`
    #[inline]
    pub fn add_bits(&mut self, value: u64, nbits: u32) {
        if nbits == 0 {
            return;
        }
        debug_assert!(nbits <= 57);
        debug_assert!(self.bit_pos + nbits <= 64);
        let mask = if nbits >= 64 {
            u64::MAX
        } else {
            (1u64 << nbits) - 1
        };
        self.container |= (value & mask) << self.bit_pos;
        self.bit_pos += nbits;
    }

    /// Flush complete bytes from the container to the output.
    /// Matches: `BIT_flushBits(bitC)`
    #[inline]
    pub fn flush_bits(&mut self) {
        let nb_bytes = (self.bit_pos / 8) as usize;
        for i in 0..nb_bytes {
            self.buf.push((self.container >> (i * 8)) as u8);
        }
        self.container >>= nb_bytes * 8;
        self.bit_pos &= 7;
    }

    /// Finalize: add sentinel 1-bit, flush remaining.
    /// Matches: `BIT_closeCStream(bitC)`
    pub fn finish(mut self) -> Vec<u8> {
        self.add_bits(1, 1);
        self.flush_bits();
        if self.bit_pos > 0 {
            self.buf.push(self.container as u8);
        }
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_writer_basic() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b1100, 4);
        w.write_bits(0b1, 1);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xE5]);
    }

    #[test]
    fn backward_writer_sentinel_only() {
        let w = BackwardBitWriter::new();
        let result = w.finish();
        // Sentinel 1-bit at position 0 → byte 0x01
        assert_eq!(result, vec![0x01]);
    }

    #[test]
    fn backward_writer_c_layout() {
        let mut w = BackwardBitWriter::new();
        w.add_bits(0xFF, 8);
        w.flush_bits();
        w.add_bits(0xAB, 8);
        let result = w.finish();
        // flush: [0xFF], then add 0xAB+sentinel → container=0x1AB, bitPos=9
        // flush 1 byte: [0xAB], remaining 0x01
        // result: [0xFF, 0xAB, 0x01]
        assert_eq!(result, vec![0xFF, 0xAB, 0x01]);
    }
}
