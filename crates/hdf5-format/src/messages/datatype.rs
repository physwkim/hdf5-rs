//! Datatype message (type 0x03) — describes element data type.
//!
//! Binary layout:
//!   Byte 0:    (class & 0x0F) | (version << 4)     version = 1
//!   Bytes 1-3: class bit-field flags (24 bits, little-endian)
//!   Bytes 4-7: element size (u32 LE)
//!   Bytes 8+:  class-specific properties

use crate::{FormatContext, FormatError, FormatResult};

const DT_VERSION: u8 = 1;

// Datatype class codes
const CLASS_FIXED_POINT: u8 = 0;
const CLASS_FLOATING_POINT: u8 = 1;
const CLASS_STRING: u8 = 3;

/// Byte order for numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// HDF5 datatype descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum DatatypeMessage {
    FixedPoint {
        size: u32,
        byte_order: ByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    },
    FloatingPoint {
        size: u32,
        byte_order: ByteOrder,
        sign_location: u8,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
    /// Fixed-length string type (class 3).
    FixedString {
        /// String size in bytes (including null terminator if null-terminated).
        size: u32,
        /// Padding type: 0 = null terminate, 1 = null pad, 2 = space pad.
        padding: u8,
        /// Character set: 0 = ASCII, 1 = UTF-8.
        charset: u8,
    },
}

// ========================================================================= factory methods

impl DatatypeMessage {
    pub fn u8_type() -> Self {
        Self::FixedPoint {
            size: 1,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    pub fn i8_type() -> Self {
        Self::FixedPoint {
            size: 1,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    pub fn u16_type() -> Self {
        Self::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    pub fn i16_type() -> Self {
        Self::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    pub fn u32_type() -> Self {
        Self::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    pub fn i32_type() -> Self {
        Self::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    pub fn u64_type() -> Self {
        Self::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 64,
        }
    }

    pub fn i64_type() -> Self {
        Self::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 64,
        }
    }

    pub fn f32_type() -> Self {
        Self::FloatingPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            sign_location: 31,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    pub fn f64_type() -> Self {
        Self::FloatingPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            sign_location: 63,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    /// Null-terminated ASCII fixed-length string.
    pub fn fixed_string(size: u32) -> Self {
        Self::FixedString {
            size,
            padding: 0, // null terminate
            charset: 0, // ASCII
        }
    }

    /// Null-terminated UTF-8 fixed-length string.
    pub fn fixed_string_utf8(size: u32) -> Self {
        Self::FixedString {
            size,
            padding: 0, // null terminate
            charset: 1, // UTF-8
        }
    }
}

// ========================================================================= queries

impl DatatypeMessage {
    /// Returns the element size in bytes.
    pub fn element_size(&self) -> u32 {
        match self {
            Self::FixedPoint { size, .. } => *size,
            Self::FloatingPoint { size, .. } => *size,
            Self::FixedString { size, .. } => *size,
        }
    }
}

// ========================================================================= encode / decode

impl DatatypeMessage {
    /// Encode into a byte vector.
    pub fn encode(&self, _ctx: &FormatContext) -> Vec<u8> {
        match self {
            Self::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                // Total: 8 header + 4 properties = 12 bytes
                let mut buf = Vec::with_capacity(12);

                // byte 0: class | version<<4
                buf.push(CLASS_FIXED_POINT | (DT_VERSION << 4));

                // bytes 1-3: class bit-field (24 bits LE)
                let mut flags0: u8 = 0;
                if *byte_order == ByteOrder::BigEndian {
                    flags0 |= 0x01; // bit 0
                }
                if *signed {
                    flags0 |= 0x08; // bit 3
                }
                buf.push(flags0);
                buf.push(0); // flags byte 1
                buf.push(0); // flags byte 2

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                // properties: bit_offset(u16) + bit_precision(u16)
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());

                buf
            }
            Self::FloatingPoint {
                size,
                byte_order,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            } => {
                // Total: 8 header + 12 properties = 20 bytes
                let mut buf = Vec::with_capacity(20);

                // byte 0: class | version<<4
                buf.push(CLASS_FLOATING_POINT | (DT_VERSION << 4));

                // bytes 1-3: class bit-field
                let mut flags0: u8 = 0;
                if *byte_order == ByteOrder::BigEndian {
                    flags0 |= 0x01; // bit 0 of byte order
                }
                // bits 4-5: mantissa normalization = 2 (implied leading 1 for IEEE)
                flags0 |= 0x02 << 4; // IMPLIED = 2
                buf.push(flags0);

                // flags byte 1: sign bit position
                buf.push(*sign_location);

                // flags byte 2: unused
                buf.push(0);

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                // properties (12 bytes)
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf.push(*exponent_location);
                buf.push(*exponent_size);
                buf.push(*mantissa_location);
                buf.push(*mantissa_size);
                buf.extend_from_slice(&exponent_bias.to_le_bytes());

                buf
            }
            Self::FixedString {
                size,
                padding,
                charset,
            } => {
                // Total: 8 header bytes, no additional properties
                let mut buf = Vec::with_capacity(8);

                // byte 0: class | version<<4
                buf.push(CLASS_STRING | (DT_VERSION << 4));

                // byte 1: (padding & 0x0f) | ((charset & 0x0f) << 4)
                buf.push((padding & 0x0F) | ((charset & 0x0F) << 4));

                // bytes 2-3: rest of class bit fields (zero)
                buf.push(0);
                buf.push(0);

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                buf
            }
        }
    }

    /// Decode from a byte buffer.  Returns `(message, bytes_consumed)`.
    pub fn decode(buf: &[u8], _ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        if buf.len() < 8 {
            return Err(FormatError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        let class = buf[0] & 0x0F;
        let version = buf[0] >> 4;
        if version != DT_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        let flags0 = buf[1];
        let flags1 = buf[2];
        // flags2 = buf[3]; // reserved / unused for classes 0 and 1

        let size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

        match class {
            CLASS_FIXED_POINT => {
                if buf.len() < 12 {
                    return Err(FormatError::BufferTooShort {
                        needed: 12,
                        available: buf.len(),
                    });
                }
                let byte_order = if (flags0 & 0x01) != 0 {
                    ByteOrder::BigEndian
                } else {
                    ByteOrder::LittleEndian
                };
                let signed = (flags0 & 0x08) != 0;

                let bit_offset = u16::from_le_bytes([buf[8], buf[9]]);
                let bit_precision = u16::from_le_bytes([buf[10], buf[11]]);

                Ok((
                    Self::FixedPoint {
                        size,
                        byte_order,
                        signed,
                        bit_offset,
                        bit_precision,
                    },
                    12,
                ))
            }
            CLASS_FLOATING_POINT => {
                if buf.len() < 20 {
                    return Err(FormatError::BufferTooShort {
                        needed: 20,
                        available: buf.len(),
                    });
                }
                let byte_order = if (flags0 & 0x01) != 0 {
                    ByteOrder::BigEndian
                } else {
                    ByteOrder::LittleEndian
                };
                let sign_location = flags1;

                let bit_offset = u16::from_le_bytes([buf[8], buf[9]]);
                let bit_precision = u16::from_le_bytes([buf[10], buf[11]]);
                let exponent_location = buf[12];
                let exponent_size = buf[13];
                let mantissa_location = buf[14];
                let mantissa_size = buf[15];
                let exponent_bias =
                    u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);

                Ok((
                    Self::FloatingPoint {
                        size,
                        byte_order,
                        sign_location,
                        bit_offset,
                        bit_precision,
                        exponent_location,
                        exponent_size,
                        mantissa_location,
                        mantissa_size,
                        exponent_bias,
                    },
                    20,
                ))
            }
            CLASS_STRING => {
                // String class: 8-byte header, no additional properties
                let padding = flags0 & 0x0F;
                let charset = (flags0 >> 4) & 0x0F;

                Ok((
                    Self::FixedString {
                        size,
                        padding,
                        charset,
                    },
                    8,
                ))
            }
            _ => Err(FormatError::UnsupportedFeature(format!(
                "datatype class {}",
                class
            ))),
        }
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    // ---- fixed point roundtrips ----

    #[test]
    fn roundtrip_u8() {
        let msg = DatatypeMessage::u8_type();
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 12);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i8() {
        let msg = DatatypeMessage::i8_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_u16() {
        let msg = DatatypeMessage::u16_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i16() {
        let msg = DatatypeMessage::i16_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_u32() {
        let msg = DatatypeMessage::u32_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i32() {
        let msg = DatatypeMessage::i32_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_u64() {
        let msg = DatatypeMessage::u64_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i64() {
        let msg = DatatypeMessage::i64_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    // ---- floating point roundtrips ----

    #[test]
    fn roundtrip_f32() {
        let msg = DatatypeMessage::f32_type();
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 20);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_f64() {
        let msg = DatatypeMessage::f64_type();
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 20);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(decoded, msg);
    }

    // ---- edge / error cases ----

    #[test]
    fn fixed_point_big_endian() {
        let msg = DatatypeMessage::FixedPoint {
            size: 4,
            byte_order: ByteOrder::BigEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn floating_point_big_endian() {
        let msg = DatatypeMessage::FloatingPoint {
            size: 8,
            byte_order: ByteOrder::BigEndian,
            sign_location: 63,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8; 4];
        let err = DatatypeMessage::decode(&buf, &ctx()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_unsupported_class() {
        // class 5, version 1
        let mut buf = [0u8; 12];
        buf[0] = 5 | (1 << 4);
        buf[4] = 1; // size = 1
        let err = DatatypeMessage::decode(&buf, &ctx()).unwrap_err();
        match err {
            FormatError::UnsupportedFeature(_) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_encoding() {
        let encoded = DatatypeMessage::u32_type().encode(&ctx());
        assert_eq!(encoded[0] >> 4, DT_VERSION);
        assert_eq!(encoded[0] & 0x0F, CLASS_FIXED_POINT);
    }

    #[test]
    fn signed_flag_encoding() {
        let unsigned = DatatypeMessage::u32_type().encode(&ctx());
        let signed = DatatypeMessage::i32_type().encode(&ctx());
        assert_eq!(unsigned[1] & 0x08, 0);
        assert_eq!(signed[1] & 0x08, 0x08);
    }

    // ---- fixed string roundtrips ----

    #[test]
    fn roundtrip_fixed_string_ascii() {
        let msg = DatatypeMessage::fixed_string(10);
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 8); // 8-byte header, no properties
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_fixed_string_utf8() {
        let msg = DatatypeMessage::fixed_string_utf8(20);
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 8);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn fixed_string_element_size() {
        let msg = DatatypeMessage::fixed_string(42);
        assert_eq!(msg.element_size(), 42);
    }

    #[test]
    fn fixed_string_class_encoding() {
        let encoded = DatatypeMessage::fixed_string(5).encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, 3); // class = 3
        assert_eq!(encoded[0] >> 4, DT_VERSION); // version = 1
    }

    #[test]
    fn fixed_string_charset_encoding() {
        let ascii = DatatypeMessage::fixed_string(5).encode(&ctx());
        assert_eq!(ascii[1] & 0x0F, 0); // padding = null terminate
        assert_eq!((ascii[1] >> 4) & 0x0F, 0); // charset = ASCII

        let utf8 = DatatypeMessage::fixed_string_utf8(5).encode(&ctx());
        assert_eq!(utf8[1] & 0x0F, 0); // padding = null terminate
        assert_eq!((utf8[1] >> 4) & 0x0F, 1); // charset = UTF-8
    }
}
