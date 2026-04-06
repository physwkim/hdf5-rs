/// Superblock v2/v3 encode/decode for HDF5 files.
///
/// The superblock is always at offset 0 (or at a user-hint offset) and
/// contains the file-level metadata: version, size parameters, and addresses
/// of the root group and end-of-file.
use crate::checksum::checksum_metadata;
use crate::{FormatError, FormatResult};

/// The 8-byte HDF5 file signature that begins every superblock.
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Superblock version 2.
pub const SUPERBLOCK_V2: u8 = 2;

/// Superblock version 3 (adds SWMR support).
pub const SUPERBLOCK_V3: u8 = 3;

/// File consistency flag: file was opened for write access.
pub const FLAG_WRITE_ACCESS: u8 = 0x01;

/// File consistency flag: file is consistent / was properly closed.
pub const FLAG_FILE_OK: u8 = 0x02;

/// File consistency flag: file was opened for single-writer/multi-reader.
pub const FLAG_SWMR_WRITE: u8 = 0x04;

/// Superblock v2/v3 structure.
///
/// Layout (O = sizeof_offsets):
/// ```text
/// [0..8]              Signature (8 bytes)
/// [8]                 Version (1 byte)
/// [9]                 Size of Offsets (1 byte)
/// [10]                Size of Lengths (1 byte)
/// [11]                File Consistency Flags (1 byte)
/// [12..12+O]          Base Address (O bytes)
/// [12+O..12+2O]       Superblock Extension Address (O bytes)
/// [12+2O..12+3O]      End of File Address (O bytes)
/// [12+3O..12+4O]      Root Group Object Header Address (O bytes)
/// [12+4O..12+4O+4]    Checksum (4 bytes)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockV2V3 {
    /// Superblock version: 2 or 3.
    pub version: u8,
    /// Size of file offsets in bytes (typically 8).
    pub sizeof_offsets: u8,
    /// Size of file lengths in bytes (typically 8).
    pub sizeof_lengths: u8,
    /// File consistency flags (see `FLAG_*` constants).
    pub file_consistency_flags: u8,
    /// Base address of the file (usually 0).
    pub base_address: u64,
    /// Address of the superblock extension object header, or UNDEF.
    pub superblock_extension_address: u64,
    /// End-of-file address.
    pub end_of_file_address: u64,
    /// Address of the root group object header.
    pub root_group_object_header_address: u64,
}

impl SuperblockV2V3 {
    /// Returns the total encoded size in bytes: 12 + 4*O + 4 (checksum).
    pub fn encoded_size(&self) -> usize {
        12 + 4 * (self.sizeof_offsets as usize) + 4
    }

    /// Encode the superblock to a byte vector, including the trailing checksum.
    pub fn encode(&self) -> Vec<u8> {
        let size = self.encoded_size();
        let mut buf = Vec::with_capacity(size);

        // Signature
        buf.extend_from_slice(&HDF5_SIGNATURE);
        // Version
        buf.push(self.version);
        // Size of Offsets
        buf.push(self.sizeof_offsets);
        // Size of Lengths
        buf.push(self.sizeof_lengths);
        // File Consistency Flags
        buf.push(self.file_consistency_flags);

        // Addresses -- encode as little-endian with sizeof_offsets bytes
        let o = self.sizeof_offsets as usize;
        encode_offset(&mut buf, self.base_address, o);
        encode_offset(&mut buf, self.superblock_extension_address, o);
        encode_offset(&mut buf, self.end_of_file_address, o);
        encode_offset(&mut buf, self.root_group_object_header_address, o);

        // Checksum over everything before the checksum field
        debug_assert_eq!(buf.len(), size - 4);
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    /// Decode a superblock from a byte buffer. Verifies the signature, version,
    /// and checksum. Returns the parsed superblock.
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        // Minimum size check: we need at least the fixed 12-byte header to
        // read sizeof_offsets before computing the full size.
        if buf.len() < 12 {
            return Err(FormatError::BufferTooShort {
                needed: 12,
                available: buf.len(),
            });
        }

        // Signature
        if buf[0..8] != HDF5_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        // Version
        let version = buf[8];
        if version != SUPERBLOCK_V2 && version != SUPERBLOCK_V3 {
            return Err(FormatError::InvalidVersion(version));
        }

        let sizeof_offsets = buf[9];
        let sizeof_lengths = buf[10];
        let file_consistency_flags = buf[11];

        let o = sizeof_offsets as usize;
        let total_size = 12 + 4 * o + 4;
        if buf.len() < total_size {
            return Err(FormatError::BufferTooShort {
                needed: total_size,
                available: buf.len(),
            });
        }

        // Verify checksum
        let data_end = total_size - 4;
        let stored_cksum = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        // Decode addresses
        let mut pos = 12;
        let base_address = decode_offset(buf, &mut pos, o);
        let superblock_extension_address = decode_offset(buf, &mut pos, o);
        let end_of_file_address = decode_offset(buf, &mut pos, o);
        let root_group_object_header_address = decode_offset(buf, &mut pos, o);

        Ok(SuperblockV2V3 {
            version,
            sizeof_offsets,
            sizeof_lengths,
            file_consistency_flags,
            base_address,
            superblock_extension_address,
            end_of_file_address,
            root_group_object_header_address,
        })
    }
}

/// Encode a u64 address as `size` little-endian bytes and append to `buf`.
fn encode_offset(buf: &mut Vec<u8>, value: u64, size: usize) {
    let bytes = value.to_le_bytes();
    buf.extend_from_slice(&bytes[..size]);
}

/// Decode a little-endian address of `size` bytes from `buf` at `*pos`,
/// advancing `*pos` past the consumed bytes.
fn decode_offset(buf: &[u8], pos: &mut usize, size: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..size].copy_from_slice(&buf[*pos..*pos + size]);
    *pos += size;
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UNDEF_ADDR;

    #[test]
    fn test_encoded_size() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        // 12 + 4*8 + 4 = 48
        assert_eq!(sb.encoded_size(), 48);
    }

    #[test]
    fn test_roundtrip_v3_offset8() {
        let original = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: FLAG_FILE_OK,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 0x1_0000,
            root_group_object_header_address: 48,
        };

        let encoded = original.encode();
        assert_eq!(encoded.len(), original.encoded_size());

        // Verify signature
        assert_eq!(&encoded[..8], &HDF5_SIGNATURE);

        let decoded = SuperblockV2V3::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_v2_offset4() {
        let original = SuperblockV2V3 {
            version: SUPERBLOCK_V2,
            sizeof_offsets: 4,
            sizeof_lengths: 4,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: 0xFFFF_FFFF,
            end_of_file_address: 8192,
            root_group_object_header_address: 28,
        };

        let encoded = original.encode();
        // 12 + 4*4 + 4 = 32
        assert_eq!(encoded.len(), 32);

        let decoded = SuperblockV2V3::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_bad_signature() {
        let mut data = vec![0u8; 48];
        // Wrong signature
        data[0] = 0x00;
        let err = SuperblockV2V3::decode(&data).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn test_decode_bad_version() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let mut encoded = sb.encode();
        // Corrupt version to 1
        encoded[8] = 1;
        let err = SuperblockV2V3::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(1)));
    }

    #[test]
    fn test_decode_checksum_mismatch() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let mut encoded = sb.encode();
        // Corrupt a data byte
        encoded[12] = 0xFF;
        let err = SuperblockV2V3::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let err = SuperblockV2V3::decode(&[0u8; 4]).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn test_flags() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let encoded = sb.encode();
        let decoded = SuperblockV2V3::decode(&encoded).unwrap();
        assert_eq!(
            decoded.file_consistency_flags,
            FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE
        );
    }

    #[test]
    fn test_roundtrip_with_extra_trailing_data() {
        // decode should succeed even if the buffer is longer than needed
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let mut encoded = sb.encode();
        encoded.extend_from_slice(&[0xAA; 100]); // trailing garbage
        let decoded = SuperblockV2V3::decode(&encoded).unwrap();
        assert_eq!(decoded, sb);
    }
}
