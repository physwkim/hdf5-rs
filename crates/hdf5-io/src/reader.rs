//! HDF5 file reader.
//!
//! Opens an HDF5 file, parses the superblock and root group, and provides
//! access to dataset metadata and raw data.

use std::path::Path;

use hdf5_format::superblock::SuperblockV2V3;
use hdf5_format::object_header::ObjectHeader;
use hdf5_format::messages::*;
use hdf5_format::messages::dataspace::DataspaceMessage;
use hdf5_format::messages::datatype::DatatypeMessage;
use hdf5_format::messages::link::LinkMessage;
use hdf5_format::messages::link::LinkTarget;
use hdf5_format::messages::data_layout::{self, DataLayoutMessage};
use hdf5_format::{FormatContext, UNDEF_ADDR};

use crate::file_handle::FileHandle;
use crate::IoResult;

/// Read-side metadata for a single dataset.
pub struct DatasetReadInfo {
    /// Dataset name (the link name in the root group).
    pub name: String,
    /// Element datatype.
    pub datatype: DatatypeMessage,
    /// Dataspace (dimensionality).
    pub dataspace: DataspaceMessage,
    /// Data layout (contiguous or compact).
    pub layout: DataLayoutMessage,
}

/// HDF5 file reader.
pub struct Hdf5Reader {
    handle: FileHandle,
    ctx: FormatContext,
    superblock: SuperblockV2V3,
    datasets: Vec<DatasetReadInfo>,
}

impl Hdf5Reader {
    /// Open an existing HDF5 file in SWMR read mode.
    ///
    /// Currently identical to `open()`, but indicates intent to use
    /// `refresh()` for re-reading metadata written by a concurrent SWMR writer.
    pub fn open_swmr(path: &Path) -> IoResult<Self> {
        Self::open(path)
    }

    /// Open an existing HDF5 file for reading.
    ///
    /// Parses the superblock, root group, and discovers all datasets
    /// linked from the root group.
    pub fn open(path: &Path) -> IoResult<Self> {
        let mut handle = FileHandle::open_read(path)?;

        // Read superblock (48 bytes for v3/8-byte offsets; 256 gives ample
        // headroom for other sizes).
        let sb_buf = handle.read_at_most(0, 256)?;
        let sb = SuperblockV2V3::decode(&sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Read root group object header. 4 KiB is generous for compact groups.
        let root_buf = handle.read_at_most(sb.root_group_object_header_address, 4096)?;
        let (root_header, _) = ObjectHeader::decode(&root_buf)?;

        // Walk link messages to discover datasets.
        let mut datasets = Vec::new();
        for msg in &root_header.messages {
            if msg.msg_type == MSG_LINK {
                let (link, _) = LinkMessage::decode(&msg.data, &ctx)?;
                if let LinkTarget::Hard { address } = &link.target {
                    // Read dataset object header.
                    let ds_buf = handle.read_at_most(*address, 4096)?;
                    let (ds_header, _) = ObjectHeader::decode(&ds_buf)?;

                    let mut datatype = None;
                    let mut dataspace = None;
                    let mut layout = None;

                    for ds_msg in &ds_header.messages {
                        match ds_msg.msg_type {
                            MSG_DATATYPE => {
                                let (dt, _) =
                                    DatatypeMessage::decode(&ds_msg.data, &ctx)?;
                                datatype = Some(dt);
                            }
                            MSG_DATASPACE => {
                                let (ds, _) =
                                    DataspaceMessage::decode(&ds_msg.data, &ctx)?;
                                dataspace = Some(ds);
                            }
                            MSG_DATA_LAYOUT => {
                                let (dl, _) =
                                    DataLayoutMessage::decode(&ds_msg.data, &ctx)?;
                                layout = Some(dl);
                            }
                            _ => {} // ignore other messages
                        }
                    }

                    if let (Some(dt), Some(ds), Some(dl)) = (datatype, dataspace, layout)
                    {
                        datasets.push(DatasetReadInfo {
                            name: link.name.clone(),
                            datatype: dt,
                            dataspace: ds,
                            layout: dl,
                        });
                    }
                }
            }
        }

        Ok(Self {
            handle,
            ctx,
            superblock: sb,
            datasets,
        })
    }

    /// Return the names of all datasets in the root group.
    pub fn dataset_names(&self) -> Vec<&str> {
        self.datasets.iter().map(|d| d.name.as_str()).collect()
    }

    /// Return metadata for a dataset by name.
    pub fn dataset_info(&self, name: &str) -> Option<&DatasetReadInfo> {
        self.datasets.iter().find(|d| d.name == name)
    }

    /// Return the dimensions of a dataset.
    pub fn dataset_shape(&self, name: &str) -> IoResult<Vec<u64>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;
        Ok(info.dataspace.dims.clone())
    }

    /// Read the raw bytes of a dataset.
    pub fn read_dataset_raw(&mut self, name: &str) -> IoResult<Vec<u8>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;

        // Clone layout to avoid borrow conflict with &mut self in read methods.
        let layout = info.layout.clone();

        match &layout {
            DataLayoutMessage::Contiguous { address, size } => {
                if *address == UNDEF_ADDR {
                    return Ok(vec![]);
                }
                let data = self.handle.read_at(*address, *size as usize)?;
                Ok(data)
            }
            DataLayoutMessage::Compact { data } => Ok(data.clone()),
            DataLayoutMessage::ChunkedV4 { chunk_dims, index_address, index_type, earray_params, .. } => {
                // The layout's chunk_dims include the element size as
                // the trailing dimension. Strip it for chunk indexing.
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_v4(name, real_chunk_dims, *index_address, *index_type, earray_params.as_ref())
            }
        }
    }

    /// Re-read the superblock and dataset metadata for SWMR.
    ///
    /// Call this periodically to pick up new data written by a concurrent
    /// SWMR writer. The superblock is re-read to get the latest EOF, then
    /// the root group is re-scanned for updated dataset headers (which may
    /// contain updated dataspace dimensions and chunk index addresses).
    pub fn refresh(&mut self) -> IoResult<()> {
        // Re-read superblock to get latest EOF and root group address.
        let sb_buf = self.handle.read_at_most(0, 256)?;
        let sb = SuperblockV2V3::decode(&sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Re-read root group object header.
        let root_buf = self.handle.read_at_most(sb.root_group_object_header_address, 4096)?;
        let (root_header, _) = ObjectHeader::decode(&root_buf)?;

        // Re-scan datasets from link messages.
        let mut datasets = Vec::new();
        for msg in &root_header.messages {
            if msg.msg_type == MSG_LINK {
                let (link, _) = LinkMessage::decode(&msg.data, &ctx)?;
                if let LinkTarget::Hard { address } = &link.target {
                    let ds_buf = self.handle.read_at_most(*address, 4096)?;
                    let (ds_header, _) = ObjectHeader::decode(&ds_buf)?;

                    let mut datatype = None;
                    let mut dataspace = None;
                    let mut layout = None;

                    for ds_msg in &ds_header.messages {
                        match ds_msg.msg_type {
                            MSG_DATATYPE => {
                                let (dt, _) = DatatypeMessage::decode(&ds_msg.data, &ctx)?;
                                datatype = Some(dt);
                            }
                            MSG_DATASPACE => {
                                let (ds, _) = DataspaceMessage::decode(&ds_msg.data, &ctx)?;
                                dataspace = Some(ds);
                            }
                            MSG_DATA_LAYOUT => {
                                let (dl, _) = DataLayoutMessage::decode(&ds_msg.data, &ctx)?;
                                layout = Some(dl);
                            }
                            _ => {}
                        }
                    }

                    if let (Some(dt), Some(ds), Some(dl)) = (datatype, dataspace, layout) {
                        datasets.push(DatasetReadInfo {
                            name: link.name.clone(),
                            datatype: dt,
                            dataspace: ds,
                            layout: dl,
                        });
                    }
                }
            }
        }

        self.superblock = sb;
        self.ctx = ctx;
        self.datasets = datasets;

        Ok(())
    }

    /// Read chunked dataset data by walking the extensible array index.
    fn read_chunked_v4(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        index_type: data_layout::ChunkIndexType,
        earray_params: Option<&data_layout::EarrayParams>,
    ) -> IoResult<Vec<u8>> {
        use hdf5_format::chunk_index::extensible_array::*;

        let info = self.dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;

        match index_type {
            data_layout::ChunkIndexType::SingleChunk => {
                // Single chunk: the index_address IS the chunk address
                let total_size: u64 = dims.iter().product::<u64>() * element_size;
                if index_address == UNDEF_ADDR || total_size == 0 {
                    return Ok(vec![]);
                }
                let data = self.handle.read_at(index_address, total_size as usize)?;
                Ok(data)
            }
            data_layout::ChunkIndexType::ExtensibleArray => {
                let params = earray_params.ok_or_else(|| {
                    crate::IoError::InvalidState("missing earray params".into())
                })?;

                if index_address == UNDEF_ADDR {
                    return Ok(vec![]);
                }

                // Read the EA header
                let hdr_buf = self.handle.read_at_most(index_address, 256)?;
                let ea_hdr = ExtensibleArrayHeader::decode(&hdr_buf, &self.ctx)?;

                if ea_hdr.idx_blk_addr == UNDEF_ADDR {
                    return Ok(vec![]);
                }

                // Compute number of chunks along the unlimited dimension (dim 0)
                let chunks_dim0 = if chunk_dims[0] > 0 {
                    (dims[0] + chunk_dims[0] - 1) / chunk_dims[0]
                } else {
                    0
                };

                // Read index block
                let ndblk_addrs = compute_ndblk_addrs(params.sup_blk_min_data_ptrs);
                let nsblk_addrs = compute_nsblk_addrs(
                    params.idx_blk_elmts,
                    params.data_blk_min_elmts,
                    params.sup_blk_min_data_ptrs,
                    params.max_nelmts_bits,
                );
                let iblk_buf = self.handle.read_at_most(ea_hdr.idx_blk_addr, 8192)?;
                let iblk = ExtensibleArrayIndexBlock::decode(
                    &iblk_buf, &self.ctx,
                    params.idx_blk_elmts as usize,
                    ndblk_addrs, nsblk_addrs,
                )?;

                // Collect chunk addresses
                let mut chunk_addrs: Vec<u64> = Vec::new();
                for &addr in &iblk.elements {
                    chunk_addrs.push(addr);
                }

                // Read data blocks if needed
                let mut dblk_nelmts = params.data_blk_min_elmts as usize;
                for &dblk_addr in &iblk.dblk_addrs {
                    if dblk_addr == UNDEF_ADDR {
                        // Add UNDEF entries for the unallocated block
                        for _ in 0..dblk_nelmts {
                            chunk_addrs.push(UNDEF_ADDR);
                        }
                    } else {
                        let dblk_buf = self.handle.read_at_most(dblk_addr, 4096)?;
                        let dblk = ExtensibleArrayDataBlock::decode(
                            &dblk_buf, &self.ctx,
                            params.max_nelmts_bits, dblk_nelmts,
                        )?;
                        for &addr in &dblk.elements {
                            chunk_addrs.push(addr);
                        }
                    }
                    // Data blocks grow: first 2 are min size, then double
                    // For simplicity, keep at min for now
                    if chunk_addrs.len() >= chunks_dim0 as usize {
                        break;
                    }
                    dblk_nelmts *= 2;
                }

                // Compute chunk byte size
                let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

                // Total output size
                let total_size: u64 = dims.iter().product::<u64>() * element_size;
                let mut output = vec![0u8; total_size as usize];

                // Read each chunk
                for i in 0..chunks_dim0 as usize {
                    if i >= chunk_addrs.len() {
                        break;
                    }
                    let addr = chunk_addrs[i];
                    if addr == UNDEF_ADDR {
                        continue;
                    }
                    let chunk_data = self.handle.read_at(addr, chunk_bytes as usize)?;
                    let offset = i as u64 * chunk_bytes;
                    let end = std::cmp::min(offset + chunk_bytes, total_size);
                    let copy_len = (end - offset) as usize;
                    output[offset as usize..offset as usize + copy_len]
                        .copy_from_slice(&chunk_data[..copy_len]);
                }

                Ok(output)
            }
            _ => Err(crate::IoError::InvalidState(format!(
                "unsupported chunk index type: {:?}", index_type
            ))),
        }
    }
}
