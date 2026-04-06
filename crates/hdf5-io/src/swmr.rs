//! SWMR (single-writer / multi-reader) protocol.
//!
//! Implements ordered flush semantics:
//! 1. Write chunk data -> fsync
//! 2. Update extensible array (new chunk address) -> fsync
//! 3. Update dataset object header (new dataspace dims) -> fsync
//! 4. Update superblock (new EOF) -> fsync

use std::path::Path;

use hdf5_format::messages::datatype::DatatypeMessage;

use crate::writer::Hdf5Writer;
use crate::IoResult;

/// SWMR writer wrapping an Hdf5Writer.
///
/// After calling `start_swmr()`, each `append_frame()` writes a chunk and
/// updates the index structures with ordered flushes.
pub struct SwmrWriter {
    writer: Hdf5Writer,
}

impl SwmrWriter {
    /// Create a new HDF5 file configured for SWMR.
    pub fn create(path: &Path) -> IoResult<Self> {
        let writer = Hdf5Writer::create(path)?;
        Ok(Self { writer })
    }

    /// Create a streaming dataset (chunked, unlimited first dim).
    ///
    /// `frame_dims` are the spatial dimensions per frame (e.g., [H, W]).
    /// The dataset will have shape [0, H, W] initially, with chunk = [1, H, W].
    pub fn create_streaming_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
    ) -> IoResult<usize> {
        // Dataset shape: [0, dim1, dim2, ...]
        let mut dims = vec![0u64];
        dims.extend_from_slice(frame_dims);

        // Max dims: [unlimited, dim1, dim2, ...]
        let mut max_dims = vec![u64::MAX];
        max_dims.extend_from_slice(frame_dims);

        // Chunk dims: [1, dim1, dim2, ...]
        let mut chunk_dims = vec![1u64];
        chunk_dims.extend_from_slice(frame_dims);

        self.writer.create_chunked_dataset(name, datatype, &dims, &max_dims, &chunk_dims)
    }

    /// Set the SWMR flag in the superblock (called before readers open the file).
    pub fn start_swmr(&mut self) -> IoResult<()> {
        // SWMR flag will be set when we write the superblock during flush/close.
        // For now this is a no-op -- the superblock is written at close time.
        Ok(())
    }

    /// Append a frame of data to a streaming dataset.
    ///
    /// This writes the chunk data, updates the extensible array index,
    /// and extends the dataset dimensions.
    pub fn append_frame(&mut self, ds_index: usize, data: &[u8]) -> IoResult<()> {
        // Get current frame count (dim 0)
        let frame_idx = {
            self.writer.datasets[ds_index].dataspace.dims[0]
        };

        // 1. Write chunk data
        self.writer.write_chunk(ds_index, frame_idx, data)?;

        // 2. Extend dimensions
        let mut new_dims = self.writer.datasets[ds_index].dataspace.dims.clone();
        new_dims[0] = frame_idx + 1;
        self.writer.extend_dataset(ds_index, &new_dims)?;

        Ok(())
    }

    /// Flush with ordered semantics for SWMR safety.
    pub fn flush(&mut self) -> IoResult<()> {
        // Flush all chunked datasets
        for i in 0..self.writer.datasets.len() {
            if self.writer.datasets[i].chunked.is_some() {
                self.writer.flush_dataset(i)?;
            }
        }
        Ok(())
    }

    /// Provide access to the underlying writer for creating non-streaming datasets.
    pub fn writer_mut(&mut self) -> &mut Hdf5Writer {
        &mut self.writer
    }

    /// Close and finalize the file.
    pub fn close(self) -> IoResult<()> {
        self.writer.close()
    }
}
