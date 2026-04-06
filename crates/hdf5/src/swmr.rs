//! Single Writer / Multiple Reader (SWMR) API.
//!
//! Provides a high-level wrapper around the SWMR protocol for streaming
//! frame-based data (e.g., area detector images).

use std::path::Path;

use hdf5_io::SwmrWriter as IoSwmrWriter;

use crate::error::Result;
use crate::types::H5Type;

/// SWMR writer for streaming frame-based data to an HDF5 file.
///
/// Usage:
/// ```no_run
/// use hdf5::swmr::SwmrFileWriter;
///
/// let mut writer = SwmrFileWriter::create("stream.h5").unwrap();
/// let ds = writer.create_streaming_dataset::<f32>("frames", &[256, 256]).unwrap();
/// writer.start_swmr().unwrap();
///
/// // Write frames
/// let frame_data = vec![0.0f32; 256 * 256];
/// let raw: Vec<u8> = frame_data.iter()
///     .flat_map(|v| v.to_le_bytes())
///     .collect();
/// writer.append_frame(ds, &raw).unwrap();
/// writer.flush().unwrap();
///
/// writer.close().unwrap();
/// ```
pub struct SwmrFileWriter {
    inner: IoSwmrWriter,
}

impl SwmrFileWriter {
    /// Create a new HDF5 file for SWMR streaming.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = IoSwmrWriter::create(path.as_ref())?;
        Ok(Self { inner })
    }

    /// Create a streaming dataset.
    ///
    /// The dataset will have shape `[0, frame_dims...]` initially, with
    /// chunk dimensions `[1, frame_dims...]` and unlimited first dimension.
    ///
    /// Returns the dataset index for use with `append_frame`.
    pub fn create_streaming_dataset<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
    ) -> Result<usize> {
        let datatype = T::hdf5_type();
        let idx = self.inner.create_streaming_dataset(name, datatype, frame_dims)?;
        Ok(idx)
    }

    /// Signal the start of SWMR mode.
    pub fn start_swmr(&mut self) -> Result<()> {
        self.inner.start_swmr()?;
        Ok(())
    }

    /// Append a frame of raw data to a streaming dataset.
    ///
    /// The data size must match one frame (product of frame_dims * element_size).
    pub fn append_frame(&mut self, ds_index: usize, data: &[u8]) -> Result<()> {
        self.inner.append_frame(ds_index, data)?;
        Ok(())
    }

    /// Flush all dataset index structures to disk with SWMR ordering.
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }

    /// Close and finalize the file.
    pub fn close(self) -> Result<()> {
        self.inner.close()?;
        Ok(())
    }
}
