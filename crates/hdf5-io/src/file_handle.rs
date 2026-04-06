use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

/// Wraps `std::fs::File` with positioned I/O convenience methods.
pub struct FileHandle {
    file: File,
}

impl FileHandle {
    /// Create a new file (truncating if it already exists) opened for
    /// read/write access.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self { file })
    }

    /// Open an existing file for read-only access.
    pub fn open_read(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        Ok(Self { file })
    }

    /// Open an existing file for read/write access.
    pub fn open_readwrite(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        Ok(Self { file })
    }

    /// Write `data` at the given byte offset.
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(data)?;
        Ok(())
    }

    /// Read exactly `len` bytes starting at the given byte offset.
    ///
    /// Returns an error if fewer than `len` bytes are available.
    pub fn read_at(&mut self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read up to `max_len` bytes starting at the given byte offset.
    ///
    /// Returns the bytes actually read, which may be fewer than `max_len` if
    /// the file ends before that.
    pub fn read_at_most(&mut self, offset: u64, max_len: usize) -> std::io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; max_len];
        let mut total = 0;
        loop {
            match self.file.read(&mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        buf.truncate(total);
        Ok(buf)
    }

    /// Flush file data (not necessarily metadata) to disk.
    pub fn sync_data(&self) -> std::io::Result<()> {
        self.file.sync_data()
    }

    /// Flush both file data and metadata to disk.
    pub fn sync_all(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    /// Return the current file size by seeking to the end.
    pub fn file_size(&mut self) -> std::io::Result<u64> {
        let pos = self.file.seek(SeekFrom::End(0))?;
        Ok(pos)
    }
}
