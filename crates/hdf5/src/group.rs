//! Group support.
//!
//! Provides basic nested group support. Groups are containers for datasets
//! and other groups, forming a hierarchical namespace within an HDF5 file.
//!
//! Currently, all datasets are created in the root group. The `H5Group`
//! type enables creating datasets at arbitrary "/" separated paths.
//!
//! # Example
//!
//! ```no_run
//! use hdf5::H5File;
//!
//! let file = H5File::create("groups.h5").unwrap();
//! let root = file.root_group();
//! // Create a dataset in the root group
//! let ds = root.new_dataset::<f32>()
//!     .shape(&[10])
//!     .create("temperature")
//!     .unwrap();
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::dataset::DatasetBuilder;
use crate::file::H5FileInner;
use crate::types::H5Type;

/// A handle to an HDF5 group.
///
/// Groups are containers for datasets and other groups. The root group
/// is always available via [`H5File::root_group`](crate::file::H5File::root_group).
pub struct H5Group {
    file_inner: Rc<RefCell<H5FileInner>>,
    /// The absolute path of this group (e.g., "/" or "/detector").
    name: String,
}

impl H5Group {
    /// Create a new group handle.
    pub(crate) fn new(file_inner: Rc<RefCell<H5FileInner>>, name: String) -> Self {
        Self { file_inner, name }
    }

    /// Return the name (path) of this group.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Start building a new dataset in this group.
    ///
    /// The dataset name will be prefixed with this group's path when
    /// stored in the file. For now, all datasets are stored as links
    /// in the root group with "/" separated names.
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilder<T> {
        DatasetBuilder::new(Rc::clone(&self.file_inner))
    }

    /// Create a sub-group within this group.
    ///
    /// Returns a handle to the new group. The group name is combined
    /// with this group's path using "/" as separator.
    ///
    /// Note: In the current implementation, sub-groups are a namespace
    /// convenience. All datasets are still stored as flat links in the
    /// root group with path-style names (e.g., "group1/data").
    pub fn create_group(&self, name: &str) -> H5Group {
        let full_name = if self.name == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", self.name, name)
        };
        H5Group {
            file_inner: Rc::clone(&self.file_inner),
            name: full_name,
        }
    }
}
