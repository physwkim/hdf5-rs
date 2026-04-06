//! Attribute support.
//!
//! Attributes are small metadata items attached to datasets (or groups).
//! They are created via the [`AttrBuilder`] API obtained from
//! [`H5Dataset::new_attr`](crate::dataset::H5Dataset::new_attr).
//!
//! # Example
//!
//! ```no_run
//! use hdf5::H5File;
//! use hdf5::types::VarLenUnicode;
//!
//! let file = H5File::create("attrs.h5").unwrap();
//! let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
//! let attr = ds.new_attr::<VarLenUnicode>().shape(()).create("units").unwrap();
//! attr.write_scalar(&VarLenUnicode("meters".to_string())).unwrap();
//! ```

use std::marker::PhantomData;

use hdf5_format::messages::attribute::AttributeMessage;

use crate::error::{Hdf5Error, Result};
use crate::file::{H5FileInner, SharedInner, borrow_inner_mut, clone_inner};
use crate::types::VarLenUnicode;

/// A handle to an HDF5 attribute.
///
/// After creating an attribute via [`AttrBuilder::create`], use
/// [`write_scalar`](Self::write_scalar) or [`write_string`](Self::write_string)
/// to set its value.
pub struct H5Attribute {
    file_inner: SharedInner,
    ds_index: usize,
    name: String,
}

impl H5Attribute {
    /// Write a scalar value to the attribute.
    ///
    /// For `VarLenUnicode`, this writes a fixed-length string attribute
    /// whose size is determined by the string value.
    pub fn write_scalar(&self, value: &VarLenUnicode) -> Result<()> {
        let attr_msg = AttributeMessage::scalar_string(&self.name, &value.0);

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.add_dataset_attribute(self.ds_index, attr_msg)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot write attributes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Write a string value to the attribute (convenience method).
    pub fn write_string(&self, value: &str) -> Result<()> {
        self.write_scalar(&VarLenUnicode(value.to_string()))
    }
}

/// A fluent builder for creating attributes on datasets.
///
/// Obtained from [`H5Dataset::new_attr::<T>()`](crate::dataset::H5Dataset::new_attr).
pub struct AttrBuilder<'a, T> {
    file_inner: &'a SharedInner,
    ds_index: usize,
    _shape_set: bool,
    _marker: PhantomData<T>,
}

impl<'a, T> AttrBuilder<'a, T> {
    pub(crate) fn new(
        file_inner: &'a SharedInner,
        ds_index: usize,
    ) -> Self {
        Self {
            file_inner,
            ds_index,
            _shape_set: false,
            _marker: PhantomData,
        }
    }

    /// Set the attribute shape. Use `()` for a scalar attribute.
    #[must_use]
    pub fn shape<S>(mut self, _shape: S) -> Self {
        // For now we only support scalar attributes.
        self._shape_set = true;
        self
    }

    /// Create the attribute with the given name.
    ///
    /// The attribute is created but does not yet have a value.
    /// Call [`H5Attribute::write_scalar`] to set the value.
    pub fn create(self, name: &str) -> Result<H5Attribute> {
        Ok(H5Attribute {
            file_inner: clone_inner(self.file_inner),
            ds_index: self.ds_index,
            name: name.to_string(),
        })
    }
}
