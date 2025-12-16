#![allow(unused_assignments)] // false positive from thiserror macro expansion

#[doc = include_str!("../ReadMe.md")]
pub(crate) mod context;

pub(crate) mod parsing;
pub(crate) mod uniforms;

/// This module provides typesafe wrapper for internal uniform buffer data.
pub mod input_type;

pub use wgpu;
// the main rendering context.
pub use context::{RenderContext, TextureDesc};

pub(crate) type VarName = String;

use thiserror::Error;
use wgpu::naga::{self};

#[derive(Debug, Clone, Default)]
pub struct ErrorLocation {
    /// 1-based line number where the error occurred.
    pub line: u32,
    /// 1-based column number where the error occurred.
    pub column: u32,
}

/// A shader compilation error with location and kind information.
#[derive(Debug, Clone)]
pub struct ShaderError {
    /// The location of the error in the source code.
    pub location: ErrorLocation,
    /// The type of error that occurred.
    pub kind: naga::front::glsl::ErrorKind,
}

/// Joint error type
#[derive(Debug, Error)]
pub enum Error {
    /// Thrown if the shader compilation fails, includes a newline separated list of compile errors.
    #[error("Shader compilation failed: {display}")]
    ShaderCompilationFailed {
        display: String,
        errors: Vec<ShaderError>,
    },

    /// Thrown if a pragma is malformed
    #[error("Document parsing failed: {0}")]
    DocumentParsingFailed(#[from] crate::parsing::Error),

    /// Thrown if uniforms were an unsupported or unexpected format
    /// or not present. Note that currently naga omits unused global
    /// variables and uniforms - this may cause this error to be erroneously thrown.
    #[error("Uniform setup failed: {0}")]
    UniformError(#[from] crate::uniforms::Error),
}
