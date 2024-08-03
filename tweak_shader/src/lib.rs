//! # Lib Tweak Shader
//!
//! The Tweak Shader Library provides a rendering and bookkeeping context for an interactive screen shader format. It allows users to create shaders reminiscent of ShaderToy or ISF shaders with custom uniforms that can be tweaked at runtime. The library features support  image,  as well as various other types, including colors, floats, integers, 2D points, and more. The design and functionality of this library were inspired by the ISF (Interactive Shader Format) project.
//!
//! ## Usage
//!
//! ```rust, ignore
//! use tweak_shader::RenderContext;
//! use wgpu::TextureFormat;
//!
//! let src =  r#"
//!#version 450
//!#pragma tweak_shader(version=1.0)
//!
//!layout(location = 0) out vec4 out_color;
//!
//!#pragma input(float, name="foo", default=0.0, min=0.0, max=1.0)
//!#pragma input(float, name="bar")
//!#pragma input(float, name="baz", default=0.5)
//!layout(set = 0, binding = 0) uniform Inputs {
//!    float foo;
//!    float bar;
//!    float baz;
//!};
//!
//!void main()
//!{
//!    out_color = vec4(foo, bar, baz, 1.0);
//!}
//! "#;
//!
//! let format = TextureFormat::Rgba8UnormSrgb;
//! let device = // your wgpu::Device here;
//! let queue = // your wgpu::Queue here;
//!
//! let render_context = RenderContext::new(isf_shader_source, format, &device, &queue).unwrap();
//!
//! // Congratulations! You now have a 255x255 blue square.
//! let output = render_context.render_to_vec(&queue, &device, 255, 255);
//!
//! ```
//! The valid document pragmas are as follows.
//!
//! ### Utility Blocks
//!
//! The Tweak Shader Library allows you to utilize utility blocks to access specific uniform or push constant fields efficiently. Here's an example of how to use utility blocks:
//!
//! ```glsl
//! #pragma utility_block(ShaderInputs)
//!layout(push_constant) uniform ShaderInputs {
//!    float time;       // shader playback time (in seconds)
//!    float time_delta; // elapsed time since the last frame in seconds
//!    float frame_rate; // estimated number of frames per second
//!    uint frame_index; // frame count
//!    vec4 mouse;       // ShaderToy mouse scheme
//!    vec4 date;        // [year, month, day, seconds]
//!    vec3 resolution;  // viewport resolution in pixels, [width, height, aspect ratio]
//!    uint pass_index;   // updated to reflect the current render pass index
//!};
//! ```
//!
//! You can use the `#pragma utility_block` to access members from the specialized utility functions, such as [`RenderContext::update_time`] and [`RenderContext::update_resolution`]. The field names may vary between uses safely.
//!
//! ### Input Types
//!
//! Input pragmas provide information about shader inputs, including type, name, and optional attributes such as default values, minimum and maximum bounds, labels, and valid values. Here are some examples of input types:
//!
//! - Float Input:
//!
//! ```glsl
//! #pragma input(float, name="foo", default=0.0, min=0.0, max=1.0)
//! ```
//!
//! - Integer Input with Labels:
//!
//! ```glsl
//! #pragma input(int, name="mode", default=0, values=[0, 1, 2], labels=["A", "B", "C"])
//! ```
//!
//! - Image Input:
//!
//! ```glsl
//! #pragma input(image, name="input_image", path="./demo.png")
//!layout(set=1, binding=1) uniform sampler default_sampler;
//!layout(set=1, binding=2) uniform texture2D input_image;
//! ```
//!
//!
//! Each input pragma corresponds to a uniform variable in the shader code, with the `name` field specifying the matching struct field in the global uniform value or the texture name that maps to the input.
//!
//! ### Additional Render Passes and Persistent Buffers
//!
//! You can define additional render passes and specify output targets for each pass using the `#pragma pass` pragma. Here's an example of how to create an additional pass:
//!
//! ```glsl
//! #pragma pass(0, persistent, target="single_pixel", height=1, width=1)
//!layout(set=0, binding=1) uniform sampler default_sampler;
//!layout(set=0, binding=2) uniform texture2D single_pixel;
//! ```
//!
//! The `#pragma pass` pragma allows you to add passes that run in the order specified by their index before the main pass. If a `target` is specified, the pass will write to a context-managed texture mapped to the specified variable. You can also specify custom `height` and `width` for the output texture; otherwise, it defaults to the render target's size.
//!
//! Please refer to the official documentation for more details and examples on using the Tweak Shader Library in your Rust project.
//!
//! [GitHub Repository](https://github.com/mobile-bungalow/tweak-shader)
//!

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

/// Joint error type
#[derive(Debug, Error)]
pub enum Error {
    /// Thrown if the shader compilation fails, includes a newline separated list of compile errors.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    /// Thrown if a pragma is malformed
    #[error("Document parsing failed: {0}")]
    DocumentParsingFailed(#[from] crate::parsing::Error),

    /// Thrown if uniforms were an unsupported or unexpected format
    /// or not present. Note that currently naga omits unused global
    /// variables and uniforms - this may cause this error to be erroneously thrown.
    #[error("Uniform setup failed: {0}")]
    UniformError(#[from] crate::uniforms::Error),
}
