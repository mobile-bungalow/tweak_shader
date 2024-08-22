use thiserror::Error;

const GLOBAL_EXAMPLES: &str = r#"
#pragma utility_block(ShaderInputs)
layout(set = 0, binding = 0) uniform ShaderInputs {
    float time;       // shader playback time (in seconds)
    float time_delta; // elapsed time since last frame in secs
    float frame_rate; // number of frames per second estimates
    int frame_index;  // frame count
    vec4 mouse;       // xy is last mouse down position,  abs(zw) is current mouse, sign(z) > 0.0 is mouse_down, sign(w) > 0.0 is click_down event
    vec4 date;        // [year, month, day, seconds]
    vec3 resolution;  // viewport resolution in pixels, [w, h, w/h]
    int pass_index;   // updated to reflect render pass
};
"#;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Could not find buffer with name {0}")]
    NoBuffer(String),

    #[error("Pass {0} was not compute compatible, compute passes can only specify an index, targets are managed through relays.")]
    ComputePass(usize),

    #[error("Length specified for buffer of static length: {0}")]
    LengthForNondynamicBuffer(String),

    #[error("Buffer(s) specified with no buffer present in pipeline: {0:?}")]
    MissingBuffer(Vec<String>),

    #[error("Tried to set output to nonexistant Target {0}.")]
    NonexistantTarget(String),

    #[error("Tried to set output to target without the `screen` specifier {0}.")]
    NotScreenTarget(String),

    #[error("Tried to set output target to a uniform that is not an output compatible storage texture {0}.")]
    NotStorageTexture(String),

    #[error("A Naga Arena was missing a handle it said it had, this might be a Naga bug.")]
    Handle,

    #[error("Only 2D Textures are supported at this time.")]
    TextureDimension,

    #[error("Inputs specified but no matching uniform found: {0:?}")]
    MissingInput(Vec<String>),

    #[error("Targets found with the `screen` attribute that do not have a copy compatible format with the output texture: {0:?}, must be: {1:?}")]
    TargetFormatMismatch(Vec<(wgpu::TextureFormat, String)>, wgpu::TextureFormat),

    #[error("Validation error: {0}")]
    TargetValidation(String),

    #[error("Unsupported uniform type: {0:?}")]
    UnsupportedUniformType(String),

    #[error("Unsupported image dimension: {0:?}")]
    UnsupportedImageDim(String),

    #[error("Error loading {0}, uniforms with array dimensions are unsupported at this time.")]
    UnsupportedArrayType(String),

    #[error("Mismatched types found: {0}, expected {1}")]
    InputTypeErr(String, String),

    #[error("Type check failed for input variable: '{0}'")]
    TypeCheck(String),

    #[error("Target specified but no matching uniform found: {0:?}")]
    MissingTarget(Vec<String>),

    #[error("The utility block specified in the pragma does not match the expected layout. \n it should match this layout - \n {}", GLOBAL_EXAMPLES)]
    UtilityBlockType,

    #[error("The utility block specified `{0}` does not exist")]
    UtilityBlockMissing(String),

    #[error("Multiple uniforms declared as `push_constant`, there can only be one.")]
    MultiplePushConstants,

    #[error("Push constant was defined outside of a struct block.")]
    PushConstantOutSideOfBlock,
}
