mod naga_bridge;
mod validation;

use crate::input_type::*;
use crate::parsing::Document;

use naga::{AddressSpace, ResourceBinding, StructMember};
use naga::{FastHashMap, FastHashSet, FastIndexMap};
pub use naga_bridge::*;
use wgpu::naga;

use bytemuck::*;
use wgpu::{util::DeviceExt, BufferUsages};

use crate::VarName;
use thiserror::Error;
use wgpu::{BindGroupLayout, ShaderStages};

macro_rules! extract {
    ($expression:expr, $(
        $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? => $output:expr
    ),+ $(,)?) => {
        match $expression {
            $($( $pattern )|+ $( if $guard )? => Some($output),)+
            _ => None
        }
    }
}

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

const DEFAULT_SAMPLER: wgpu::SamplerDescriptor = wgpu::SamplerDescriptor {
    label: Some("Default Sampler"),
    address_mode_u: wgpu::AddressMode::ClampToEdge,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Nearest,
    min_filter: wgpu::FilterMode::Nearest,
    mipmap_filter: wgpu::FilterMode::Nearest,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: 1,
    border_color: None,
};

const DEFAULT_VIEW: wgpu::TextureViewDescriptor = wgpu::TextureViewDescriptor {
    label: Some("Default View"),
    format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
    dimension: Some(wgpu::TextureViewDimension::D2),
    aspect: wgpu::TextureAspect::All,
    base_mip_level: 0,
    mip_level_count: Some(1),
    base_array_layer: 0,
    array_layer_count: Some(1),
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("A Naga Arena was missing a handle it said it had, this might be a Naga bug.")]
    Handle,

    #[error("Only 2D Textures are supported at this time.")]
    TextureDimension,

    #[error("Inputs specified but no matching uniform found: {0:?}")]
    MissingInput(Vec<String>),

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

#[derive(Debug, Copy, Clone)]
struct VariableAddress {
    // actually just index into set list.
    pub set: usize,
    // actually just index into bind group list
    pub binding: usize,
}

#[derive(Debug)]
pub struct Uniforms {
    /// mapping of names to (set, binding) pairs for faster lookup in the
    /// sets vector
    lookup_table: FastHashMap<VarName, VariableAddress>,
    /// Textures that are either in render targets or not bound
    /// ton any inputs
    private_textures: FastHashSet<VarName>,
    /// The binding location of the utility block if it exists.
    utility_block_addr: Option<VariableAddress>,
    /// The push constants if they exist
    push_constants: Option<PushConstant>,
    /// all (sets, binding) pairs
    sets: Vec<TweakBindGroup>,
    /// Backing for the utility block, regardless if it is used or exists
    /// in the shader.
    utility_block_data: GlobalData,
    /// A place holder texture with one transparent pixel in case the
    /// user doesn't bind a texture, we can pass this in.
    place_holder_texture: wgpu::Texture,
    /// a buffer use to hold pass indices 0..last index.
    /// This is needed in case pass index is not in the push constants and we can't update
    /// it reliably between draw passes
    pass_indices: wgpu::Buffer,
    /// The output texture of the render context, kept for convenience.
    format: wgpu::TextureFormat,
}

impl Uniforms {
    pub fn new(
        document: &Document,
        format: &wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sets: Vec<TweakBindGroup>,
        push_constants: Option<PushConstant>,
        pass_count: usize,
    ) -> Result<Self, Error> {
        let mut lookup_table = FastHashMap::default();
        let mut utility_block_addr = None;

        // Fill out lookup table
        for (set_idx, set) in sets.iter().enumerate() {
            for (binding_idx, binding) in set.binding_entries.iter().enumerate() {
                match binding {
                    BindingEntry::UtilityUniformBlock { .. } => {
                        utility_block_addr = Some(VariableAddress {
                            set: set_idx,
                            binding: binding_idx,
                        });
                    }
                    BindingEntry::UniformBlock { inputs, .. } => {
                        for (name, _) in inputs.iter() {
                            lookup_table.insert(
                                name.clone(),
                                VariableAddress {
                                    set: set_idx,
                                    binding: binding_idx,
                                },
                            );
                        }
                    }
                    BindingEntry::Texture { name, .. }
                    | BindingEntry::Sampler { name, .. }
                    | BindingEntry::StorageTexture { name, .. } => {
                        lookup_table.insert(
                            name.clone(),
                            VariableAddress {
                                set: set_idx,
                                binding: binding_idx,
                            },
                        );
                    }
                }
            }
        }

        let place_holder_texture = device.create_texture_with_data(
            queue,
            &txtr_desc(1, 1),
            Default::default(),
            &[0, 0, 0, 255u8],
        );

        let mut utility_block_data: GlobalData = bytemuck::Zeroable::zeroed();
        utility_block_data.mouse = [0.0, 0.0, -0.0, -0.0];

        let mut private_textures: FastHashSet<_> = document
            .passes
            .iter()
            .filter_map(|pass| pass.target_texture.clone())
            .collect();

        for tex in sets
            .iter()
            .flat_map(|set| &set.binding_entries)
            .filter_map(|entry| extract!(entry, BindingEntry::Texture { ref name, ..} => name))
        {
            let texture_in_doc = document.inputs.contains_key(tex);
            if !texture_in_doc {
                private_textures.insert(tex.clone());
            }
        }

        // During the render pass we need to queue `copy_buffer_to_buffer`
        // calls in order to update the index in a predictable way. but only
        // to update the `pass_index` utility block member.
        let contents = (0..pass_count as u32).collect::<Vec<u32>>();
        let pass_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&contents),
            label: None,
            usage: BufferUsages::COPY_SRC,
        });

        let out = Self {
            pass_indices,
            push_constants,
            private_textures,
            utility_block_addr,
            lookup_table,
            sets,
            format: *format,
            place_holder_texture,
            utility_block_data,
        };

        out.validate(document, format)?;

        Ok(out)
    }

    // Copy this uniforms data into other - goes by name
    pub fn copy_into(&mut self, other: &mut Self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Copy Uniforms
        other.utility_block_data = self.utility_block_data;

        for (name, addr) in other.lookup_table.iter() {
            let Some(other_value) = other
                .sets
                .get_mut(addr.set)
                .and_then(|s| s.get_mut(name, addr))
            else {
                continue;
            };

            let Some(self_value) = self.query_addr_mut(name, addr) else {
                continue;
            };

            let mut self_input: MutInput = self_value;
            let mut other_input: MutInput = other_value;
            self_input.copy_into(&mut other_input);
        }

        // Copy Textures
        let mut command_encoder = device.create_command_encoder(&Default::default());

        for (name, addr) in self.lookup_table.iter() {
            let Some(mut self_image_input) = self
                .sets
                .get_mut(addr.set)
                .and_then(|s| s.get_mut(name, addr))
            else {
                continue;
            };

            let Some(other_addr) = other.lookup_table.get(name) else {
                continue;
            };

            let Some(mut other_image_input) = other.query_addr_mut(name, &other_addr.clone())
            else {
                continue;
            };

            if let (Some(_), Some(TextureStatus::Loaded { width, height })) = (
                other_image_input.texture_status(),
                self_image_input.texture_status(),
            ) {
                let size = wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                };

                let Some(self_tex) = self.get_texture(name) else {
                    continue;
                };

                let new_tex = if other.private_textures.contains(name) {
                    let mut desc = txtr_desc(width, height);
                    desc.format = wgpu::TextureFormat::Rgba16Float;
                    device.create_texture(&desc)
                } else {
                    device.create_texture(&txtr_desc(width, height))
                };

                command_encoder.copy_texture_to_texture(
                    self_tex.as_image_copy(),
                    new_tex.as_image_copy(),
                    size,
                );

                other.set_texture(name, new_tex);
            }
        }
        queue.submit(Some(command_encoder.finish()));
    }

    pub fn update_uniform_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(addr) = self.utility_block_addr {
            if let Some(BindingEntry::UtilityUniformBlock { backing, .. }) = self
                .sets
                .get_mut(addr.set)
                .and_then(|set| set.binding_entries.get_mut(addr.binding))
            {
                *backing = self.utility_block_data
            }
        }

        if let Some(push_constants) = self.push_constants.as_mut() {
            match push_constants {
                PushConstant::UtilityBlock { backing } => *backing = self.utility_block_data,
                PushConstant::Struct {
                    backing,
                    inputs,
                    align,
                    ..
                } => {
                    let mut offset = 0;
                    for (_, input) in inputs.iter() {
                        let bytes = input.as_bytes();

                        let padding = (*align - (offset % *align)) % *align;
                        if bytes.len() > padding {
                            offset += padding;
                        }

                        backing[offset..bytes.len() + offset].copy_from_slice(bytes);
                        offset += bytes.len();
                    }
                }
            }
        }

        for set in self.sets.iter_mut() {
            set.update_uniforms(device, queue, &self.place_holder_texture);
        }
    }

    pub fn unload_texture(&mut self, var: &str) -> bool {
        if self.private_textures.contains(var) {
            return false;
        }

        let Some(addr) = self.lookup_table.get(var) else {
            return false;
        };
        if let Some(set) = self.sets.get_mut(addr.set) {
            set.unload_texture(addr)
        } else {
            false
        }
    }

    pub fn push_constant_bytes(&self) -> Option<&[u8]> {
        match self.push_constants.as_ref() {
            Some(PushConstant::UtilityBlock { backing }) => Some(bytes_of(backing)),
            Some(PushConstant::Struct { backing, .. }) => Some(backing.as_slice()),
            None => None,
        }
    }

    pub fn push_constant_ranges(&self) -> Option<wgpu::PushConstantRange> {
        if let Some(push) = self.push_constants.as_ref() {
            let size = match push {
                PushConstant::UtilityBlock { .. } => std::mem::size_of::<GlobalData>(),
                PushConstant::Struct { backing, .. } => backing.len(),
            };
            Some(wgpu::PushConstantRange {
                stages: ShaderStages::VERTEX_FRAGMENT,
                range: 0..size as u32,
            })
        } else {
            None
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn load_texture(
        &mut self,
        var_name: &str,
        data: &[u8],
        height: u32,
        width: u32,
        bytes_per_row: Option<u32>,
        format: &wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let addr = self.lookup_table.get(var_name).copied();
        let input_type = addr
            .and_then(|addr| self.query_addr_mut(var_name, &addr))
            .and_then(|mut t| t.texture_status());

        let status = match input_type {
            Some(t) => t,
            None => {
                return;
            }
        };

        let wgpu_texture = self.get_texture(var_name);

        // If a texture of identical dimension
        // exists: write to it. otherwise init a new texture with the data.
        let texture = match (&status, wgpu_texture) {
            (_, Some(texture))
                if texture.height() == height
                    && texture.width() == width
                    && texture.format() == *format =>
            {
                texture
            }
            _ => {
                let mut desc = txtr_desc(width, height);
                desc.format = *format;
                let tex = device.create_texture(&desc);
                self.set_texture(var_name, tex);
                self.get_texture(var_name).unwrap()
            }
        };

        let block_size = texture
            .format()
            .block_copy_size(Some(wgpu::TextureAspect::All))
            .expect("It seems like you are trying to render to a Depth Stencil. Stop that.");

        queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: bytes_per_row.or(Some(width * block_size)),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn global_data_mut(&mut self) -> &mut GlobalData {
        &mut self.utility_block_data
    }

    pub fn override_texture_view_with_tex(&mut self, name: &str, new_tex: &wgpu::Texture) -> bool {
        let mut desc = DEFAULT_VIEW;
        desc.format = Some(new_tex.format());
        let tex_view = new_tex.create_view(&desc);
        let width = new_tex.width();
        let height = new_tex.height();

        let Some(addr) = self.lookup_table.get(name) else {
            return false;
        };

        if let Some(set) = self.sets.get_mut(addr.set) {
            set.override_texture_view(height, width, addr, tex_view)
        } else {
            false
        }
    }

    pub fn override_texture_view_with_view(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        new_tex_view: wgpu::TextureView,
    ) -> bool {
        let Some(addr) = self.lookup_table.get(name) else {
            return false;
        };

        if let Some(set) = self.sets.get_mut(addr.set) {
            set.override_texture_view(height, width, addr, new_tex_view)
        } else {
            false
        }
    }

    pub fn set_texture(&mut self, name: &str, tex: wgpu::Texture) -> bool {
        let Some(addr) = self.lookup_table.get(name) else {
            return false;
        };
        if let Some(set) = self.sets.get_mut(addr.set) {
            let out = set.set_texture(addr, tex);
            if out {
                set.needs_rebind = true;
            }
            out
        } else {
            false
        }
    }

    pub fn get_texture(&self, name: &str) -> Option<&wgpu::Texture> {
        let addr = self.lookup_table.get(name)?;
        self.sets.get(addr.set)?.get_texture(addr)
    }

    pub fn set_pass_index(&mut self, index: usize, enc: &mut wgpu::CommandEncoder) {
        self.utility_block_data.pass_index = index as u32;
        if let Some(PushConstant::UtilityBlock { backing }) = self.push_constants.as_mut() {
            backing.pass_index = index as u32;
        }

        if let Some(addr) = self.utility_block_addr {
            if let Some(BindingEntry::UtilityUniformBlock {
                backing, buffer, ..
            }) = self
                .sets
                .get_mut(addr.set)
                .and_then(|set| set.binding_entries.get_mut(addr.binding))
            {
                backing.pass_index = index as u32;

                enc.copy_buffer_to_buffer(
                    &self.pass_indices,
                    index as u64 * 4,
                    buffer,
                    bytemuck::offset_of!(GlobalData, pass_index) as u64,
                    std::mem::size_of::<u32>() as u64,
                );
                //queue.write_buffer(buffer, 0, bytemuck::bytes_of(backing));
            }
        }
    }

    pub fn get_input_mut(&mut self, name: &str) -> Option<MutInput> {
        let out = self.push_constants.as_mut().and_then(|p| match p {
            PushConstant::Struct { inputs, .. } => inputs.get_mut(name).map(|i| i.into()),
            _ => None,
        });
        if out.is_some() {
            out
        } else {
            let addr = self.lookup_table.get(name)?;
            let set = self.sets.get_mut(addr.set)?;
            set.get_mut(name, addr)
        }
    }

    pub fn get_input(&self, name: &str) -> Option<&InputType> {
        let out = self.push_constants.as_ref().and_then(|p| match p {
            PushConstant::Struct { inputs, .. } => inputs.get(name),
            _ => None,
        });

        if out.is_some() {
            out
        } else {
            let addr = self.lookup_table.get(name)?;
            let set = self.sets.get(addr.set)?;
            set.get(name, addr)
        }
    }

    fn query_addr_mut(&mut self, name: &str, addr: &VariableAddress) -> Option<MutInput> {
        let set = self.sets.get_mut(addr.set)?;
        set.get_mut(name, addr)
    }

    pub fn iter_custom_uniforms_mut(&mut self) -> impl Iterator<Item = (&String, MutInput)> {
        let push = if let Some(PushConstant::Struct { inputs, .. }) = self.push_constants.as_mut() {
            Some(Box::new(inputs.iter_mut().map(|(k, v)| (k, v.into())))
                as Box<dyn Iterator<Item = _>>)
        } else {
            None
        };

        self.sets
            .iter_mut()
            .flat_map(|b| b.binding_entries.iter_mut())
            .filter_map(|entry| match entry {
                BindingEntry::UniformBlock { inputs, .. } => {
                    let iter = inputs.iter_mut().filter_map(|(k, v)| {
                        if !matches!(v, InputType::RawBytes(_)) && self.lookup_table.contains_key(k)
                        {
                            Some((k, v.into()))
                        } else {
                            None
                        }
                    });
                    Some(Box::new(iter) as _)
                }
                BindingEntry::Texture { input, name, .. } => {
                    if self.private_textures.contains(name) {
                        None
                    } else {
                        Some(Box::new(std::iter::once_with(move || (&*name, input.into()))) as _)
                    }
                }
                _ => None,
            })
            .chain(push)
            .flatten()
    }

    pub fn iter_custom_uniforms(&self) -> impl Iterator<Item = (&String, &InputType)> {
        let push = if let Some(PushConstant::Struct { inputs, .. }) = self.push_constants.as_ref() {
            Some(Box::new(inputs.iter()) as Box<dyn Iterator<Item = _>>)
        } else {
            None
        };

        self.sets
            .iter()
            .flat_map(|b| b.binding_entries.iter())
            .filter_map(|entry| match entry {
                BindingEntry::UniformBlock { inputs, .. } => {
                    let iter = inputs.iter().filter(|(k, v)| {
                        !matches!(v, InputType::RawBytes(_)) && self.lookup_table.contains_key(*k)
                    });
                    Some(Box::new(iter) as _)
                }
                BindingEntry::Texture { input, name, .. } => {
                    if self.private_textures.contains(name) {
                        None
                    } else {
                        Some(Box::new(std::iter::once_with(move || (name, input))) as _)
                    }
                }
                _ => None,
            })
            .chain(push)
            .flatten()
    }

    pub fn iter_sets(&self) -> impl Iterator<Item = (u32, &wgpu::BindGroup)> {
        self.sets.iter().map(|set| (set.set, &set.bind_group))
    }

    pub fn iter_layouts(&self) -> impl Iterator<Item = &wgpu::BindGroupLayout> {
        self.sets.iter().map(|set| &set.layout)
    }
}

pub fn sets(
    module: &naga::Module,
    document: &crate::parsing::Document,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: &wgpu::TextureFormat,
) -> Result<Vec<TweakBindGroup>, Error> {
    let max_set_index = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| var.binding.as_ref().map(|b| b.group))
        .max()
        .unwrap_or(0);

    // collect all set indices, find the max then create bind sets contiguous up to max.
    // some might be empty.
    (0..=max_set_index)
        .map(|set| TweakBindGroup::new_from_naga(set, &module, &document, device, queue, &format))
        .collect::<Result<_, _>>()
}

// CPU representation of the shadertoy-like bind group
// This is uploaded to the gpu using std430 memory layout
// keep that in mind when editing this structure
// It also must be 64 byte aligned
#[repr(C)]
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Pod, Zeroable)]
pub struct GlobalData {
    pub time: f32,
    pub time_delta: f32,
    pub frame_rate: f32,
    pub frame: u32,
    pub mouse: [f32; 4],
    pub date: [f32; 4],
    pub resolution: [f32; 3],
    pub pass_index: u32,
}

impl Default for GlobalData {
    fn default() -> Self {
        let mut out: Self = bytemuck::Zeroable::zeroed();
        out.mouse = [0.0, 0.0, -0.0, -0.0];
        out
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Storage {
    Uniform,
    Push,
    TextureAccess(wgpu::StorageTextureAccess),
}

#[derive(Debug)]
pub enum BindingEntry {
    UtilityUniformBlock {
        binding: u32,
        // Global layout data
        backing: crate::uniforms::GlobalData,
        // buffer this uniform is mapped to
        buffer: wgpu::Buffer,
        // storage location
        storage: Storage,
    },
    UniformBlock {
        backing: Vec<u8>,
        // the binding index , might not be contiguous
        binding: u32,
        // inputs with a non `raw_bytes` type if they are mapped in the document
        inputs: FastIndexMap<String, InputType>,
        // buffer this uniform is mapped to
        buffer: wgpu::Buffer,
        // the largest struct size in the inputs
        align: usize,
        // storage location
        storage: Storage,
    },
    StorageTexture {
        // the binding index , might not be contiguous
        binding: u32,
        // texture resource if not default
        tex: wgpu::Texture,
        view: wgpu::TextureView,
        // variable name
        name: String,
        // storage location
        storage: Storage,
        // can be copied to view
        supports_screen: bool,
    },
    Texture {
        // the binding index , might not be contiguous
        binding: u32,
        // texture resource if not default
        tex: Option<wgpu::Texture>,
        view: Option<wgpu::TextureView>,
        // image status for bookeeping
        input: InputType,
        // variable name
        name: String,
        // likely just uniform
        storage: Storage,
    },
    Sampler {
        // the binding index , might not be contiguous
        binding: u32,
        // texture resource if not default
        samp: wgpu::Sampler,
        name: String,
    },
}

struct StructDescriptor<'a> {
    padded_size: usize,
    name: String,
    storage: Storage,
    binding: u32,
    members: &'a [StructMember],
}

impl BindingEntry {
    fn new_struct_entry(
        device: &wgpu::Device,
        module: &naga::Module,
        document: &crate::parsing::Document,
        desc: StructDescriptor,
    ) -> Result<Self, Error> {
        let StructDescriptor {
            padded_size,
            name,
            storage,
            binding,
            members,
        } = desc;
        // if this is the utility block make sure it lines
        // up then return it.
        if document
            .utility_block_name
            .as_ref()
            .is_some_and(|util_name| *util_name == name)
        {
            let layout = members.iter().filter_map(|member| {
                let ty = module.types.get_handle(member.ty).ok()?;
                let offset = member.offset as usize;
                Some((ty.inner.clone(), offset))
            });

            GlobalData::validate(layout)?;

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<crate::uniforms::GlobalData>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            return Ok(Self::UtilityUniformBlock {
                binding,
                backing: Default::default(),
                buffer,
                storage,
            });
        }

        let mut inputs = FastIndexMap::default();

        for member in members {
            let name = member.name.clone().unwrap_or_default();
            let ty = module
                .types
                .get_handle(member.ty)
                .map_err(|_| Error::Handle)?;

            if let Some(var) = document.inputs.get(&name) {
                var.validate(&ty.inner)?;
                inputs.insert(name, var.clone());
            } else {
                let input = InputType::RawBytes(crate::input_type::RawBytes {
                    inner: vec![0; ty.inner.size(module.to_ctx()) as usize],
                });
                inputs.insert(name, input);
            }
        }

        let align = inputs
            .iter()
            .map(|(_, i)| match i {
                InputType::Point(_) => std::mem::size_of::<[f32; 4]>(),
                _ => i.as_bytes().len(),
            })
            .max()
            .unwrap_or(0);

        let align = (align + 15) / 16 * 16;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_size as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self::UniformBlock {
            align,
            backing: vec![0u8; padded_size],
            binding,
            inputs,
            buffer,
            storage,
        })
    }

    pub fn storage(&self) -> Storage {
        match self {
            BindingEntry::UtilityUniformBlock { storage, .. }
            | BindingEntry::UniformBlock { storage, .. }
            | BindingEntry::StorageTexture { storage, .. }
            | BindingEntry::Texture { storage, .. } => *storage,
            BindingEntry::Sampler { .. } => Storage::Uniform,
        }
    }
}

#[derive(Debug)]
pub enum PushConstant {
    UtilityBlock {
        backing: crate::uniforms::GlobalData,
    },
    Struct {
        backing: Vec<u8>,
        inputs: FastIndexMap<String, InputType>,
        align: usize,
    },
}

pub fn push_constant(
    module: &naga::Module,
    document: &crate::parsing::Document,
) -> Result<Option<PushConstant>, Error> {
    let push_constants: Vec<_> = module
        .global_variables
        .iter()
        .filter(|(_, var)| var.space == AddressSpace::PushConstant)
        .collect();

    let (_, push_constant) = match push_constants.as_slice() {
        [one] => one,
        [] => return Ok(None),
        _ => {
            return Err(Error::MultiplePushConstants);
        }
    };

    let push_type = module
        .types
        .get_handle(push_constant.ty)
        .map_err(|_| Error::Handle)?;

    let naga::TypeInner::Struct { members, span } = &push_type.inner else {
        return Err(Error::PushConstantOutSideOfBlock);
    };

    if document
        .utility_block_name
        .as_ref()
        .is_some_and(|util_name| *util_name == push_type.name.clone().unwrap_or_default())
    {
        let layout = members.iter().filter_map(|member| {
            let ty = module.types.get_handle(member.ty).ok()?;
            let offset = member.offset as usize;
            Some((ty.inner.clone(), offset))
        });

        GlobalData::validate(layout)?;

        Ok(Some(PushConstant::UtilityBlock {
            backing: Default::default(),
        }))
    } else {
        let mut inputs = FastIndexMap::default();

        for member in members {
            let name = member.name.clone().unwrap_or_default();

            let ty = module
                .types
                .get_handle(member.ty)
                .map_err(|_| Error::Handle)?;

            if let Some(var) = document.inputs.get(&name) {
                var.validate(&ty.inner)?;
                inputs.insert(name, var.clone());
            } else {
                let input = InputType::RawBytes(crate::input_type::RawBytes {
                    inner: vec![0; ty.inner.size(module.to_ctx()) as usize],
                });
                inputs.insert(name.clone(), input.clone());
            }
        }

        let align = wgpu::PUSH_CONSTANT_ALIGNMENT as usize;

        Ok(Some(PushConstant::Struct {
            backing: vec![0; *span as usize],
            inputs,
            align,
        }))
    }
}

#[derive(Debug)]
pub struct TweakBindGroup {
    // the Bind set
    pub set: u32,
    pub binding_entries: Vec<BindingEntry>,
    //true if the bind group needs a rebuild
    pub needs_rebind: bool,
    // the bind group, frequently rebound
    pub bind_group: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
}

impl TweakBindGroup {
    fn get_mut(&mut self, name: &str, addr: &VariableAddress) -> Option<MutInput> {
        match self.binding_entries.get_mut(addr.binding)? {
            BindingEntry::Texture { input, .. } => Some(input.into()),
            BindingEntry::UniformBlock { inputs, .. } => inputs.get_mut(name).map(|i| i.into()),
            _ => None,
        }
    }

    fn get_texture(&self, addr: &VariableAddress) -> Option<&wgpu::Texture> {
        match self.binding_entries.get(addr.binding)? {
            BindingEntry::Texture { tex, .. } => tex.as_ref(),
            _ => None,
        }
    }

    fn unload_texture(&mut self, addr: &VariableAddress) -> bool {
        match self.binding_entries.get_mut(addr.binding) {
            Some(BindingEntry::Texture {
                tex, view, input, ..
            }) => {
                self.needs_rebind = true;
                *tex = None;
                *view = None;
                let status = match input {
                    InputType::Image(status) => status,
                    _ => {
                        return false;
                    }
                };
                *status = TextureStatus::Uninit;
                true
            }
            _ => false,
        }
    }

    fn override_texture_view(
        &mut self,
        height: u32,
        width: u32,
        addr: &VariableAddress,
        new_view: wgpu::TextureView,
    ) -> bool {
        match self.binding_entries.get_mut(addr.binding) {
            Some(BindingEntry::Texture { view, input, .. }) => {
                let status = match input {
                    InputType::Image(status) => status,
                    _ => {
                        return false;
                    }
                };
                *view = Some(new_view);
                *status = TextureStatus::Loaded { width, height };
                self.needs_rebind = true;
                true
            }
            _ => false,
        }
    }

    fn set_texture(&mut self, addr: &VariableAddress, new_tex: wgpu::Texture) -> bool {
        match self.binding_entries.get_mut(addr.binding) {
            Some(BindingEntry::Texture {
                tex, view, input, ..
            }) => {
                let mut desc = DEFAULT_VIEW;
                desc.format = Some(new_tex.format());
                let new_view = new_tex.create_view(&desc);
                let width = new_tex.width();
                let height = new_tex.height();
                *view = Some(new_view);
                *tex = Some(new_tex);
                let status = match input {
                    InputType::Image(status) => status,
                    _ => {
                        return false;
                    }
                };
                *status = TextureStatus::Loaded { width, height };
                self.needs_rebind = true;
                true
            }
            _ => false,
        }
    }

    fn get(&self, name: &str, addr: &VariableAddress) -> Option<&InputType> {
        match self.binding_entries.get(addr.binding)? {
            BindingEntry::Texture { input, .. } => Some(input),
            BindingEntry::UniformBlock { inputs, .. } => inputs.get(name),
            _ => None,
        }
    }

    pub fn contains_util(&self) -> bool {
        self.binding_entries
            .iter()
            .any(|e| matches!(e, BindingEntry::UtilityUniformBlock { .. }))
    }

    fn update_uniforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        place_holder_tex: &wgpu::Texture,
    ) {
        if self.needs_rebind {
            self.bind_group = Self::update_uniforms_helper(
                &mut self.binding_entries,
                place_holder_tex,
                &self.layout,
                device,
                queue,
            );
            self.needs_rebind = false;
        } else {
            for binding in self.binding_entries.iter_mut() {
                match binding {
                    BindingEntry::UtilityUniformBlock {
                        backing, buffer, ..
                    } => {
                        queue.write_buffer(buffer, 0, bytemuck::bytes_of(backing));
                    }
                    BindingEntry::UniformBlock {
                        ref mut backing,
                        inputs,
                        buffer,
                        align,
                        ..
                    } => {
                        let mut offset = 0;
                        let mut changed = false;
                        for input in inputs.iter() {
                            let bytes = input.1.as_bytes();

                            let padding = (*align - (offset % *align)) % *align;
                            if bytes.len() > padding {
                                offset += padding;
                            }

                            changed |= &backing[offset..bytes.len() + offset] != bytes;
                            backing[offset..bytes.len() + offset].copy_from_slice(bytes);
                            offset += bytes.len();
                        }
                        if changed {
                            queue.write_buffer(buffer, 0, backing);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn update_uniforms_helper<'a>(
        bindings: &'a mut [BindingEntry],
        place_holder_tex: &'a wgpu::Texture,
        layout: &BindGroupLayout,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::BindGroup {
        let placeholder_view = place_holder_tex.create_view(&DEFAULT_VIEW);
        let mut out = Vec::new();
        for binding in bindings.iter_mut() {
            match binding {
                BindingEntry::UtilityUniformBlock {
                    binding,
                    backing,
                    buffer,
                    ..
                } => {
                    queue.write_buffer(buffer, 0, bytemuck::bytes_of(backing));
                    out.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    })
                }
                BindingEntry::UniformBlock {
                    ref mut backing,
                    binding,
                    inputs,
                    buffer,
                    align,
                    ..
                } => {
                    let mut offset = 0;
                    let mut changed = false;
                    for input in inputs.values() {
                        let bytes = input.as_bytes();

                        // Calculate the padding needed to satisfy the alignment requirement
                        let padding = (*align - (offset % *align)) % *align;

                        if bytes.len() > padding {
                            offset += padding;
                        }

                        changed |= &backing[offset..bytes.len() + offset] != bytes;
                        backing[offset..bytes.len() + offset].copy_from_slice(bytes);
                        offset += bytes.len();
                    }
                    if changed {
                        queue.write_buffer(buffer, 0, backing);
                    }
                    out.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    })
                }
                BindingEntry::Texture { binding, view, .. } => {
                    if let Some(view) = view {
                        out.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::TextureView(view),
                        });
                    } else {
                        out.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::TextureView(&placeholder_view),
                        });
                    };
                }
                BindingEntry::Sampler { binding, samp, .. } => {
                    out.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::Sampler(&*samp),
                    });
                }
                BindingEntry::StorageTexture { binding, view, .. } => {
                    out.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
            }
        }

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: out.as_slice(),
        })
    }
}
