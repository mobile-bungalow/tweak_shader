use crate::input_type::*;
use crate::parsing::DocumentDescriptor;
use __core::num::NonZeroU32;
use bytemuck::{checked::cast_slice, offset_of};
use naga::{AddressSpace, ResourceBinding, StorageAccess, StructMember};
use wgpu::{naga, TextureFormat};

use std::collections::{BTreeMap, BTreeSet};

use bytemuck::*;
use wgpu::{util::DeviceExt, BufferUsages};

use crate::VarName;
use std::fmt;
use wgpu::{BindGroupLayout, ShaderStages};

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

#[derive(Debug)]
pub enum Error {
    Handle,
    MissingInput(Vec<String>),
    UnsupportedUniformType(String),
    UnsupportedImageDim(String),
    UnsupportedArrayType(String),
    InputTypeErr(String, &'static str),
    TypeCheck(String),
    UtilityBlockType,
    UtilityBlockMissing(String),
    MultiplePushConstants,
    PushConstantOutSideOfBlock,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::PushConstantOutSideOfBlock => {
                write!(f, "Push constant was defined outside of a struct block.")
            }
            Error::MultiplePushConstants => write!(
                f,
                "Multiple uniforms declared as `push_constant`, there can only be one."
            ),
            Error::Handle => write!(
                f,
                "A Naga Arena was missing a handle it said it had, this might be a Naga bug."
            ),
            Error::MissingInput(vars) => write!(
                f,
                "Inputs specified but no matching uniform found: {}",
                vars.join(", ")
            ),
            Error::UnsupportedUniformType(ty) => {
                write!(f, "Unsupported uniform type: {:?}", ty)
            }
            Error::UnsupportedImageDim(dim) => {
                write!(f, "Unsupported image dimension: {:?}", dim)
            }
            Error::UnsupportedArrayType(binding) => {
                write!(f, "Error loading {binding}, uniforms with array dimensions are unsupported at this time.")
            }
            Error::InputTypeErr(var, expected) => {
                write!(f, "Mismatched types found: {} -> {}", var, expected)
            }
            Error::TypeCheck(name) => {
                write!(f, "Type check failed for input variable: '{}'", name)
            }
            Error::UtilityBlockType => {
                write!(
                    f,
                    "The utility block specified in the pragma does not match the expected layout. \n it should match this layout - \n {}", 
                    GLOBAL_EXAMPLES
                )
            }
            Error::UtilityBlockMissing(name) => {
                write!(f, "The utility block specified `{}` does not exist", name)
            }
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Copy, Clone)]
struct VariableAddress {
    // actually just index into set list.
    pub set: usize,
    // actually just index into bind group list
    pub binding: usize,
    pub field: Option<usize>,
}

#[derive(Debug)]
pub struct Uniforms {
    lookup_table: BTreeMap<VarName, VariableAddress>,
    render_pass_targets: BTreeSet<VarName>,
    utility_block_addr: Option<VariableAddress>,
    push_constants: Option<PushConstant>,
    sets: Vec<TweakBindGroup>,
    utility_block_data: GlobalData,
    place_holder_texture: wgpu::Texture,
    pass_indices: wgpu::Buffer,
    format: wgpu::TextureFormat,
}

impl Uniforms {
    pub fn new(
        desc: &DocumentDescriptor,
        format: &wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sets: Vec<TweakBindGroup>,
        push_constants: Option<PushConstant>,
        pass_count: usize,
    ) -> Result<Self, Error> {
        let mut lookup_table = BTreeMap::new();
        let mut push_addr = None;
        let mut utility_block_addr = None;

        // Fill out lookup table
        for (set_idx, set) in sets.iter().enumerate() {
            for (binding_idx, binding) in set.binding_entries.iter().enumerate() {
                if binding.storage() == Storage::Push {
                    if push_addr.is_some() {
                        return Err(Error::MultiplePushConstants);
                    }
                    push_addr = Some(VariableAddress {
                        set: set_idx,
                        binding: binding_idx,
                        field: None,
                    })
                }
                match binding {
                    BindingEntry::UtilityUniformBlock { .. } => {
                        utility_block_addr = Some(VariableAddress {
                            set: set_idx,
                            binding: binding_idx,
                            field: None,
                        });
                    }
                    BindingEntry::UniformBlock { inputs, .. } => {
                        for (field_idx, field) in inputs.iter().enumerate() {
                            lookup_table.insert(
                                field.0.clone(),
                                VariableAddress {
                                    set: set_idx,
                                    binding: binding_idx,
                                    field: Some(field_idx),
                                },
                            );
                        }
                    }
                    BindingEntry::Texture { name, .. } | BindingEntry::Sampler { name, .. } => {
                        lookup_table.insert(
                            name.clone(),
                            VariableAddress {
                                set: set_idx,
                                binding: binding_idx,
                                field: None,
                            },
                        );
                    }
                }
            }
        }

        // Look for input pragmas that are missing targets
        let missing_input: Vec<_> = desc
            .inputs
            .keys()
            .filter(|key| {
                let not_in_push_constants =
                    if let Some(PushConstant::Struct { input_map, .. }) = &push_constants {
                        !input_map.contains_key(*key)
                    } else {
                        true
                    };
                let not_in_uni = sets.iter().all(|binding| {
                    lookup_table
                        .get(*key)
                        .map(|addr| binding.get(addr))
                        .is_none()
                });
                not_in_push_constants && not_in_uni
            })
            .cloned()
            .collect();

        // error out early if a specified input does not exists
        if !missing_input.is_empty() {
            Err(Error::MissingInput(missing_input))?;
        }

        let no_util_present = !sets.iter().any(|b| b.contains_util());
        let push_is_util = push_constants
            .as_ref()
            .is_some_and(|p| matches!(p, PushConstant::UtilityBlock { .. }));

        // error out early a utility block was specified and not found
        if desc.utility_block_name.is_some() && no_util_present && !push_is_util {
            Err(Error::UtilityBlockMissing(
                desc.utility_block_name.as_ref().unwrap().clone(),
            ))?;
        }

        let place_holder_texture = device.create_texture_with_data(
            queue,
            &txtr_desc(1, 1),
            Default::default(),
            &[0, 0, 0, 255u8],
        );

        let mut utility_block_data: GlobalData = bytemuck::Zeroable::zeroed();
        utility_block_data.mouse = [0.0, 0.0, -0.0, -0.0];

        let render_pass_targets = desc
            .passes
            .iter()
            .filter_map(|pass| pass.target_texture.clone())
            .collect();

        let contents = (0..pass_count as u32).collect::<Vec<u32>>();

        // During the render pass we need to queue `copy_buffer_to_buffer`
        // calls in order to update the index in a predictable way. but only
        // to update the `pass_index` utility block member.
        let pass_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: cast_slice(&contents),
            label: None,
            usage: BufferUsages::COPY_SRC,
        });

        Ok(Self {
            pass_indices,
            push_constants,
            render_pass_targets,
            utility_block_addr,
            lookup_table,
            sets,
            format: *format,
            place_holder_texture,
            utility_block_data,
        })
    }

    // Copy this uniforms data into other - goes by name
    pub fn copy_into(&mut self, other: &mut Self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Copy Uniforms
        other.utility_block_data = self.utility_block_data;

        for addr in other.lookup_table.values() {
            let Some(other_value) = other.sets.get_mut(addr.set).and_then(|s| s.get_mut(addr))
            else {
                continue;
            };

            let Some(self_value) = self.query_addr_mut(addr) else {
                continue;
            };

            let mut self_input: MutInput = self_value;
            let mut other_input: MutInput = other_value;
            self_input.copy_into(&mut other_input);
        }

        // Copy Textures
        let mut command_encoder = device.create_command_encoder(&Default::default());

        for (name, addr) in self.lookup_table.iter() {
            let Some(mut self_image_input) =
                self.sets.get_mut(addr.set).and_then(|s| s.get_mut(addr))
            else {
                continue;
            };

            let Some(other_addr) = other.lookup_table.get(name) else {
                continue;
            };

            let Some(mut other_image_input) = other.query_addr_mut(&other_addr.clone()) else {
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

                let new_tex = if other.render_pass_targets.contains(name) {
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
        if self.render_pass_targets.contains(var) {
            return false;
        }

        let Some(addr) = self.lookup_table.get(var) else {
            return false;
        };
        if let Some(set) = self.sets.get_mut(addr.set) {
            let out = set.unload_texture(addr);
            if out {
                set.needs_rebind = true;
            }
            out
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
        variable_name: &str,
        data: &[u8],
        height: u32,
        width: u32,
        bytes_per_row: Option<u32>,
        format: &wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let addr = self.lookup_table.get(variable_name).copied();
        let input_type = addr
            .and_then(|addr| self.query_addr_mut(&addr))
            .and_then(|mut t| t.texture_status());

        let status = match input_type {
            Some(t) => t,
            None => {
                return;
            }
        };

        let wgpu_texture = self.get_texture(variable_name);

        // If a texture of identical dimension
        // exists: write to it. otherwise init a new texture with the data.
        match (&status, wgpu_texture) {
            (
                TextureStatus::Loaded {
                    width: w,
                    height: h,
                },
                Some(tex),
            ) if *h == height && *w == width && tex.format() == *format => {
                let block_size = tex
                    .format()
                    .block_copy_size(Some(wgpu::TextureAspect::All))
                    .expect(
                        "It seems like you are trying to render to a Depth Stencil. Stop that.",
                    );
                queue.write_texture(
                    tex.as_image_copy(),
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
            _ => {
                let mut desc = txtr_desc(width, height);
                desc.format = *format;
                let tex = device.create_texture(&desc);

                let block_size = tex
                    .format()
                    .block_copy_size(Some(wgpu::TextureAspect::All))
                    .expect(
                        "It seems like you are trying to render to a Depth Stencil. Stop that.",
                    );

                queue.write_texture(
                    tex.as_image_copy(),
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

                self.set_texture(variable_name, tex);
            }
        }
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
            let out = set.override_texture_view(height, width, addr, tex_view);
            if out {
                set.needs_rebind = true;
            }
            out
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
            let out = set.override_texture_view(height, width, addr, new_tex_view);
            if out {
                set.needs_rebind = true;
            }
            out
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
            PushConstant::Struct {
                inputs, input_map, ..
            } => input_map
                .get(name)
                .and_then(|i| inputs.get_mut(*i))
                .map(|(_, i)| MutInput::from(i)),
            _ => None,
        });
        if out.is_some() {
            out
        } else {
            let addr = self.lookup_table.get(name)?;
            let set = self.sets.get_mut(addr.set)?;
            set.get_mut(addr)
        }
    }

    pub fn get_input(&self, name: &str) -> Option<&InputType> {
        let out = self.push_constants.as_ref().and_then(|p| match p {
            PushConstant::Struct {
                inputs, input_map, ..
            } => input_map.get(name).and_then(|i| inputs.get(*i)),
            _ => None,
        });
        if out.is_some() {
            out.map(|(_, s)| s)
        } else {
            let addr = self.lookup_table.get(name)?;
            let set = self.sets.get(addr.set)?;
            set.get(addr)
        }
    }

    fn query_addr_mut(&mut self, addr: &VariableAddress) -> Option<MutInput> {
        let set = self.sets.get_mut(addr.set)?;
        set.get_mut(addr)
    }

    pub fn iter_custom_uniforms_mut(&mut self) -> impl Iterator<Item = (&str, MutInput)> {
        let mut push =
            if let Some(PushConstant::Struct { inputs, .. }) = self.push_constants.as_mut() {
                Some(
                    inputs
                        .iter_mut()
                        .map(|(k, v)| (k.as_str(), MutInput::from(v))),
                )
            } else {
                None
            };

        let mut field_iter = None;

        let mut set_iter = self
            .sets
            .iter_mut()
            .flat_map(|b| b.binding_entries.iter_mut());

        let render_pass_targets = &self.render_pass_targets;

        std::iter::from_fn(move || {
            if let Some(push_constant) = push.as_mut().and_then(|iter| iter.next()) {
                return Some(push_constant);
            }
            loop {
                while field_iter.is_none() {
                    field_iter = match set_iter.next()? {
                        BindingEntry::UniformBlock { inputs, .. } => Some(
                            inputs
                                .iter_mut()
                                .filter(|(_, ty)| !matches!(ty, InputType::RawBytes(_))),
                        ),
                        BindingEntry::Texture { input, name, .. } => {
                            if render_pass_targets.contains(name) {
                                None
                            } else {
                                return Some((name.as_str(), input.into()));
                            }
                        }
                        _ => None,
                    };
                }

                if let Some(field_iter_next) = field_iter.as_mut() {
                    let next = field_iter_next
                        .next()
                        .map(|(name, binding)| (name.as_str(), binding.into()));
                    if let Some(next) = next {
                        return Some(next);
                    } else {
                        field_iter = None;
                    }
                }
            }
        })
    }

    pub fn iter_custom_uniforms(&self) -> impl Iterator<Item = (&str, &InputType)> {
        let mut push =
            if let Some(PushConstant::Struct { inputs, .. }) = self.push_constants.as_ref() {
                Some(inputs.iter().map(|(k, v)| (k.as_str(), v)))
            } else {
                None
            };

        let mut field_iter = None;
        let mut iter = self.sets.iter().flat_map(|b| b.binding_entries.iter());
        let set_ref = &self.render_pass_targets;

        std::iter::from_fn(move || {
            if let Some(push_constant) = push.as_mut().and_then(|iter| iter.next()) {
                return Some(push_constant);
            }
            loop {
                while field_iter.is_none() {
                    field_iter = match iter.next()? {
                        BindingEntry::UniformBlock { inputs, .. } => Some(
                            inputs
                                .iter()
                                .filter(|(_, ty)| !matches!(ty, InputType::RawBytes(_))),
                        ),
                        BindingEntry::Texture { input, name, .. } => {
                            if set_ref.contains(name) {
                                None
                            } else {
                                return Some((name.as_str(), input));
                            }
                        }
                        _ => None,
                    };
                }

                if let Some(field_iter_next) = field_iter.as_mut() {
                    let next = field_iter_next
                        .next()
                        .map(|(name, binding)| (name.as_str(), binding));
                    if let Some(next) = next {
                        return Some(next);
                    } else {
                        field_iter = None;
                    }
                }
            }
        })
    }

    pub fn iter_sets(&self) -> impl Iterator<Item = (u32, &wgpu::BindGroup)> {
        self.sets.iter().map(|set| (set.set, &set.bind_group))
    }

    pub fn iter_layouts(&self) -> impl Iterator<Item = &wgpu::BindGroupLayout> {
        self.sets.iter().map(|set| &set.layout)
    }
}

pub fn sets(module: &naga::Module) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();

    for (_, var) in module.global_variables.iter() {
        if let Some(bind) = var.binding.as_ref() {
            out.insert(bind.group);
        }
    }

    out
}

// CPU representation of the shadertoy-like bind group
// This is uploaded to the gpu using std430 memory layout
// keep that in mind when editing this structure
// It also must be 64 byte aligned
#[repr(C)]
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Pod, Zeroable, Default)]
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

impl GlobalData {
    pub fn matches_layout_naga(it: impl Iterator<Item = (naga::TypeInner, usize)>) -> bool {
        let f32_ty = naga::TypeInner::Scalar(naga::Scalar {
            width: 4,
            kind: naga::ScalarKind::Float,
        });
        let u32_ty = naga::TypeInner::Scalar(naga::Scalar {
            width: 4,
            kind: naga::ScalarKind::Uint,
        });

        let vec4_ty = naga::TypeInner::Vector {
            scalar: naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            },
            size: naga::VectorSize::Quad,
        };
        let vec3 = naga::TypeInner::Vector {
            scalar: naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            },
            size: naga::VectorSize::Tri,
        };
        let self_iter = [
            (f32_ty.clone(), offset_of!(GlobalData, time)),
            (f32_ty.clone(), offset_of!(GlobalData, time_delta)),
            (f32_ty.clone(), offset_of!(GlobalData, frame_rate)),
            (u32_ty.clone(), offset_of!(GlobalData, frame)),
            (vec4_ty.clone(), offset_of!(GlobalData, mouse)),
            (vec4_ty.clone(), offset_of!(GlobalData, date)),
            (vec3, offset_of!(GlobalData, resolution)),
            (u32_ty.clone(), offset_of!(GlobalData, pass_index)),
        ];
        self_iter.iter().zip(it).all(|(a, b)| *a == b)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Storage {
    Uniform,
    Push,
    Storage(wgpu::StorageTextureAccess),
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
        inputs: Vec<(String, InputType)>,
        // buffer this uniform is mapped to
        buffer: wgpu::Buffer,
        // the largest struct size in the inputs
        align: usize,
        // storage location
        storage: Storage,
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
        // storage location
        storage: Storage,
    },
    Sampler {
        // the binding index , might not be contiguous
        binding: u32,
        // texture resource if not default
        samp: Option<wgpu::Sampler>,
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
    fn new(
        device: &wgpu::Device,
        module: &naga::Module,
        document: &crate::parsing::DocumentDescriptor,
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

            if crate::uniforms::GlobalData::matches_layout_naga(layout) {
                let mut backing: crate::uniforms::GlobalData = bytemuck::Zeroable::zeroed();
                backing.mouse = [0.0, 0.0, -0.0, -0.0];

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
                    backing,
                    buffer,
                    storage,
                });
            } else {
                Err(Error::UtilityBlockType)?
            }
        }

        let mut inputs = vec![];

        for member in members {
            let name = member.name.clone().unwrap_or_default();
            let ty = module
                .types
                .get_handle(member.ty)
                .map_err(|_| Error::Handle)?;

            if let Some(var) = document.inputs.get(&name) {
                if var.type_check_struct_member(&ty.inner) {
                    inputs.push((name, var.clone()));
                } else {
                    Err(Error::TypeCheck(name.clone()))?
                }
            } else {
                let input = InputType::RawBytes(crate::input_type::RawBytes {
                    inner: vec![0; ty.inner.size(module.to_ctx()) as usize],
                });
                inputs.push((name, input));
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
        // inputs with a non `raw_bytes` type if they are mapped in the document
        input_map: BTreeMap<String, usize>,
        inputs: Vec<(String, InputType)>,
        align: usize,
    },
}

pub fn push_constant(
    module: &naga::Module,
    document: &crate::parsing::DocumentDescriptor,
) -> Result<Option<PushConstant>, Error> {
    let mut out = None;

    for (_, push_constant) in module.global_variables.iter() {
        if push_constant.space != AddressSpace::PushConstant {
            continue;
        }

        if out.is_some() {
            Err(Error::MultiplePushConstants)?
        }

        let push_type = module
            .types
            .get_handle(push_constant.ty)
            .map_err(|_| Error::Handle)?;

        let naga::TypeInner::Struct { members, span } = &push_type.inner else {
            Err(Error::MultiplePushConstants)?
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

            if crate::uniforms::GlobalData::matches_layout_naga(layout) {
                let mut backing: crate::uniforms::GlobalData = bytemuck::Zeroable::zeroed();
                backing.mouse = [0.0, 0.0, -0.0, -0.0];

                out = Some(PushConstant::UtilityBlock { backing });
            } else {
                Err(Error::UtilityBlockType)?
            }
        } else {
            let mut input_map = BTreeMap::new();
            let mut inputs = Vec::new();

            for member in members {
                let name = member.name.clone().unwrap_or_default();
                let ty = module
                    .types
                    .get_handle(member.ty)
                    .map_err(|_| Error::Handle)?;

                if let Some(var) = document.inputs.get(&name) {
                    if var.type_check_struct_member(&ty.inner) {
                        input_map.insert(name.clone(), inputs.len());
                        inputs.push((name, var.clone()));
                    } else {
                        Err(Error::TypeCheck(name.clone()))?
                    }
                } else {
                    let input = InputType::RawBytes(crate::input_type::RawBytes {
                        inner: vec![0; ty.inner.size(module.to_ctx()) as usize],
                    });
                    inputs.push((name.clone(), input.clone()));
                    input_map.insert(name.clone(), inputs.len());
                }
            }

            let align = wgpu::PUSH_CONSTANT_ALIGNMENT as usize;

            out = Some(PushConstant::Struct {
                backing: vec![0; *span as usize],
                input_map,
                inputs,
                align,
            })
        }
    }

    Ok(out)
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
    pub fn new_from_naga(
        set: u32,
        module: &naga::Module,
        document: &crate::parsing::DocumentDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: &wgpu::TextureFormat,
    ) -> Result<TweakBindGroup, Error> {
        let mut layout_entries = vec![];
        let mut binding_entries = vec![];

        for (_, uniform) in module.global_variables.iter() {
            let Some(ResourceBinding { group, binding }) = uniform.binding else {
                continue;
            };

            if group != set {
                continue;
            }

            let ty = module
                .types
                .get_handle(uniform.ty)
                .map_err(|_| Error::Handle)?;

            let storage = match uniform.space {
                naga::AddressSpace::Uniform | naga::AddressSpace::Handle => Storage::Uniform,
                naga::AddressSpace::PushConstant => Storage::Push,
                naga::AddressSpace::Storage { access } => Storage::Storage(storage_access(&access)),
                _ => continue,
            };

            match &ty.inner {
                naga::TypeInner::Array {
                    base: _,
                    size: _,
                    stride: _,
                } => {
                    // How would we even support these?
                    return Err(Error::UnsupportedUniformType("Error loading uniform with an array type. We do not support uniform arrays at this time.".into()));
                }
                naga::TypeInner::Struct { members, span } => {
                    let entry = BindingEntry::new(
                        device,
                        module,
                        document,
                        StructDescriptor {
                            padded_size: *span as usize,
                            name: ty.name.clone().unwrap_or_default(),
                            storage,
                            binding,
                            members: members.as_slice(),
                        },
                    )?;

                    binding_entries.push(entry);

                    let entry = wgpu::BindGroupLayoutEntry {
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        binding,
                        count: None,
                    };

                    layout_entries.push(entry);
                }

                naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    let input = match document.inputs.get(uniform.name.as_ref().unwrap()) {
                        Some(v) if v.is_stored_as_texture() => v.clone(),
                        Some(v) => Err(Error::InputTypeErr(format!("{v}"), "image"))?,
                        None => InputType::Image(crate::input_type::TextureStatus::Uninit),
                    };

                    let entry = image_entry_from_naga(class, dim, *arrayed, binding, format);

                    layout_entries.push(entry);

                    binding_entries.push(BindingEntry::Texture {
                        binding,
                        tex: None,
                        view: None,
                        name: uniform.name.clone().unwrap_or_default(),
                        input: input.clone(),
                        storage,
                    });
                }
                naga::TypeInner::Sampler { .. } => {
                    binding_entries.push(BindingEntry::Sampler {
                        binding,
                        samp: None,
                        name: uniform.name.clone().unwrap_or_default(),
                    });

                    let sampler_type = if matches!(format, wgpu::TextureFormat::Rgba32Float) {
                        wgpu::SamplerBindingType::NonFiltering
                    } else {
                        wgpu::SamplerBindingType::Filtering
                    };

                    let entry = wgpu::BindGroupLayoutEntry {
                        ty: wgpu::BindingType::Sampler(sampler_type),
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        binding,
                        count: None,
                    };

                    layout_entries.push(entry);
                }
                e => panic!("This type should not be in a glsl 440+ uniform: {e:?}"),
            }
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Set {}", set)),
            entries: &layout_entries,
        });

        let place_holder_texture = device.create_texture_with_data(
            queue,
            &txtr_desc(1, 1),
            Default::default(),
            &[0, 0, 0, 0],
        );

        let bind_group = Self::update_uniforms_helper(
            &mut binding_entries,
            &place_holder_texture,
            &layout,
            device,
            queue,
        );

        Ok(Self {
            set,
            binding_entries,
            needs_rebind: false,
            bind_group,
            layout,
        })
    }

    fn get_mut(&mut self, addr: &VariableAddress) -> Option<MutInput> {
        match self.binding_entries.get_mut(addr.binding)? {
            BindingEntry::Texture { input, .. } => Some(input.into()),
            BindingEntry::UniformBlock { inputs, .. } => inputs
                .get_mut(addr.field.unwrap())
                .map(|(_, i)| MutInput::new(i)),
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
                *tex = None;
                *view = None;
                let status = match input {
                    InputType::Image(status)
                    | InputType::Audio(status, _)
                    | InputType::AudioFft(status, _) => status,
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
                    InputType::Image(status)
                    | InputType::Audio(status, _)
                    | InputType::AudioFft(status, _) => status,
                    _ => {
                        return false;
                    }
                };
                *view = Some(new_view);
                *status = TextureStatus::Loaded { width, height };
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
                    InputType::Image(status)
                    | InputType::Audio(status, _)
                    | InputType::AudioFft(status, _) => status,
                    _ => {
                        return false;
                    }
                };
                *status = TextureStatus::Loaded { width, height };
                true
            }
            _ => false,
        }
    }

    fn get(&self, addr: &VariableAddress) -> Option<&InputType> {
        match self.binding_entries.get(addr.binding)? {
            BindingEntry::Texture { input, .. } => Some(input),
            BindingEntry::UniformBlock { inputs, .. } => inputs.get(addr.field?).map(|(_, i)| i),
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
                    for input in inputs.iter() {
                        let bytes = input.1.as_bytes();

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
                    if let Some(samp) = samp {
                        out.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::Sampler(&*samp),
                        });
                    } else {
                        let mut samp_desc = DEFAULT_SAMPLER;

                        if matches!(place_holder_tex.format(), wgpu::TextureFormat::Rgba32Float) {
                            samp_desc.mag_filter = wgpu::FilterMode::Nearest;
                            samp_desc.min_filter = wgpu::FilterMode::Nearest;
                            samp_desc.mipmap_filter = wgpu::FilterMode::Nearest;
                        };

                        let default_sampler = device.create_sampler(&samp_desc);

                        *samp = Some(default_sampler);
                        out.push(wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::Sampler(samp.as_ref().unwrap()),
                        });
                    }
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

fn image_entry_from_naga(
    class: &naga::ImageClass,
    dim: &naga::ImageDimension,
    arrayed: bool,
    binding: u32,
    format: &wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    // no support for texture arrays just yet
    let count = if arrayed { NonZeroU32::new(1) } else { None };
    match class {
        naga::ImageClass::Sampled { kind, multi } => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: sample_kind(kind, format),
                view_dimension: image_dim(dim),
                multisampled: *multi,
            },
            visibility: ShaderStages::VERTEX_FRAGMENT,
            binding,
            count,
        },
        naga::ImageClass::Depth { multi } => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: image_dim(dim),
                multisampled: *multi,
            },
            visibility: ShaderStages::VERTEX_FRAGMENT,
            binding,
            count,
        },
        naga::ImageClass::Storage { format, access } => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::StorageTexture {
                view_dimension: image_dim(dim),
                format: texture_fmt(format),
                access: storage_access(access),
            },
            binding,
            count,
            visibility: ShaderStages::VERTEX_FRAGMENT,
        },
    }
}

fn texture_fmt(fmt: &naga::StorageFormat) -> wgpu::TextureFormat {
    match fmt {
        naga::StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        naga::StorageFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        naga::StorageFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        naga::StorageFormat::R8Sint => wgpu::TextureFormat::R8Sint,
        naga::StorageFormat::R16Uint => wgpu::TextureFormat::R16Uint,
        naga::StorageFormat::R16Sint => wgpu::TextureFormat::R16Sint,
        naga::StorageFormat::R16Float => wgpu::TextureFormat::R16Float,
        naga::StorageFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        naga::StorageFormat::Rg8Snorm => wgpu::TextureFormat::Rg8Snorm,
        naga::StorageFormat::Rg8Uint => wgpu::TextureFormat::Rg8Uint,
        naga::StorageFormat::Rg8Sint => wgpu::TextureFormat::Rg8Sint,
        naga::StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        naga::StorageFormat::R32Sint => wgpu::TextureFormat::R32Sint,
        naga::StorageFormat::R32Float => wgpu::TextureFormat::R32Float,
        naga::StorageFormat::Rg16Uint => wgpu::TextureFormat::Rg16Uint,
        naga::StorageFormat::Rg16Sint => wgpu::TextureFormat::Rg16Sint,
        naga::StorageFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        naga::StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        naga::StorageFormat::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
        naga::StorageFormat::Rgba8Uint => wgpu::TextureFormat::Rgba8Uint,
        naga::StorageFormat::Rgba8Sint => wgpu::TextureFormat::Rgba8Sint,
        naga::StorageFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        naga::StorageFormat::Rg11b10Float => wgpu::TextureFormat::Rg11b10Float,
        naga::StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
        naga::StorageFormat::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
        naga::StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        naga::StorageFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
        naga::StorageFormat::Rgba16Sint => wgpu::TextureFormat::Rgba16Sint,
        naga::StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        naga::StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
        naga::StorageFormat::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,
        naga::StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        naga::StorageFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
        naga::StorageFormat::R16Snorm => wgpu::TextureFormat::R16Snorm,
        naga::StorageFormat::Rg16Unorm => wgpu::TextureFormat::Rg16Unorm,
        naga::StorageFormat::Rg16Snorm => wgpu::TextureFormat::Rg16Snorm,
        naga::StorageFormat::Rgba16Unorm => wgpu::TextureFormat::Rgba16Unorm,
        naga::StorageFormat::Rgba16Snorm => wgpu::TextureFormat::Rgba16Snorm,
        naga::StorageFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        naga::StorageFormat::Rgb10a2Uint => wgpu::TextureFormat::Rgb10a2Uint,
    }
}

fn storage_access(access: &naga::StorageAccess) -> wgpu::StorageTextureAccess {
    let r = access.contains(StorageAccess::LOAD);
    let w = access.contains(StorageAccess::STORE);
    match (r, w) {
        (true, true) => wgpu::StorageTextureAccess::ReadWrite,
        (false, true) => wgpu::StorageTextureAccess::WriteOnly,
        (false | true, false) => wgpu::StorageTextureAccess::ReadOnly,
    }
}

fn sample_kind(scalar: &naga::ScalarKind, format: &TextureFormat) -> wgpu::TextureSampleType {
    match scalar {
        naga::ScalarKind::Sint => wgpu::TextureSampleType::Sint,
        naga::ScalarKind::Uint => wgpu::TextureSampleType::Uint,
        naga::ScalarKind::Float => wgpu::TextureSampleType::Float {
            filterable: !matches!(format, wgpu::TextureFormat::Rgba32Float),
        },
        naga::ScalarKind::Bool => wgpu::TextureSampleType::Uint,
        naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat => unreachable!(),
    }
}

fn image_dim(dim: &naga::ImageDimension) -> wgpu::TextureViewDimension {
    match dim {
        naga::ImageDimension::D1 => wgpu::TextureViewDimension::D1,
        naga::ImageDimension::D2 => wgpu::TextureViewDimension::D2,
        naga::ImageDimension::D3 => wgpu::TextureViewDimension::D3,
        naga::ImageDimension::Cube => wgpu::TextureViewDimension::Cube,
    }
}

fn txtr_desc(width: u32, height: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1, // crunch crunch
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}
