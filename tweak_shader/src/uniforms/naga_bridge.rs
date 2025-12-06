use super::{
    BindingEntry, Error, ResourceBinding, StorageTextureState, StructDescriptor, TweakBindGroup,
    DEFAULT_SAMPLER, DEFAULT_VIEW,
};
use crate::input_type::InputType;
use wgpu::{
    self,
    naga::{self, StorageAccess},
    util::DeviceExt,
    ShaderStages, TextureFormat,
};

use core::num::NonZeroU32;

impl TweakBindGroup {
    pub fn new_from_naga(
        set: u32,
        module: &naga::Module,
        document: &crate::parsing::Document,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: &wgpu::TextureFormat,
    ) -> Result<Self, Error> {
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
                    let entry = BindingEntry::new_struct_entry(
                        device,
                        module,
                        document,
                        StructDescriptor {
                            padded_size: *span as usize,
                            name: ty.name.clone().unwrap_or_default(),
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
                        visibility: if document.stage == naga::ShaderStage::Compute {
                            ShaderStages::COMPUTE
                        } else {
                            ShaderStages::FRAGMENT
                        },
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
                    let entry = image_entry_from_naga(class, dim, *arrayed, binding, format)?;

                    if let wgpu::BindingType::StorageTexture { format, .. } = entry.ty {
                        let pixel_size = format.block_copy_size(None).unwrap();

                        let place_holder_tex = device.create_texture_with_data(
                            queue,
                            &crate::uniforms::storage_desc(1, 1, format),
                            Default::default(),
                            &vec![0; pixel_size as usize],
                        );

                        let mut view_desc = DEFAULT_VIEW;
                        view_desc.format = Some(format);

                        let placeholder_view = place_holder_tex.create_view(&view_desc);

                        let maybe_target = document
                            .targets
                            .iter()
                            .find(|t| Some(&t.name) == uniform.name.as_ref());

                        let maybe_relay = document
                            .relays
                            .iter()
                            .find(|r| Some(&r.name) == uniform.name.as_ref());

                        let state = match (maybe_target, maybe_relay) {
                            (Some(targ), None) => StorageTextureState::Target {
                                user_provided_view: None,
                                previous_view: None,
                                persistent: targ.persistent,
                                width: targ.width,
                                height: targ.height,
                            },
                            (None, Some(relay)) => StorageTextureState::Relay {
                                persistent: relay.persistent,
                                target: relay.target.clone(),
                                width: relay.width,
                                height: relay.height,
                            },
                            (None, None) => {
                                return Err(Error::TargetValidation(format!(
                                    "Storage texture not found {}",
                                    uniform.name.clone().unwrap_or_default()
                                )))
                            }
                            (Some(_), Some(_)) => {
                                return Err(Error::TargetValidation(format!(
                                    "Storage texture referenced in both a target and a relay {}",
                                    uniform.name.clone().unwrap_or_default()
                                )))
                            }
                        };

                        binding_entries.push(BindingEntry::StorageTexture {
                            binding,
                            state,
                            tex: place_holder_tex,
                            view: placeholder_view,
                            name: uniform.name.clone().unwrap_or_default(),
                        });
                    } else {
                        let input = match document.inputs.get(uniform.name.as_ref().unwrap()) {
                            Some(v) if v.is_stored_as_texture() => v.clone(),
                            Some(v) => {
                                Err(Error::InputTypeErr(format!("{v}"), "image".to_owned()))?
                            }
                            None => InputType::Image(crate::input_type::TextureStatus::Uninit),
                        };

                        binding_entries.push(BindingEntry::Texture {
                            binding,
                            tex: None,
                            view: None,
                            name: uniform.name.clone().unwrap_or_default(),
                            input: input.clone(),
                        });
                    };

                    layout_entries.push(entry);
                }
                naga::TypeInner::Sampler { .. } => {
                    let mut samp_desc = DEFAULT_SAMPLER;

                    if let Some(name) = uniform.name.as_ref() {
                        if let Some(config) = document.samplers.iter().find(|e| &e.name == name) {
                            samp_desc.mag_filter = config.filter_mode;
                            samp_desc.min_filter = config.filter_mode;
                            samp_desc.mipmap_filter = config.filter_mode;

                            samp_desc.address_mode_u = config.clamp_mode;
                            samp_desc.address_mode_v = config.clamp_mode;
                            samp_desc.address_mode_w = config.clamp_mode;
                        }
                    };

                    if matches!(format, wgpu::TextureFormat::Rgba32Float) {
                        samp_desc.mag_filter = wgpu::FilterMode::Nearest;
                        samp_desc.min_filter = wgpu::FilterMode::Nearest;
                        samp_desc.mipmap_filter = wgpu::FilterMode::Nearest;
                    };

                    let samp = device.create_sampler(&samp_desc);

                    binding_entries.push(BindingEntry::Sampler {
                        binding,
                        samp,
                        name: uniform.name.clone().unwrap_or_default(),
                    });

                    let sampler_type = if matches!(format, wgpu::TextureFormat::Rgba32Float) {
                        wgpu::SamplerBindingType::NonFiltering
                    } else {
                        wgpu::SamplerBindingType::Filtering
                    };

                    let entry = wgpu::BindGroupLayoutEntry {
                        ty: wgpu::BindingType::Sampler(sampler_type),
                        visibility: if document.stage == naga::ShaderStage::Compute {
                            ShaderStages::COMPUTE
                        } else {
                            ShaderStages::FRAGMENT
                        },
                        binding,
                        count: None,
                    };

                    layout_entries.push(entry);
                }
                t => return Err(Error::UnsupportedUniformType(format!("{t:?}"))),
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
}

pub fn texture_fmt(fmt: &naga::StorageFormat) -> wgpu::TextureFormat {
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
        naga::StorageFormat::Rg11b10Ufloat => wgpu::TextureFormat::Rg11b10Ufloat,
        naga::StorageFormat::R64Uint => wgpu::TextureFormat::R64Uint,
    }
}

pub fn storage_access(access: &naga::StorageAccess) -> wgpu::StorageTextureAccess {
    let r = access.contains(StorageAccess::LOAD);
    let w = access.contains(StorageAccess::STORE);
    match (r, w) {
        (true, true) => wgpu::StorageTextureAccess::ReadWrite,
        (false, true) => wgpu::StorageTextureAccess::WriteOnly,
        (false | true, false) => wgpu::StorageTextureAccess::ReadOnly,
    }
}

pub fn sample_kind(scalar: &naga::ScalarKind, format: &TextureFormat) -> wgpu::TextureSampleType {
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

pub fn image_dim(dim: &naga::ImageDimension) -> Result<wgpu::TextureViewDimension, super::Error> {
    match dim {
        naga::ImageDimension::D2 => Ok(wgpu::TextureViewDimension::D2),
        _ => Err(super::Error::TextureDimension),
    }
}

// default texture type for immutable textures
pub fn txtr_desc(width: u32, height: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

pub fn work_groups_from_naga(module: &naga::Module) -> [u32; 3] {
    module
        .entry_points
        .first()
        .map(|e| e.workgroup_size)
        .expect("no entry point.")
}

pub fn image_entry_from_naga(
    class: &naga::ImageClass,
    dim: &naga::ImageDimension,
    arrayed: bool,
    binding: u32,
    format: &wgpu::TextureFormat,
) -> Result<wgpu::BindGroupLayoutEntry, super::Error> {
    // no support for texture arrays just yet
    let count = if arrayed { NonZeroU32::new(1) } else { None };
    let out = match class {
        naga::ImageClass::Sampled { kind, multi } => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: sample_kind(kind, format),
                view_dimension: image_dim(dim)?,
                multisampled: *multi,
            },
            visibility: ShaderStages::all(),
            binding,
            count,
        },
        naga::ImageClass::Depth { multi } => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: image_dim(dim)?,
                multisampled: *multi,
            },
            visibility: ShaderStages::all(),
            binding,
            count,
        },
        naga::ImageClass::Storage { format, access } => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::StorageTexture {
                view_dimension: image_dim(dim)?,
                format: texture_fmt(format),
                access: storage_access(access),
            },
            binding,
            count,
            visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
        },
        naga::ImageClass::External => wgpu::BindGroupLayoutEntry {
            ty: wgpu::BindingType::Texture {
                sample_type: sample_kind(&naga::ScalarKind::Float, format),
                view_dimension: image_dim(dim)?,
                multisampled: false,
            },
            visibility: ShaderStages::all(),
            binding,
            count,
        },
    };
    Ok(out)
}
