mod error;
mod naga_bridge;
mod validation;

use crate::context;
use crate::input_type::*;
use crate::parsing::Document;

pub use error::Error;
use naga::{AddressSpace as NagaAddressSpace, ResourceBinding};
use naga::{FastHashMap, FastIndexMap};
pub use naga_bridge::*;
use wgpu::naga;

use bytemuck::*;
use wgpu::{util::DeviceExt, BufferUsages};

use crate::VarName;
use wgpu::BindGroupLayout;

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

pub struct TargetDescriptor<'a> {
    pub persistent: bool,
    pub name: &'a str,
    pub format: wgpu::TextureFormat,
}

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

#[derive(Debug, Copy, Clone)]
struct VariableAddress {
    // actually just index into set list.
    pub set: usize,
    // actually just index into bind group list
    pub binding: usize,
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct PrivateTexture {
    name: VarName,
    pass_target: bool,
}

#[derive(Debug)]
pub struct Uniforms {
    /// mapping of names to (set, binding) pairs for faster lookup in the
    /// sets vector
    lookup_table: FastHashMap<VarName, VariableAddress>,
    /// Textures that are either in render targets or not bound
    /// to any inputs
    private_textures: Vec<PrivateTexture>,
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
    /// If this is a compute shader uniform, then use either this or the first
    /// applicable texture as the output for "render_to_vec / slice" calls
    default_compute_texture: Option<String>,
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
                    BindingEntry::Buffer { inputs, .. } => {
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

        let mut private_textures: Vec<_> = document
            .passes
            .iter()
            .filter_map(|pass| {
                PrivateTexture {
                    name: pass.target_texture.clone()?,
                    pass_target: true,
                }
                .into()
            })
            .collect();

        for tex in sets
            .iter()
            .flat_map(|set| &set.binding_entries)
            .filter_map(|entry| extract!(entry, BindingEntry::Texture { ref name, ..} => name))
        {
            let texture_in_doc = document.inputs.contains_key(tex);
            if !texture_in_doc {
                private_textures.push(PrivateTexture {
                    name: tex.clone(),
                    pass_target: false,
                });
            }
        }

        // During the render pass we need to queue `copy_buffer_to_buffer`
        // calls in order to update the pass index in a predictable way.
        // This means we need a sequential list of indices from 0 to max_pass
        let contents = (0..pass_count as u32).collect::<Vec<u32>>();
        let pass_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&contents),
            label: None,
            usage: BufferUsages::COPY_SRC,
        });

        let first_target = sets
            .iter()
            .flat_map(|s| s.binding_entries.iter())
            .find_map(|t| {
                if let BindingEntry::StorageTexture {
                    purpose: StorageTexturePurpose::Target { .. },
                    name,
                    ..
                } = t
                {
                    name.clone().into()
                } else {
                    None
                }
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
            default_compute_texture: first_target,
        };

        out.validate(document, format)?;

        Ok(out)
    }

    pub fn set_compute_target(&mut self, name: &str) -> Result<(), Error> {
        let addr = self
            .lookup_table
            .get(name)
            .ok_or(Error::NonexistantTarget(name.to_owned()))?;

        let set = self
            .sets
            .get(addr.set)
            .ok_or(Error::NonexistantTarget(name.to_owned()))?;

        if let Some(BindingEntry::StorageTexture { purpose: state, .. }) =
            set.binding_entries.get(addr.binding)
        {
            if matches!(state, StorageTexturePurpose::Target { .. }) {
                self.default_compute_texture = Some(name.to_owned());
                Ok(())
            } else {
                Err(Error::NotScreenTarget(name.to_owned()))
            }
        } else {
            Err(Error::NotStorageTexture(name.to_owned()))
        }
    }

    // Copy this uniforms data into other - goes by name
    pub fn copy_into(&mut self, other: &mut Self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Copy Uniforms
        other.utility_block_data = self.utility_block_data;
        let default_target = self.default_compute_texture.as_deref();

        if other.iter_targets().any(|t| Some(t.name) == default_target) {
            other.default_compute_texture = self.default_compute_texture.clone();
        }

        for (name, addr) in other.lookup_table.iter() {
            let Some(other_value) = other
                .sets
                .get_mut(addr.set)
                .and_then(|s| s.get_input_mut(name, addr))
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

        for name in self.lookup_table.keys() {
            let Some(self_image) = self.get_texture(name) else {
                continue;
            };

            if other
                .get_input(name)
                .map(|t| !t.is_stored_as_texture())
                .unwrap_or_default()
            {
                continue;
            };

            let width = self_image.width();
            let height = self_image.height();

            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };

            let new_tex = if other
                .private_textures
                .iter()
                .any(|s| &s.name == name && s.pass_target)
            {
                let mut desc = txtr_desc(width, height);
                desc.format = wgpu::TextureFormat::Rgba16Float;
                device.create_texture(&desc)
            } else {
                device.create_texture(&txtr_desc(width, height))
            };

            command_encoder.copy_texture_to_texture(
                self_image.as_image_copy(),
                new_tex.as_image_copy(),
                size,
            );

            other.set_texture(name, new_tex);
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

    pub fn adjust_storage_texture_sizes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_width: u32,
        screen_height: u32,
    ) {
        let groups = self
            .sets
            .iter_mut()
            .flat_map(|e| e.binding_entries.iter_mut());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut pass_jobs = vec![];
        for target in groups {
            if let BindingEntry::StorageTexture {
                tex,
                view,
                purpose:
                    StorageTexturePurpose::Relay {
                        target: tname,
                        width,
                        height,
                        ..
                    },
                ..
            } = target
            {
                let new_tex = resize_texture(
                    device,
                    &mut encoder,
                    tex,
                    width.unwrap_or(screen_width),
                    height.unwrap_or(screen_height),
                );

                if let Some(new_tex) = new_tex {
                    *view = new_tex.create_view(&Default::default());
                    *tex = new_tex;
                }

                pass_jobs.push((tex.format(), tname.clone(), *width, *height));
            }

            if let BindingEntry::StorageTexture {
                name,
                tex,
                view,
                purpose:
                    StorageTexturePurpose::Target {
                        width,
                        height,
                        user_provided_view: None,
                        ..
                    },
                ..
            } = target
            {
                if Some(&*name) == self.default_compute_texture.as_ref() {
                    continue;
                }

                let new_tex = resize_texture(
                    device,
                    &mut encoder,
                    tex,
                    width.unwrap_or(screen_width),
                    height.unwrap_or(screen_height),
                );

                if let Some(new_tex) = new_tex {
                    *view = new_tex.create_view(&Default::default());
                    *tex = new_tex;
                }
            }
        }

        //  For each relay resize the relay target texture
        for (fmt, storage_name, height, width) in pass_jobs {
            let width = width.unwrap_or(screen_width);
            let height = height.unwrap_or(screen_height);

            let mut groups = self
                .sets
                .iter_mut()
                .flat_map(|e| e.binding_entries.iter_mut());

            let Some((tex, view)) = groups.find_map(|entry| match entry {
                BindingEntry::Texture {
                    tex,
                    user_provided_override: view,
                    name,
                    ..
                } if name.as_str() == storage_name.as_str() => Some((tex, view)),
                _ => None,
            }) else {
                continue;
            };

            if tex.is_none() {
                let mut desc = txtr_desc(width, height);
                desc.format = fmt;
                *tex = Some(device.create_texture(&desc));
                *view = tex.as_ref().map(|t| t.create_view(&Default::default()));
            }

            let tex = tex.as_mut().unwrap();
            let view = view.as_mut().unwrap();

            let new_tex = resize_texture(device, &mut encoder, tex, width, height);

            if let Some(new_tex) = new_tex {
                *view = new_tex.create_view(&Default::default());
                *tex = new_tex;
            }
        }

        queue.submit(Some(encoder.finish()));
    }

    // replace the views in the storage textures with those
    // proved by the target set.
    pub fn map_target_view(&mut self, substitute_view: wgpu::TextureView) {
        let mut groups = self.sets.iter_mut().flat_map(|e| {
            //TODO: this is hacky! only rebind if needed
            // and certainly not as a side effect of a map
            e.needs_rebind = true;
            e.binding_entries.iter_mut()
        });

        let user_view = groups.find_map(|group| {
            if let BindingEntry::StorageTexture {
                purpose:
                    StorageTexturePurpose::Target {
                        user_provided_view,
                        previous_view,
                        ..
                    },
                name,
                ..
            } = group
            {
                if Some(&*name) == self.default_compute_texture.as_ref() {
                    Some((user_provided_view, previous_view))
                } else {
                    None
                }
            } else {
                None
            }
        });

        if let Some((view, previous)) = user_view {
            *previous = view.take();
            *view = Some(substitute_view);
        }
    }

    pub fn forward_relays(&self, command_encoder: &mut wgpu::CommandEncoder) {
        let groups = self.sets.iter().flat_map(|e| e.binding_entries.iter());

        let relay_textures = groups.filter_map(|group| {
            extract!(group, BindingEntry::StorageTexture { tex, purpose: StorageTexturePurpose::Relay { target,.. }, .. } => (target, tex))
        });

        for (target, tex) in relay_textures {
            let Some(target_tex) = self.get_texture(target) else {
                continue;
            };

            command_encoder.copy_texture_to_texture(
                tex.as_image_copy(),
                target_tex.as_image_copy(),
                wgpu::Extent3d {
                    width: target_tex.width().min(tex.width()),
                    height: target_tex.height().min(tex.height()),
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub fn reset_targets_to_context_managed(&mut self) {
        let groups = self
            .sets
            .iter_mut()
            .flat_map(|e| e.binding_entries.iter_mut());

        groups.for_each(|group| {
            if let BindingEntry::StorageTexture {
                purpose:
                    StorageTexturePurpose::Target {
                        user_provided_view,
                        previous_view,
                        ..
                    },
                ..
            } = group
            {
                *user_provided_view = previous_view.take()
            }
        });
    }

    // zero the textures of any targets
    // that are not labeled as persistent
    pub fn clear_ephemeral_targets_and_buffers(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        let groups = self.sets.iter().flat_map(|e| e.binding_entries.iter());

        let views = groups.filter_map(|group| match group {
            BindingEntry::StorageTexture {
                view,
                purpose:
                    StorageTexturePurpose::Target {
                        persistent: false,
                        user_provided_view,
                        ..
                    },
                ..
            } => Some(user_provided_view.as_ref().unwrap_or(view)),
            BindingEntry::StorageTexture {
                view,
                purpose:
                    StorageTexturePurpose::Relay {
                        persistent: false, ..
                    },
                ..
            } => Some(view),
            _ => None,
        });

        for ephemeral_view in views {
            command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Texture Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ephemeral_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
    }

    pub fn iter_targets(&self) -> impl Iterator<Item = TargetDescriptor<'_>> {
        let groups = self.sets.iter().flat_map(|e| e.binding_entries.iter());
        groups.filter_map(|group| {
            if let BindingEntry::StorageTexture {
                name,
                tex,
                purpose: StorageTexturePurpose::Target { persistent, .. },
                ..
            } = group
            {
                TargetDescriptor {
                    persistent: *persistent,
                    name,
                    format: tex.format(),
                }
                .into()
            } else {
                None
            }
        })
    }

    pub fn get_buffer(&mut self, var_name: &str) -> Option<&wgpu::Buffer> {
        self.sets
            .iter()
            .flat_map(|b| b.binding_entries.iter())
            .find_map(|entries| match entries {
                BindingEntry::Buffer {
                    buffer,
                    id: Some(n),
                    ..
                } if n.as_str() == var_name => Some(buffer),
                _ => None,
            })
    }

    pub fn resize_dynamic_array(
        &mut self,
        var_name: &str,
        num_elements: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Error> {
        let mut_buffer = self
            .sets
            .iter_mut()
            .flat_map(|b| b.binding_entries.iter_mut())
            .find_map(|entries| match entries {
                BindingEntry::Buffer {
                    buffer,
                    align,
                    id: Some(n),
                    ..
                } if n.as_str() == var_name => Some((align, buffer)),
                _ => None,
            });

        if let Some((stride, buffer)) = mut_buffer {
            let new_size = num_elements * *stride;

            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(var_name),
                size: new_size as wgpu::BufferAddress,
                usage: buffer.usage(),
                mapped_at_creation: false,
            });

            let copy_size = buffer.size().min(new_size as wgpu::BufferAddress);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer resize encoder"),
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &new_buffer, 0, copy_size);
            queue.submit(std::iter::once(encoder.finish()));

            *buffer = new_buffer;
            Ok(())
        } else {
            Err(Error::NoBuffer(var_name.to_string()))
        }
    }

    pub fn unload_texture(&mut self, var: &str) -> bool {
        if self.private_textures.iter().any(|t| t.name.as_str() == var) {
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

    pub fn push_constant_ranges(&self, compute: bool) -> Option<wgpu::PushConstantRange> {
        if let Some(push) = self.push_constants.as_ref() {
            let size = match push {
                PushConstant::UtilityBlock { .. } => std::mem::size_of::<GlobalData>(),
                PushConstant::Struct { backing, .. } => backing.len(),
            };
            Some(wgpu::PushConstantRange {
                stages: if compute {
                    wgpu::ShaderStages::COMPUTE
                } else {
                    wgpu::ShaderStages::VERTEX_FRAGMENT
                },
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
        desc: context::TextureDesc,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let context::TextureDesc {
            width,
            height,
            stride,
            data,
            format,
        } = desc;
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
                    && texture.format() == format =>
            {
                texture
            }
            _ => {
                let mut desc = txtr_desc(width, height);
                desc.format = format;
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
                bytes_per_row: stride.or(Some(width * block_size)),
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

    pub fn get_texture_ownership(&self, name: &str) -> Option<TexOwnership> {
        let groups = self.sets.iter().flat_map(|e| e.binding_entries.iter());

        let mut targets = groups.filter_map(
            |group| extract!(group, BindingEntry::StorageTexture { name, tex, .. } => (name, tex)),
        );

        if let Some((_, tex)) = targets.find(|(targ_name, _)| targ_name.as_str() == name) {
            Some(TexOwnership::Tex(tex))
        } else {
            let addr = self.lookup_table.get(name)?;
            let bind_group = self.sets.get(addr.set)?;
            match bind_group.binding_entries.get(addr.binding)? {
                BindingEntry::Texture {
                    tex,
                    user_provided_override: view,
                    ..
                } => match (tex, view) {
                    (Some(t), None) => TexOwnership::Tex(t).into(),
                    (_, Some(v)) => TexOwnership::View(v).into(),
                    (None, None) => None,
                },
                _ => None,
            }
        }
    }

    pub fn get_texture(&self, name: &str) -> Option<&wgpu::Texture> {
        let groups = self.sets.iter().flat_map(|e| e.binding_entries.iter());
        let mut targets = groups.filter_map(
            |group| extract!(group, BindingEntry::StorageTexture { name, tex, .. } => (name, tex)),
        );

        if let Some((_, tex)) = targets.find(|(targ_name, _)| targ_name.as_str() == name) {
            Some(tex)
        } else {
            let addr = self.lookup_table.get(name)?;
            self.sets.get(addr.set)?.get_texture(addr)
        }
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
        let out = self
            .push_constants
            .as_mut()
            .and_then(|p| extract!(p, PushConstant::Struct { inputs, .. } => inputs.get_mut(name) ))
            .flatten()
            .map(|i| i.into());

        if out.is_some() {
            out
        } else {
            let addr = self.lookup_table.get(name)?;
            let set = self.sets.get_mut(addr.set)?;
            set.get_input_mut(name, addr)
        }
    }

    pub fn get_input(&self, name: &str) -> Option<&InputType> {
        let out = self
            .push_constants
            .as_ref()
            .and_then(|p| extract!(p, PushConstant::Struct { inputs, .. } => inputs.get(name) ))
            .flatten();

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
        set.get_input_mut(name, addr)
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
                BindingEntry::Buffer { inputs, .. } => {
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
                    if self.private_textures.iter().any(|t| &t.name == name) {
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
                BindingEntry::Buffer { inputs, .. } => {
                    let iter = inputs.iter().filter(|(k, v)| {
                        !matches!(v, InputType::RawBytes(_)) && self.lookup_table.contains_key(*k)
                    });
                    Some(Box::new(iter) as _)
                }
                BindingEntry::Texture { input, name, .. } => {
                    if self.private_textures.iter().any(|t| &t.name == name) {
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
pub enum TSAddressSpace {
    Uniform,
    Push,
    Storage(wgpu::StorageTextureAccess),
}

#[derive(Debug)]
pub enum StorageTexturePurpose {
    Relay {
        // the target texture to forward this into after each pass
        target: String,
        persistent: bool,
        width: Option<u32>,
        height: Option<u32>,
    },
    Target {
        user_provided_view: Option<wgpu::TextureView>,
        previous_view: Option<wgpu::TextureView>,
        persistent: bool,
        width: Option<u32>,
        height: Option<u32>,
    },
}

#[derive(Debug)]
pub enum BindingEntry {
    UtilityUniformBlock {
        binding: u32,
        // Global layout data
        backing: crate::uniforms::GlobalData,
        // buffer this uniform is mapped to
        buffer: wgpu::Buffer,
    },
    Buffer {
        // the binding index , might not be contiguous
        binding: u32,
        // inputs with a non `raw_bytes` type if they are mapped in the document
        inputs: FastIndexMap<String, InputType>,
        // buffer this uniform is mapped to
        backing: Vec<u8>,
        buffer: wgpu::Buffer,
        // the largest struct size in the inputs
        align: usize,
        // storage location
        storage: TSAddressSpace,
        // the name of the variable, if any.
        type_name: Option<String>,
        id: Option<String>,
    },
    StorageTexture {
        // the binding index , might not be contiguous
        binding: u32,
        tex: wgpu::Texture,
        view: wgpu::TextureView,
        purpose: StorageTexturePurpose,
        // variable name
        name: String,
    },
    Texture {
        // the binding index , might not be contiguous
        binding: u32,
        // texture resource if not default
        tex: Option<wgpu::Texture>,
        // a user may replace this texture with a shared reference
        // to an outside texture held in this view.
        user_provided_override: Option<wgpu::TextureView>,
        temp_view: Option<wgpu::TextureView>,
        // image status for bookeeping
        input: InputType,
        // variable name
        name: String,
    },
    Sampler {
        // the binding index , might not be contiguous
        binding: u32,
        // texture resource if not default
        samp: wgpu::Sampler,
        name: String,
    },
}

struct PrimitiveDescriptor {
    // name of struct type
    type_name: Option<String>,
    // name of global identifier if not a destructures struct
    id: Option<String>,
    storage: TSAddressSpace,
    binding: u32,
    element_type: naga::Handle<naga::Type>,
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
        .filter(|(_, var)| var.space == NagaAddressSpace::PushConstant)
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

pub enum TexOwnership<'a> {
    View(&'a wgpu::TextureView),
    Tex(&'a wgpu::Texture),
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
    fn get_input_mut(&mut self, name: &str, addr: &VariableAddress) -> Option<MutInput> {
        match self.binding_entries.get_mut(addr.binding)? {
            BindingEntry::Texture { input, .. } => Some(input.into()),
            BindingEntry::Buffer { inputs, .. } => inputs.get_mut(name).map(|i| i.into()),
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
                tex,
                user_provided_override: view,
                input,
                ..
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
            Some(BindingEntry::StorageTexture {
                purpose:
                    StorageTexturePurpose::Target {
                        user_provided_view, ..
                    },
                ..
            }) => {
                *user_provided_view = None;
                false
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
            Some(BindingEntry::Texture {
                user_provided_override: view,
                input,
                ..
            }) => {
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
            Some(BindingEntry::StorageTexture {
                purpose:
                    StorageTexturePurpose::Target {
                        user_provided_view, ..
                    },
                ..
            }) => {
                *user_provided_view = Some(new_view);
                true
            }
            _ => false,
        }
    }

    fn set_texture(&mut self, addr: &VariableAddress, new_tex: wgpu::Texture) -> bool {
        match self.binding_entries.get_mut(addr.binding) {
            Some(BindingEntry::Texture {
                tex,
                user_provided_override: view,
                input,
                ..
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
            BindingEntry::Buffer { inputs, .. } => inputs.get(name),
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
                    BindingEntry::Buffer {
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
        let out: Vec<_> = bindings
            .iter_mut()
            .map(|binding| {
                match binding {
                    BindingEntry::UtilityUniformBlock {
                        binding,
                        backing,
                        buffer,
                        ..
                    } => {
                        queue.write_buffer(buffer, 0, bytemuck::bytes_of(backing));
                        wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: buffer.as_entire_binding(),
                        }
                    }
                    BindingEntry::Buffer {
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
                        wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: buffer.as_entire_binding(),
                        }
                    }
                    BindingEntry::Texture {
                        binding,
                        user_provided_override,
                        tex,
                        temp_view,
                        ..
                    } => {
                        if let Some(view) = user_provided_override.as_ref() {
                            wgpu::BindGroupEntry {
                                binding: *binding,
                                resource: wgpu::BindingResource::TextureView(view),
                            }
                        } else if let Some(tex) = tex {
                            let new_view = tex.create_view(&Default::default());
                            *temp_view = Some(new_view);

                            wgpu::BindGroupEntry {
                                binding: *binding,
                                resource: wgpu::BindingResource::TextureView(
                                    temp_view.as_ref().unwrap(),
                                ),
                            }
                        } else {
                            wgpu::BindGroupEntry {
                                binding: *binding,
                                resource: wgpu::BindingResource::TextureView(&placeholder_view),
                            }
                        }
                    }
                    BindingEntry::Sampler { binding, samp, .. } => wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::Sampler(&*samp),
                    },
                    BindingEntry::StorageTexture {
                        binding,
                        view,
                        purpose: state,
                        ..
                    } => match state {
                        StorageTexturePurpose::Relay { .. } => wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::TextureView(&*view),
                        },
                        StorageTexturePurpose::Target {
                            user_provided_view, ..
                        } => wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: wgpu::BindingResource::TextureView(
                                user_provided_view.as_ref().unwrap_or(&*view),
                            ),
                        },
                    },
                }
            })
            .collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: out.as_slice(),
        });

        for b in bindings.iter_mut() {
            if let BindingEntry::Texture { temp_view, .. } = b {
                *temp_view = None
            }
        }

        bind_group
    }
}

// default texture constructor for storage textures
pub fn storage_desc(
    width: u32,
    height: u32,
    fmt: wgpu::TextureFormat,
) -> wgpu::TextureDescriptor<'static> {
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
        format: fmt,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

fn resize_texture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    old_tex: &wgpu::Texture,
    target_width: u32,
    target_height: u32,
) -> Option<wgpu::Texture> {
    if target_width == old_tex.width() && target_height == old_tex.height() {
        return None;
    }

    let new_desc = storage_desc(target_width, target_height, old_tex.format());
    let new_tex = device.create_texture(&new_desc);

    encoder.copy_texture_to_texture(
        old_tex.as_image_copy(),
        new_tex.as_image_copy(),
        wgpu::Extent3d {
            width: old_tex.width().min(target_width),
            height: old_tex.height().min(target_height),
            depth_or_array_layers: 1,
        },
    );

    Some(new_tex)
}
