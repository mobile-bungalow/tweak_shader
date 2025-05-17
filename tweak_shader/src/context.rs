use crate::input_type::*;
use crate::uniforms;
use naga::front::glsl;
use wgpu::naga;

use naga::{front::glsl::Options, ShaderStage};

use crate::Error;

use wgpu::TextureFormat;

/// The main rendering and bookkeeping context.
#[derive(Debug)]
pub struct RenderContext {
    uniforms: uniforms::Uniforms,
    passes: Vec<RenderPass>,
    pipeline: Pipeline,
    screen_buffer_cache: BufferCache,
    stage: wgpu::naga::ShaderStage,
}

#[derive(Debug)]
enum Pipeline {
    Compute {
        compute_pipeline: wgpu::ComputePipeline,
        workgroups: [u32; 3],
    },
    Pixel {
        pipeline: wgpu::RenderPipeline,
        float_pipeline: wgpu::RenderPipeline,
    },
}

pub struct TextureDesc<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: Option<u32>,
    pub data: &'a [u8],
    pub format: wgpu::TextureFormat,
}

impl RenderContext {
    /// Creates a new [RenderContext] with a pipeline corresponding to the shader file
    /// capable of rendering to texture views with the specified `format`.
    ///
    /// Warning:
    /// may throw a validation error to the `device`, if you are not certain
    /// whether or not you are passing in valid shaders you should handles these
    /// by pushing the proper error scopes.
    pub fn new<Src: AsRef<str>>(
        source: Src,
        format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self, Error> {
        let source = source.as_ref();

        let document = crate::parsing::parse_document(source)?;

        let stripped_src: String = source
            .lines()
            .filter(|line| !line.trim().starts_with("#pragma"))
            .collect::<Vec<_>>()
            .join("\n");

        let options = Options {
            stage: document.stage,
            defines: [("TWEAK_SHADER".to_owned(), "1".to_owned())]
                .into_iter()
                .collect(),
        };

        let naga_mod = glsl::Frontend::default()
            .parse(&options, &stripped_src)
            .map_err(|e| {
                Error::ShaderCompilationFailed(display_errors(&e.errors, &stripped_src))
            })?;

        // internal passes should be HDR, they are often used
        // pass data around.
        let pass_texture = if is_floating_point_in_shader(&format) {
            TextureFormat::Rgba16Float
        } else {
            TextureFormat::Rgba16Uint
        };

        let pass_structure = document
            .passes
            .iter()
            .map(|pass| RenderPass::new(pass, pass_texture))
            .chain(std::iter::once(RenderPass::new(
                &Default::default(),
                format,
            )))
            .collect::<Vec<_>>();

        let sets = uniforms::sets(&naga_mod, &document, device, queue, &format)?;

        // theres is only every 1 or 0 push constant blocks
        let push_const = uniforms::push_constant(&naga_mod, &document)?;

        let uniforms = uniforms::Uniforms::new(
            &document,
            &format,
            device,
            queue,
            sets,
            push_const,
            pass_structure.len(),
        )?;

        let bind_group_layouts: Vec<_> = uniforms.iter_layouts().collect();
        let is_compute = document.stage == ShaderStage::Compute;

        let push_constant_ranges = uniforms
            .push_constant_ranges(is_compute)
            .map(|e| vec![e])
            .unwrap_or(vec![]);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: bind_group_layouts.as_slice(),
            push_constant_ranges: &push_constant_ranges,
        });

        let pipeline = if is_compute {
            let compute_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(naga_mod.clone())),
            });

            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &compute_mod,
                    entry_point: "main".into(),
                    compilation_options: Default::default(),
                    cache: None,
                });

            Pipeline::Compute {
                compute_pipeline,
                workgroups: uniforms::work_groups_from_naga(&naga_mod),
            }
        } else {
            let fs_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(naga_mod.clone())),
            });

            let vs_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../resources/stub.wgsl"
                ))),
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: Some(&pipeline_layout),
                fragment: Some(wgpu::FragmentState {
                    compilation_options: Default::default(),
                    module: &fs_shader_module,
                    entry_point: "main".into(),
                    targets: &[Some(format.into())],
                }),
                vertex: wgpu::VertexState {
                    compilation_options: Default::default(),
                    module: &vs_shader_module,
                    entry_point: "vs_main".into(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                label: None,
                cache: None,
            });

            // HDR pipeline for internal render passes
            let float_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: Some(&pipeline_layout),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_shader_module,
                    compilation_options: Default::default(),
                    entry_point: "main".into(),
                    targets: &[Some(pass_texture.into())],
                }),
                vertex: wgpu::VertexState {
                    compilation_options: Default::default(),
                    module: &vs_shader_module,
                    entry_point: "vs_main".into(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                label: None,
                cache: None,
            });

            Pipeline::Pixel {
                pipeline,
                float_pipeline,
            }
        };

        Ok(RenderContext {
            uniforms,
            pipeline,
            passes: pass_structure,
            screen_buffer_cache: BufferCache::new(
                &format,
                1,
                1,
                None,
                device,
                document.stage == wgpu::naga::ShaderStage::Compute,
            ),
            stage: document.stage,
        })
    }

    /// Returns true if the loaded shader is a compute shader.
    pub fn is_compute(&self) -> bool {
        self.stage == ShaderStage::Compute
    }

    /// If the shader is a compute shader, then subsequent calls to
    /// `render_to_vec` and `render_to_slice`. will yield data from those
    /// storage textures.
    pub fn set_compute_target(&mut self, uniform_name: &str) -> Result<(), uniforms::Error> {
        self.uniforms.set_compute_target(uniform_name)?;
        Ok(())
    }

    /// Encodes the renderpasses and buffer copies in the correct order into
    /// `command` encoder targeting `view`.
    pub fn render<T: Into<Option<wgpu::TextureView>>>(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        target: T,
        width: u32,
        height: u32,
    ) {
        let target = target.into();

        match &self.pipeline {
            Pipeline::Compute { .. } => {
                self.encode_compute_render(command_encoder, target, device, queue, width, height);
            }
            Pipeline::Pixel { .. } => {
                // resize render targets and copy over texture contents for consistency
                self.update_pass_textures(command_encoder, device, width, height);

                // write changes to uniforms to gpu mapped buffers
                self.uniforms.update_uniform_buffers(device, queue);

                let target = target.unwrap_or_else(|| {
                    let fmt = self.uniforms.format();

                    if !self
                        .screen_buffer_cache
                        .supports_render(&fmt, width, height, None)
                    {
                        self.screen_buffer_cache =
                            BufferCache::new(&fmt, width, height, None, device, self.is_compute());
                    };

                    self.screen_buffer_cache
                        .tex()
                        .create_view(&Default::default())
                });

                self.encode_pixel_render(device, command_encoder, target);
            }
        }
    }

    fn encode_compute_render(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        target: Option<wgpu::TextureView>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) {
        if let Some(view) = target {
            self.uniforms.map_target_view(view);
        }

        self.uniforms
            .adjust_storage_texture_sizes(device, queue, width, height);

        self.uniforms
            .clear_ephemeral_targets_and_buffers(command_encoder);

        // write changes to uniforms to gpu mapped buffers
        self.uniforms.update_uniform_buffers(device, queue);

        for (idx, _) in self.passes.iter().enumerate() {
            self.uniforms.set_pass_index(idx, command_encoder);

            let Pipeline::Compute {
                ref compute_pipeline,
                workgroups: [x, y, z],
            } = self.pipeline
            else {
                return;
            };

            let mut rpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            rpass.set_pipeline(compute_pipeline);

            for (set, bind_group) in self.uniforms.iter_sets() {
                rpass.set_bind_group(set, bind_group, &[]);
            }

            if let Some(bytes) = self.uniforms.push_constant_bytes() {
                rpass.set_push_constants(0, bytes);
            }

            let dispatch_x = (width + x) / x;
            let dispatch_y = (height + y) / y;

            rpass.dispatch_workgroups(dispatch_x, dispatch_y, z);
            std::mem::drop(rpass);

            self.uniforms.forward_relays(command_encoder);
        }

        self.uniforms.reset_targets_to_context_managed();
        self.post_render();
    }

    fn encode_pixel_render(
        &mut self,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        view: wgpu::TextureView,
    ) {
        let Pipeline::Pixel {
            ref pipeline,
            ref float_pipeline,
        } = self.pipeline
        else {
            return;
        };

        for (idx, pass) in self.passes.iter().enumerate() {
            self.uniforms.set_pass_index(idx, command_encoder);

            // Run one pass to a render attachment,
            // then copy it to a bind group mapped buffer
            if let Some(tex_view) = pass.get_view() {
                let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: tex_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: pass.get_load_op(),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    depth_stencil_attachment: None,
                });

                rpass.set_pipeline(float_pipeline);

                for (set, bind_group) in self.uniforms.iter_sets() {
                    rpass.set_bind_group(set, bind_group, &[]);
                }

                if let Some(bytes) = self.uniforms.push_constant_bytes() {
                    rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, bytes);
                }

                rpass.draw(0..3, 0..1);
            } else {
                let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    depth_stencil_attachment: None,
                });

                rpass.set_pipeline(pipeline);

                for (set, bind_group) in self.uniforms.iter_sets() {
                    rpass.set_bind_group(set, bind_group, &[]);
                }
                if let Some(bytes) = self.uniforms.push_constant_bytes() {
                    rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, bytes);
                }
                rpass.draw(0..3, 0..1);
            }

            self.uniforms.forward_relays(command_encoder);

            // copy the render pass target over to the
            // texture that is used in the pipeline
            if let Some((target_tex, _)) = pass.render_target_texture.as_ref() {
                if let Some(bind_group_tex) = pass
                    .pass_target_var_name
                    .as_ref()
                    .and_then(|name| self.uniforms.get_texture(name))
                {
                    command_encoder.copy_texture_to_texture(
                        target_tex.as_image_copy(),
                        bind_group_tex.as_image_copy(),
                        wgpu::Extent3d {
                            width: target_tex.width(),
                            height: target_tex.height(),
                            depth_or_array_layers: 1,
                        },
                    );
                };
            };
        }

        let mut set = std::collections::HashSet::new();
        // take all but the output pass
        for pass in self.passes.iter().take(self.passes.len() - 1) {
            // zero out the target texture
            // if the pass wasn't persistent
            // the clear loadop does not work
            if !pass.persistent {
                if let Some(ref tex_name) = pass.pass_target_var_name {
                    if !set.contains(tex_name) {
                        break;
                    } else {
                        set.insert(tex_name.clone());
                    }

                    if device.features().contains(wgpu::Features::CLEAR_TEXTURE) {
                        if let Some(tex) = self.uniforms.get_texture(tex_name) {
                            command_encoder.clear_texture(
                                tex,
                                &wgpu::ImageSubresourceRange {
                                    aspect: wgpu::TextureAspect::All,
                                    base_mip_level: 0,
                                    mip_level_count: None,
                                    base_array_layer: 0,
                                    array_layer_count: None,
                                },
                            );
                        }
                    } else {
                        let Some(tex) = self.uniforms.get_texture(tex_name) else {
                            continue;
                        };

                        let view = tex.create_view(&Default::default());

                        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Clear Texture Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
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
            }
        }

        self.post_render();
    }

    /// Returns a iterator over mutable custom values and names of
    /// the inputs provided by the user, as well as
    /// the raw bytes of all the uniforms maintained by the [RenderContext]
    /// that do not have input pragmas.
    pub fn iter_inputs_mut(&mut self) -> impl Iterator<Item = (&String, MutInput)> {
        self.uniforms.iter_custom_uniforms_mut()
    }

    /// Returns a iterator over custom values and names of
    /// the inputs provided by the user, as well as
    /// the raw bytes of all the uniforms maintained by the [RenderContext]
    /// that do not have input pragmas.
    pub fn iter_inputs(&self) -> impl Iterator<Item = (&String, &InputType)> {
        self.uniforms.iter_custom_uniforms()
    }

    /// Returns an option of a mutable reference to the custom input of the given name if it exists
    pub fn get_input_mut(&mut self, name: &str) -> Option<MutInput> {
        self.uniforms.get_input_mut(name)
    }

    /// Optionally returns a reference to the underlying type of a variable in
    /// the uniforms.
    pub fn get_input_as<T>(&mut self, name: &str) -> Option<&mut T>
    where
        InputType: TryAsMut<T>,
    {
        self.uniforms
            .get_input_mut(name)
            .and_then(|i| i.inner.try_as_mut())
    }

    /// Returns an option of the custom input of the given name if it exists
    pub fn get_input(&self, name: &str) -> Option<&InputType> {
        self.uniforms.get_input(name)
    }

    /// Loads a texture of the specified format into the variable with name `name`
    /// immediately.
    pub fn load_texture<S: AsRef<str>>(
        &mut self,
        name: S,
        texture_desc: TextureDesc,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.uniforms
            .load_texture(name.as_ref(), texture_desc, device, queue);
    }

    /// Creates a texture view and maps it to the pipeline in place of a locally
    /// stored texture. this will fail if you try to override a render target texture.
    pub fn load_shared_texture(&mut self, texture: &wgpu::Texture, variable_name: &str) -> bool {
        // fizzle on attempting to write a target texture
        if self
            .passes
            .iter()
            .filter_map(|p| p.pass_target_var_name.as_ref())
            .any(|t| t == variable_name)
        {
            return false;
        }

        self.uniforms
            .override_texture_view_with_tex(variable_name, texture)
    }

    /// If a texture is loaded in the pipeline under `variable_name` this will reference to it.
    /// this includes storage textures
    pub fn get_texture(&mut self, variable_name: &str) -> Option<&wgpu::Texture> {
        let tex = self.uniforms.get_texture(variable_name)?;
        Some(tex)
    }

    /// returns an instance of the render context in a default error state
    /// displaying a test card  Useful when displaying errors to the user
    /// if you don't want to bail the program and have no visual fallback.
    pub fn error_state(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        // If we can't work with the provided shader open an error view
        // and print the errors on the watched files until they work
        Self::new(
            include_str!("../resources/error.glsl"),
            output_format,
            device,
            queue,
        )
        .unwrap()
    }

    /// Renders the shader at provided resolution and writes the result to a vector
    /// in the texture format that the context was configured with.
    /// ## Warning!
    /// this function is pretty wasteful and memory intensive
    /// It is mostly here as a utitlity to ffi interfaces.
    pub fn render_to_vec<'a>(
        &'a mut self,
        queue: &'a wgpu::Queue,
        device: &'a wgpu::Device,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let block_size = self
            .uniforms
            .format()
            .block_copy_size(Some(wgpu::TextureAspect::All))
            .expect("It seems like you are trying to render to a Depth Stencil. Stop that.");

        let fmt = self.uniforms.format();

        if !self
            .screen_buffer_cache
            .supports_render(&fmt, width, height, None)
        {
            self.screen_buffer_cache =
                BufferCache::new(&fmt, width, height, None, device, self.is_compute());
        };

        let view = self
            .screen_buffer_cache
            .tex()
            .create_view(&Default::default());

        let mut command_encoder = device.create_command_encoder(&Default::default());
        self.render(queue, device, &mut command_encoder, view, width, height);
        queue.submit(Some(command_encoder.finish()));

        let mut out = vec![0; (width * height * block_size) as usize];

        read_texture_contents_to_slice(
            device,
            queue,
            &self.screen_buffer_cache,
            height,
            width,
            None,
            &mut out,
        );

        out
    }

    /// Renders the shader at provided resolution and writes the result to the user provided
    /// slice in the texture format that the context was configured with.
    ///
    /// # Panics!
    ///
    /// if the provided slice does not exactly match the buffer size requirements of
    /// the resolution and output format specified in the context.
    pub fn render_to_slice<'a>(
        &'a mut self,
        queue: &'a wgpu::Queue,
        device: &'a wgpu::Device,
        width: u32,
        height: u32,
        slice: &mut [u8],
        stride: Option<u32>,
    ) {
        let fmt = self.uniforms.format();

        if !self
            .screen_buffer_cache
            .supports_render(&fmt, width, height, None)
        {
            self.screen_buffer_cache =
                BufferCache::new(&fmt, width, height, None, device, self.is_compute());
        };

        let view = self
            .screen_buffer_cache
            .tex()
            .create_view(&Default::default());

        let mut command_encoder = device.create_command_encoder(&Default::default());
        self.render(queue, device, &mut command_encoder, view, width, height);
        queue.submit(Some(command_encoder.finish()));

        read_texture_contents_to_slice(
            device,
            queue,
            &self.screen_buffer_cache,
            height,
            width,
            stride,
            slice,
        );
    }

    // resizes all the render pass target textures
    // or initializes them.
    fn update_pass_textures(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) {
        if self
            .passes
            .iter()
            .all(|pass| pass.pass_target_var_name.is_none())
        {
            return;
        }
        let compute = self.is_compute();

        for pass in self.passes.iter_mut() {
            let Some(target) = pass.pass_target_var_name.clone() else {
                continue;
            };

            if pass.needs_resize(height, width) {
                let new_width = pass.const_width.unwrap_or(width);
                let new_height = pass.const_height.unwrap_or(height);

                let binding_tex = device.create_texture(&render_pass_result_desc(
                    new_width,
                    new_height,
                    pass.target_format,
                ));

                let new_target = device.create_texture(&target_desc(
                    new_width,
                    new_height,
                    pass.target_format,
                    compute,
                ));

                if let Some((old_target, _)) = pass.render_target_texture.as_ref() {
                    let min_width = u32::min(new_width, old_target.width());
                    let min_height = u32::min(new_height, old_target.height());

                    command_encoder.copy_texture_to_texture(
                        old_target.as_image_copy(),
                        binding_tex.as_image_copy(),
                        wgpu::Extent3d {
                            width: min_width,
                            height: min_height,
                            depth_or_array_layers: 1,
                        },
                    );

                    command_encoder.copy_texture_to_texture(
                        old_target.as_image_copy(),
                        new_target.as_image_copy(),
                        wgpu::Extent3d {
                            width: min_width,
                            height: min_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                let view = new_target.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(new_target.format()),
                    ..Default::default()
                });
                pass.render_target_texture = Some((new_target, view));
                self.uniforms.set_texture(target.as_ref(), binding_tex);
            }
        }
    }

    // utility function for post render cleanup
    fn post_render(&mut self) {
        // reset the mouse
        let old = self.uniforms.global_data_mut().mouse[3];
        self.uniforms.global_data_mut().mouse[3] = -f32::abs(old);
    }

    /// Removes a texture with the variable name `var` from the pipeline,
    /// It will be replaced with a placeholder texture which is a 1x1 transparent pixel.
    /// returns true if the texture existed.
    pub fn remove_texture(&mut self, var: &str) -> bool {
        self.uniforms.unload_texture(var)
    }

    /// Iterate over the names and peristence states of the
    /// storage texture targets in the document.
    pub fn iter_targets(&self) -> impl Iterator<Item = uniforms::TargetDescriptor<'_>> {
        self.uniforms.iter_targets()
    }

    /// Returns true if this render context builds up
    /// a state over its runtime using persistent targets
    pub fn is_stateful(&mut self) -> bool {
        self.passes.iter().any(|pass| pass.persistent)
            || self.uniforms.iter_targets().any(|targ| targ.persistent)
    }

    /// copy all common textures and
    /// common uniform values from `self` to `other`.
    /// Will not copy textures loaded with `load_shared_texture`
    pub fn copy_resources_into(
        &mut self,
        other: &mut Self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.uniforms.copy_into(&mut other.uniforms, device, queue);
    }

    /// Call this to put the mouse in the down position,
    /// this resets the `z` and `w` components of the `mouse`
    /// uniform to the most recent mouse position.
    pub fn set_mouse_down(&mut self) {
        let [x, y, ..] = self.uniforms.global_data_mut().mouse;
        self.uniforms.global_data_mut().mouse = [x, y, x, y];
    }

    /// Call this to put the mouse in the down up position,
    /// this sets the sign of the `z` and `w` components of the `mouse`
    /// uniform negative.
    pub fn set_mouse_up(&mut self) {
        let [x, y, z, w] = self.uniforms.global_data_mut().mouse;
        self.uniforms.global_data_mut().mouse = [x, y, -f32::abs(z), -f32::abs(w)];
    }

    /// Updates the `mouse` uniform.
    pub fn set_mouse_input(&mut self, new_position: [f32; 2]) {
        let data = self.uniforms.global_data_mut();
        let last_mouse = data.mouse;
        let size = data.resolution;

        let (x, y) = {
            (
                f32::clamp(new_position[0], 0.0, size[0]),
                f32::clamp(size[1] - new_position[1], 0.0, size[1]),
            )
        };

        let out = match (
            last_mouse[2].is_sign_positive(),
            last_mouse[3].is_sign_positive(),
        ) {
            (true, true) => [x, y, x, y],
            (true, false) => [x, y, f32::abs(last_mouse[2]), -f32::abs(last_mouse[3])],
            (false, false) => [
                last_mouse[0],
                last_mouse[1],
                -f32::abs(last_mouse[2]),
                -f32::abs(last_mouse[3]),
            ],
            (false, true) => [x, y, -f32::abs(last_mouse[2]), f32::abs(last_mouse[3])],
        };

        self.uniforms.global_data_mut().mouse = out;
    }

    /// Updates the `frame_index` utility uniform member.
    pub fn update_frame_count(&mut self, frame: u32) {
        self.uniforms.global_data_mut().frame = frame;
    }

    /// Updates the `date` utility uniform member with values corresponding to
    /// year,  month, day, and seconds since midnight.
    pub fn update_datetime(&mut self, date: [f32; 4]) {
        self.uniforms.global_data_mut().date = date;
    }

    /// Updates the `time_delta` utility uniform with the delta value.
    pub fn update_delta(&mut self, delta: f32) {
        self.uniforms.global_data_mut().time_delta = delta;
    }

    /// Updates the `time` uniform member.
    pub fn update_time(&mut self, time: f32) {
        self.uniforms.global_data_mut().time = time;
    }

    /// Updates the `resolution` uniform member.
    pub fn update_resolution(&mut self, res: [f32; 2]) {
        self.uniforms.global_data_mut().resolution = [res[0], res[1], res[0] / res[1]];
    }
}

#[derive(Debug)]
struct RenderPass {
    target_format: wgpu::TextureFormat,
    // whether or not the buffer is cleared every render
    persistent: bool,
    // the width if constant otherwise maps to the current window width
    const_width: Option<u32>,
    // the height if constant otherwise maps to the window height
    const_height: Option<u32>,
    // The texture variable name, if it exists
    pass_target_var_name: Option<String>,
    // The target to render into
    pub render_target_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl RenderPass {
    pub fn needs_resize(&self, target_height: u32, target_width: u32) -> bool {
        let target_height = self.const_height.unwrap_or(target_height);
        let target_width = self.const_width.unwrap_or(target_width);
        let mapped_as_input = self.pass_target_var_name.is_some();
        let unallocated = self.render_target_texture.is_none();
        let wrong_size = self
            .render_target_texture
            .as_ref()
            .is_some_and(|t| t.0.height() != target_height)
            || self
                .render_target_texture
                .as_ref()
                .is_some_and(|t| t.0.width() != target_width);
        (mapped_as_input && unallocated) || (!unallocated && wrong_size)
    }

    pub fn get_view(&self) -> Option<&wgpu::TextureView> {
        self.render_target_texture.as_ref().map(|t| &t.1)
    }

    pub fn get_load_op(&self) -> wgpu::LoadOp<wgpu::Color> {
        if self.persistent {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        }
    }

    pub fn new(value: &crate::parsing::RenderPass, target_format: TextureFormat) -> Self {
        Self {
            target_format,
            persistent: value.persistent,
            const_width: value.width,
            const_height: value.height,
            pass_target_var_name: value.target_texture.clone(),
            render_target_texture: None,
        }
    }
}

#[derive(Debug)]
pub struct BufferCache {
    tex: wgpu::Texture,
    /// 256 byte aligned buffer
    buf: wgpu::Buffer,
    stride: usize,
}

impl BufferCache {
    pub fn new(
        format: &wgpu::TextureFormat,
        width: u32,
        height: u32,
        stride: Option<usize>,
        device: &wgpu::Device,
        compute: bool,
    ) -> Self {
        let block_size = format
            .block_copy_size(Some(wgpu::TextureAspect::All))
            .unwrap();

        let row_byte_ct = stride.unwrap_or((block_size * width) as usize);
        let stride = (row_byte_ct + 255) & !255;

        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Read Buffer"),
            size: (height as usize * stride) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tex = device.create_texture(&target_desc(width, height, *format, compute));

        Self { buf, tex, stride }
    }

    //TODO: allow using subsections of the buffer and texture through views
    // reallocating if the requested size is too small.
    pub fn supports_render(
        &self,
        fmt: &wgpu::TextureFormat,
        width: u32,
        height: u32,
        stride: Option<usize>,
    ) -> bool {
        let block_size = fmt.block_copy_size(Some(wgpu::TextureAspect::All)).unwrap() as usize;

        let stride_matches = stride.is_some_and(|s| s == self.stride)
            || (stride.is_none() && width as usize * block_size == self.stride);

        self.tex.format() == *fmt
            && self.tex.height() == height
            && self.tex.width() == width
            && stride_matches
    }

    pub fn tex(&self) -> &wgpu::Texture {
        &self.tex
    }

    pub fn buf(&self) -> &wgpu::Buffer {
        &self.buf
    }
}

// if a texture format is used as a shader color target,
// this returns true if it requires a vec4 color output.
fn is_floating_point_in_shader(format: &wgpu::TextureFormat) -> bool {
    !matches!(
        format,
        TextureFormat::R8Uint
            | TextureFormat::R8Sint
            | TextureFormat::R16Uint
            | TextureFormat::R16Sint
            | TextureFormat::Rg8Uint
            | TextureFormat::Rg8Sint
            | TextureFormat::Rg16Uint
            | TextureFormat::Rg16Sint
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba8Sint
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba16Sint
            | TextureFormat::R32Uint
            | TextureFormat::R32Sint
            | TextureFormat::Rg32Uint
            | TextureFormat::Rg32Sint
            | TextureFormat::Rgba32Uint
            | TextureFormat::Rgba32Sint
    )
}

// template for input textures, copied to from the gpu render target
fn render_pass_result_desc(
    width: u32,
    height: u32,
    format: TextureFormat,
) -> wgpu::TextureDescriptor<'static> {
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
        format,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

// template for targets written to by the gpu
fn target_desc(
    width: u32,
    height: u32,
    format: TextureFormat,
    compute: bool,
) -> wgpu::TextureDescriptor<'static> {
    let mut usage = wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::COPY_SRC;

    if compute {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }

    wgpu::TextureDescriptor {
        usage,
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1, // crunch crunch
        dimension: wgpu::TextureDimension::D2,
        format,
        view_formats: &[],
    }
}

fn display_errors(errors: &[glsl::Error], source: &str) -> String {
    let mut output_string = Vec::new();
    for e in errors {
        let glsl::Error { kind, meta } = e;
        let loc = meta.location(source);
        let output = pretty_print_error(source, loc, format!("{kind}"));
        output_string.push(output)
    }
    output_string.join("\n")
}

fn pretty_print_error(source: &str, location: naga::SourceLocation, kind: String) -> String {
    let mut result = String::new();
    let lines: Vec<&str> = source.lines().collect();

    if location.line_number <= lines.len() as u32 {
        let line = lines[(location.line_number - 1) as usize].trim();

        // Build the error message
        result.push_str(&format!(
            "{kind} at location {}:{} :  {line}  \n",
            location.line_number, location.line_position
        ));
    } else {
        result.push_str("Invalid line number in source location.");
    }

    result
}

// slow, bad, synchronous.
fn read_texture_contents_to_slice(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cpu_buffer_cache: &BufferCache,
    height: u32,
    width: u32,
    stride: Option<u32>,
    slice: &mut [u8],
) {
    // Create a command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Read Encoder"),
    });

    // Copy the texture contents to the buffer
    encoder.copy_texture_to_buffer(
        cpu_buffer_cache.tex().as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: cpu_buffer_cache.buf(),
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cpu_buffer_cache.stride as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let block_size = cpu_buffer_cache
        .tex()
        .format()
        .block_copy_size(Some(wgpu::TextureAspect::All))
        .unwrap();

    let row_byte_ct = stride.unwrap_or(block_size * width);

    {
        let buffer_slice = cpu_buffer_cache.buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| r.unwrap());
        let _ = device.poll(wgpu::Maintain::Wait);

        let gpu_slice = buffer_slice.get_mapped_range();
        let gpu_chunks = gpu_slice.chunks(cpu_buffer_cache.stride);
        let slice_chunks = slice.chunks_mut(row_byte_ct as usize);
        let iter = slice_chunks.zip(gpu_chunks);

        for (output_chunk, gpu_chunk) in iter {
            output_chunk.copy_from_slice(&gpu_chunk[..row_byte_ct as usize]);
        }
    };

    cpu_buffer_cache.buf().unmap();
}
