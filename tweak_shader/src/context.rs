use crate::input_type::*;
use crate::uniforms;
use crate::VarName;
use naga::front::glsl;
use wgpu::naga;

use naga::{
    front::glsl::{Frontend, Options},
    ShaderStage,
};

use std::collections::BTreeMap;

use crate::{Error, UserJobs};

use wgpu::TextureFormat;

/// The main rendering and bookkeeping context.
#[derive(Debug)]
pub struct RenderContext {
    uniforms: uniforms::Uniforms,
    passes: Vec<RenderPass>,
    pipeline: wgpu::RenderPipeline,
    streams: BTreeMap<VarName, StreamInfo>,
    texture_job_queue: BTreeMap<VarName, TextureJob>,
    user_set_up_jobs: Vec<crate::UserJobs>,
    cpu_view_cache: Option<wgpu::Texture>,
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

        let document =
            crate::parsing::parse_document(source).map_err(Error::DocumentParsingFailed)?;

        let user_set_up_jobs = document
            .preloads
            .iter()
            .filter_map(
                |(var_name, location)| match document.inputs.get(var_name.as_str()) {
                    Some(InputType::Audio(_, max_samples)) => Some(UserJobs::LoadAudioFile {
                        location: std::path::PathBuf::from(location),
                        var_name: var_name.clone(),
                        fft: false,
                        max_samples: *max_samples,
                    }),

                    Some(InputType::AudioFft(_, max_samples)) => Some(UserJobs::LoadAudioFile {
                        location: std::path::PathBuf::from(location),
                        var_name: var_name.clone(),
                        fft: true,
                        max_samples: *max_samples,
                    }),

                    Some(InputType::Image(_)) => Some(UserJobs::LoadImageFile {
                        location: std::path::PathBuf::from(location),
                        var_name: var_name.clone(),
                    }),
                    _ => None,
                },
            )
            .collect();

        let mut frontend = Frontend::default();

        let stripped_src: String = source
            .lines()
            .filter(|line| !line.trim().starts_with("#pragma"))
            .collect::<Vec<_>>()
            .join("\n");

        let mut options = Options::from(ShaderStage::Fragment);

        options
            .defines
            .insert("TWEAK_SHADER".to_owned(), "1".to_owned());

        let naga_mod = frontend
            .parse(&options, &stripped_src)
            .map_err(|e| Error::ShaderCompilationFailed(display_errors(&e, &stripped_src)))?;

        let mut isf_pass_structure = vec![];

        isf_pass_structure.extend(
            document
                .passes
                .iter()
                .map(|pass| RenderPass::new(pass, format)),
        );

        isf_pass_structure.push(RenderPass::new(&Default::default(), format));

        // collect all set indices, find the max then create bind sets contiguous up to max.
        // some might be empty.
        let sets = uniforms::sets(&naga_mod);
        let sets = sets.iter().map(|s| *s as i32).next_back().unwrap_or(-1);
        let sets = (0..(sets + 1))
            .map(|s| {
                uniforms::BindGroup::new_from_naga(s as u32, &naga_mod, &document, device, queue)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(crate::Error::UniformError)?;

        let push_const =
            uniforms::push_constant(&naga_mod, &document).map_err(Error::UniformError)?;

        let uniforms = uniforms::Uniforms::new(
            &document,
            &format,
            device,
            queue,
            sets,
            push_const,
            isf_pass_structure.len(),
        )
        .map_err(Error::UniformError)?;

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

        let bind_group_layouts: Vec<_> = uniforms.iter_layouts().collect();

        let push_constant_ranges = uniforms
            .push_constant_ranges()
            .map(|e| vec![e])
            .unwrap_or(vec![]);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: bind_group_layouts.as_slice(),
            push_constant_ranges: &push_constant_ranges,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: Some(&pipeline_layout),
            fragment: Some(wgpu::FragmentState {
                module: &fs_shader_module,
                entry_point: "main",
                targets: &[Some(format.into())],
            }),
            vertex: wgpu::VertexState {
                module: &vs_shader_module,
                entry_point: "vs_main",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            label: None,
        });

        Ok(RenderContext {
            uniforms,
            user_set_up_jobs,
            pipeline,
            passes: isf_pass_structure,
            cpu_view_cache: None,
            texture_job_queue: BTreeMap::new(),
            streams: BTreeMap::new(),
        })
    }

    /// Returns a slice of jobs the user should complete.
    /// see [UserJobs] for more information. handling these jobs is totally
    /// optional.
    pub fn list_set_up_jobs(&self) -> &[UserJobs] {
        &self.user_set_up_jobs
    }

    /// Renders the shader maintained by this context to the provided texture view.
    /// this will produce validation errors if the view format does not match the
    /// format the context was configured with in [RenderContext::new].
    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let mut command_encoder = device.create_command_encoder(&Default::default());
        self.encode_render(queue, device, &mut command_encoder, view, width, height);
        queue.submit(Some(command_encoder.finish()));
    }

    /// Encodes the renderpasses and buffer copies in the correct order into
    /// `command` encoder targeting `view`.
    pub fn encode_render(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        // resize render targets and copy over texture contents for consistency
        self.update_isf_pass_textures(command_encoder, device, width, height);
        // updates video, audio, streams, shows new images.
        self.update_display_textures(device, queue);
        // write changes to uniforms to gpu mapped buffers
        self.uniforms.update_uniform_buffers(device, queue);

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

                rpass.set_pipeline(&self.pipeline);
                for (set, bind_group) in self.uniforms.iter_sets() {
                    rpass.set_bind_group(set, bind_group, &[]);
                }
                if let Some(bytes) = self.uniforms.push_constant_bytes() {
                    rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, bytes);
                }
                rpass.draw(0..6, 0..1);
            } else {
                let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
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

                rpass.set_pipeline(&self.pipeline);
                for (set, bind_group) in self.uniforms.iter_sets() {
                    rpass.set_bind_group(set, bind_group, &[]);
                }
                if let Some(bytes) = self.uniforms.push_constant_bytes() {
                    rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, bytes);
                }
                rpass.draw(0..6, 0..1);
            }

            // copy the render pass target over to the
            // texture that is used in the pipeline
            if let Some((target_tex, _)) = pass.render_target_texture.as_ref() {
                if let Some(bind_group_tex) = pass
                    .bind_group_target_texture
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

            // zero out the target texture
            // if the pass wasn't persistent
            // the clear loadop does not work
            if !pass.persistent {
                if let Some(ref tex_name) = pass.bind_group_target_texture {
                    let size = height * width * 4;
                    let slice = &vec![0; size as usize];
                    self.uniforms
                        .load_texture(tex_name, slice, height, width, device, queue);
                }
            }
        }

        self.post_render();
    }

    /// Returns a iterator over mutable custom values and names of
    /// the inputs provided by the user, as well as
    /// the raw bytes of all the uniforms maintained by the [RenderContext]
    /// that do not have input pragmas.
    pub fn iter_inputs_mut(&mut self) -> impl Iterator<Item = (&str, MutInput)> {
        self.uniforms.iter_custom_uniforms_mut()
    }

    /// Returns a iterator over custom values and names of
    /// the inputs provided by the user, as well as
    /// the raw bytes of all the uniforms maintained by the [RenderContext]
    /// that do not have input pragmas.
    pub fn iter_inputs(&mut self) -> impl Iterator<Item = (&str, &InputType)> {
        self.uniforms.iter_custom_uniforms()
    }

    /// Returns an option of a mutable reference to the custom input of the given name if it exists
    pub fn get_input_mut(&mut self, name: &str) -> Option<MutInput> {
        self.uniforms.get_input_mut(name)
    }

    /// Returns an option of the custom input of the given name if it exists
    pub fn get_input(&self, name: &str) -> Option<&InputType> {
        self.uniforms.get_input(name)
    }

    /// Provides access to the raw float array of sample data
    /// that will be converted into a texture
    /// for the audio stream with the name `name`
    /// * The data must be in a planar format.
    /// * The data may have an FFT performed on it
    /// returns None if the name provided does not correspond to a
    /// valid audio input.
    pub fn load_audio_stream_raw<S: AsRef<str>>(
        &mut self,
        name: S,
        channels: u16,
        samples: usize,
    ) -> Option<&mut [f32]> {
        if self.streams.get_mut(name.as_ref()).is_some() {
            // weird borrow checker nuance
            let stream = self.streams.get_mut(name.as_ref()).unwrap();
            match stream {
                StreamInfo::Audio { dirty, data, .. } => {
                    *dirty = true;
                    Some(data.as_mut_slice())
                }
                _ => None,
            }
        } else if self
            .uniforms
            .get_input_mut(name.as_ref())
            .is_some_and(|ty| matches!(ty.variant(), InputVariant::AudioFft | InputVariant::Audio))
        {
            let val = StreamInfo::Audio {
                dirty: true,
                channels: channels as usize,
                samples,
                data: vec![0.0; channels as usize * samples],
            };
            self.streams.insert(name.as_ref().to_owned(), val);
            if let StreamInfo::Audio { data, .. } = self.streams.get_mut(name.as_ref())? {
                Some(data.as_mut_slice())
            } else {
                // unreachable
                None
            }
        } else {
            None
        }
    }

    /// Provides access to the raw byte array of rgba8unormsrgb
    /// formatted pixels that will be uploaded into the texture
    /// with name `name`. This causes the context to allocate memory
    /// large enough to store a frame of your video until you call
    /// [`RenderContext::remove_texture`].
    pub fn load_video_stream_raw<S: AsRef<str>>(
        &mut self,
        name: S,
        height: u32,
        width: u32,
    ) -> Option<&mut [u8]> {
        if self.streams.get_mut(name.as_ref()).is_some() {
            // weird borrow checker nuance
            let stream = self.streams.get_mut(name.as_ref()).unwrap();
            match stream {
                StreamInfo::Video { dirty, data, .. } => {
                    *dirty = true;
                    Some(data.as_mut_slice())
                }
                _ => None,
            }
        } else if self
            .uniforms
            .get_input_mut(name.as_ref())
            .is_some_and(|ty| matches!(ty.variant(), InputVariant::Image))
        {
            let val = StreamInfo::Video {
                dirty: true,
                height,
                width,
                data: vec![0; (height * width * 4) as usize],
            };
            self.streams.insert(name.as_ref().to_owned(), val);
            if let StreamInfo::Video { data, .. } = self.streams.get_mut(name.as_ref())? {
                Some(data.as_mut_slice())
            } else {
                // unreachable
                None
            }
        } else {
            None
        }
    }

    /// Creates a texture view and maps it to the pipeline in place of a locally
    /// stored texture. this will fail if you try to override a render target texture.
    pub fn load_shared_texture(&mut self, texture: &wgpu::Texture, variable_name: &str) -> bool {
        // fizzle on attempting to write a target texture
        if self
            .passes
            .iter()
            .filter_map(|p| p.bind_group_target_texture.as_ref())
            .any(|t| t == variable_name)
        {
            return false;
        }

        self.uniforms.override_texture_view(variable_name, texture)
    }

    /// If a texture is loaded in the pipeline under `variable_name` this will return a new view into it.
    pub fn get_texture_view(&mut self, variable_name: &str) -> Option<wgpu::TextureView> {
        let tex = self.uniforms.get_texture(variable_name)?;
        let desc = wgpu::TextureViewDescriptor {
            label: Some(variable_name),
            format: Some(tex.format()),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        };
        Some(tex.create_view(&desc))
    }

    /// Queues the texture with `variable_name`
    /// to be written to with the data in `data`.
    /// data is assumed to be in rgba8unormsrgb format.
    /// data will NOT be loaded if it is the wrong size
    /// or if you attempt to write to a
    /// render pass target texture.
    pub fn load_texture(
        &mut self,
        data: Vec<u8>,
        variable_name: String,
        width: u32,
        height: u32,
    ) -> bool {
        // fizzle on attempting to write a target texture
        if self
            .passes
            .iter()
            .filter_map(|p| p.bind_group_target_texture.as_ref())
            .any(|t| t == &variable_name)
        {
            return false;
        }

        if data.len() != (width * height * 4) as usize {
            return false;
        }

        // only keep the most recent texture update
        self.texture_job_queue.insert(
            variable_name.clone(),
            TextureJob {
                data,
                width,
                height,
                variable_name,
            },
        );
        true
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
            .block_size(Some(wgpu::TextureAspect::All))
            .expect("It seems like you are trying to render to a Depth Stencil. Stop that.");

        let mut out = vec![0; (block_size * width * height) as usize];

        if !self
            .cpu_view_cache
            .as_ref()
            .is_some_and(|t| t.width() == width && t.height() == height)
        {
            let tex = device.create_texture(&target_desc(width, height, self.uniforms.format()));
            self.cpu_view_cache = Some(tex);
        };

        let view = self
            .cpu_view_cache
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        self.render(queue, device, &view, width, height);

        read_texture_contents_to_slice(
            device,
            queue,
            self.cpu_view_cache.as_ref().unwrap(),
            &self.uniforms.format(),
            height,
            width,
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
    ) {
        if !self
            .cpu_view_cache
            .as_ref()
            .is_some_and(|t| t.width() == width && t.height() == height)
        {
            let tex = device.create_texture(&target_desc(width, height, self.uniforms.format()));
            self.cpu_view_cache = Some(tex);
        };

        let view = self
            .cpu_view_cache
            .as_ref()
            .unwrap()
            .create_view(&Default::default());

        self.render(queue, device, &view, width, height);

        read_texture_contents_to_slice(
            device,
            queue,
            self.cpu_view_cache.as_ref().unwrap(),
            &self.uniforms.format(),
            height,
            width,
            slice,
        );
    }

    fn update_display_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for (name, val) in self.streams.iter_mut() {
            match val {
                StreamInfo::Video {
                    dirty,
                    height,
                    width,
                    ref mut data,
                } if *dirty => {
                    self.uniforms
                        .load_texture(name, data.as_ref(), *height, *width, device, queue);
                    *dirty = false;
                }
                StreamInfo::Audio {
                    dirty,
                    samples,
                    channels,
                    data,
                } if *dirty => {
                    self.uniforms.load_texture(
                        name,
                        float_to_rgba8_snorm(data).as_slice(),
                        *channels as u32,
                        *samples as u32,
                        device,
                        queue,
                    );
                    *dirty = false;
                }
                _ => {}
            }
        }

        while let Some((_, job)) = self.texture_job_queue.pop_first() {
            self.uniforms.load_texture(
                &job.variable_name,
                job.data.as_ref(),
                job.height,
                job.width,
                device,
                queue,
            );
        }
    }

    // resizes all the render pass target textures
    // or initializes them.
    fn update_isf_pass_textures(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) {
        if self
            .passes
            .iter()
            .all(|pass| pass.bind_group_target_texture.is_none())
        {
            return;
        }

        for pass in self.passes.iter_mut() {
            let Some(target) = pass.bind_group_target_texture.clone() else {
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

                let new_target =
                    device.create_texture(&target_desc(new_width, new_height, pass.target_format));

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

    // Removes a texture with the variable name `var` from the pipeline,
    // It will be replaced with a placeholder texture which is a 1x1 black pixel.
    // returns true if the texture existed.
    pub fn remove_texture(&mut self, var: &str) -> bool {
        let stat = self.uniforms.unload_texture(var);
        let stream = self.streams.remove(var).is_some();
        stat || stream
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
enum StreamInfo {
    Video {
        dirty: bool,
        height: u32,
        width: u32,
        data: Vec<u8>,
    },
    Audio {
        dirty: bool,
        samples: usize,
        channels: usize,
        data: Vec<f32>,
    },
}

#[derive(Debug)]
// a queued rgba8unormsrgb texture job
struct TextureJob {
    data: Vec<u8>,
    variable_name: String,
    height: u32,
    width: u32,
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
    bind_group_target_texture: Option<String>,
    // The target to render into
    pub render_target_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl RenderPass {
    pub fn needs_resize(&self, target_height: u32, target_width: u32) -> bool {
        let target_height = self.const_height.unwrap_or(target_height);
        let target_width = self.const_width.unwrap_or(target_width);
        let mapped_as_input = self.bind_group_target_texture.is_some();
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
            bind_group_target_texture: value.target_texture.clone(),
            render_target_texture: None,
        }
    }
}

struct NoopWriter;

impl std::fmt::Write for NoopWriter {
    fn write_str(&mut self, _x: &str) -> std::fmt::Result {
        Ok(())
    }
}

// template for every texture ever loaded by out pipelines
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

// template for every texture ever loaded by out pipelines
fn target_desc(width: u32, height: u32, format: TextureFormat) -> wgpu::TextureDescriptor<'static> {
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
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

fn float_to_rgba8_snorm(input: &[f32]) -> Vec<u8> {
    let mut output = vec![0; input.len() * 4];

    let sum: f32 = input.iter().sum();
    let avg = sum / input.len() as f32;

    for (i, &value) in input.iter().enumerate() {
        let normalized_value = 0.75 + (value - avg);

        let color = (normalized_value * 255.0).round().clamp(0.0, 255.0) as u8;

        output[i * 4] = color;
        output[i * 4 + 1] = color;
        output[i * 4 + 2] = color;
        output[i * 4 + 3] = 255;
    }

    output
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
    texture: &wgpu::Texture,
    format: &wgpu::TextureFormat,
    height: u32,
    width: u32,
    slice: &mut [u8],
) {
    let block_size = format
        .block_size(Some(wgpu::TextureAspect::All))
        .expect("It seems like you are trying to render to a Depth Stencil. Stop that.");
    // Create a buffer to store the texture data
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Texture Read Buffer"),
        size: (height * width * block_size) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create a command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Read Encoder"),
    });

    // Copy the texture contents to the buffer
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(block_size * width),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    {
        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::Maintain::Wait);

        slice.copy_from_slice(buffer_slice.get_mapped_range().as_ref());
    };

    buffer.unmap();
}
