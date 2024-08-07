use crate::initialization::GuiContext;
use crate::read_file;
use crate::ui;
use crate::ui::UiState;
use crate::video::{VideoLoader, VideoLoaderTrait};

use chrono::{Datelike, Local, Timelike};

use tweak_shader::Error;
use tweak_shader::RenderContext;

use notify::RecommendedWatcher;

use std::cell::Cell;
use std::collections::BTreeMap;

use std::path::Path;
use std::path::PathBuf;

use egui_wgpu::wgpu;

#[derive(Debug)]
enum RunnerError {
    Validation(String),
    MissingFile,
    Shader(Error),
    Video(String),
    Image(String),
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RunnerError::Validation(msg) => write!(f, "Validation error: {}", msg),
            RunnerError::MissingFile => write!(f, "Missing file error"),
            RunnerError::Shader(error) => write!(f, "Shader error: {}", error),
            RunnerError::Video(err) => write!(f, "{}", err),
            RunnerError::Image(msg) => write!(f, "Image error: {}", msg),
        }
    }
}

#[derive(Clone, Debug)]
pub enum RunnerMessage {
    MouseDown,
    MouseUp,
    MouseMove { x: f64, y: f64, w: f64, h: f64 },
    WatchedFileChanged,
    WatchedFileDeleted,
    UnloadImage { var: String },
    LoadImage { var: String, path: PathBuf },
    ValidationError(String),
    RenderFinished,
    Resized { width: f32, height: f32 },
    ScreenShot(PathBuf),
    AspectChanged,
    ToggleTweakMenu,
    TogglePause,
    PrintEphemralError { error: String },
}

pub enum AppStatus {
    Ok {
        runner: RenderContext,
    },
    CompilerError {
        old_shader: Option<RenderContext>,
        err_string: String,
    },
}

pub(crate) struct App {
    _watcher: RecommendedWatcher,
    // Runs a letter boxing operation on top of the main render context.
    letter_box: RenderContext,
    // Contains the render context
    status: AppStatus,
    // the target texture for the main context
    output_texture: wgpu::Texture,
    // dirty textures specified by bytes of the next frame and binding location
    // We build the new ctx here before moving it into current_isf_ctx
    temp_isf_ctx: Cell<Option<Result<RenderContext, RunnerError>>>,
    // Videos that are being polled tracked by variable name
    video_streams: BTreeMap<String, VideoLoader>,
    // A list of in memory images to load before rendering,
    texture_jobs: Vec<(String, Vec<u8>, u32, u32)>,
    // If true we will recompile on every file modified event that changes the md5.
    // set to true when there has been a file event, set to low when the compiled shader is
    // moved into the temp_isf_ctx
    recompile_scheduled: bool,
    // We received a validation error, rendering will produce more
    // Hit the breaks on out pipeline.
    pipeline_invalid: bool,
    // The render targets have been reconfigured by the user
    must_update_render_targets: bool,
    // Screen output format
    shader_path: PathBuf,
    frame_ct: u32,
    local: chrono::DateTime<Local>,
    start_time: std::time::Instant,
    last_frame: std::time::Instant,
    gui_context: GuiContext,
    ui_state: UiState,
    // You may be asking "why do we have both ends of an mpsc channel?"
    // sometimes we spawn a thread and wait for a file to return
    messages: std::sync::mpsc::Receiver<RunnerMessage>,
    message_sender: std::sync::mpsc::Sender<RunnerMessage>,
}

impl App {
    pub fn init(
        shader_source: &str,
        shader_path: &std::path::Path,
        wgpu_device: &wgpu::Device,
        wgpu_queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        gui_context: GuiContext,
        watcher: RecommendedWatcher,
    ) -> Result<App, Error> {
        let (message_sender, messages) = std::sync::mpsc::channel();

        let mut letter_box = RenderContext::new(
            include_str!("../resources/letterbox.glsl"),
            output_format,
            wgpu_device,
            wgpu_queue,
        )
        .unwrap();

        wgpu_device.push_error_scope(wgpu::ErrorFilter::Validation);
        let ctx = RenderContext::new(
            shader_source,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu_device,
            wgpu_queue,
        );
        let err_fut = wgpu_device.pop_error_scope();
        let err = pollster::block_on(err_fut);

        let status = match (ctx, err) {
            (Ok(runner), None) => AppStatus::Ok { runner },
            (Err(e), _) => {
                let out = match e {
                    Error::ShaderCompilationFailed(s) => s,
                    Error::UniformError(s) => s.to_string(),
                    Error::DocumentParsingFailed(s) => s.to_string(),
                };

                AppStatus::CompilerError {
                    old_shader: Some(RenderContext::error_state(
                        wgpu_device,
                        wgpu_queue,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )),
                    err_string: out,
                }
            }
            (_, Some(e)) => AppStatus::CompilerError {
                old_shader: Some(RenderContext::error_state(
                    wgpu_device,
                    wgpu_queue,
                    wgpu::TextureFormat::Rgba8Unorm,
                )),
                err_string: format!("{e}"),
            },
        };

        let ui_state = UiState::new();

        let [width, height] = ui_state.options.lock_aspect_ratio.unwrap();

        let output_texture = wgpu_device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // crunch crunch
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        letter_box.load_shared_texture(&output_texture, "image");

        let frame_ct = 0;
        let start_time = std::time::Instant::now();
        let last_frame = std::time::Instant::now();
        Ok(Self {
            texture_jobs: vec![],
            must_update_render_targets: false,
            output_texture: output_texture.into(),
            letter_box,
            messages,
            video_streams: BTreeMap::new(),
            message_sender,
            _watcher: watcher,
            temp_isf_ctx: Cell::new(None),
            status,
            frame_ct,
            recompile_scheduled: false,
            pipeline_invalid: false,
            local: Local::now(),
            start_time,
            last_frame,
            shader_path: shader_path.to_owned(),
            gui_context,
            ui_state,
        })
    }

    fn update_stream_textures(&mut self, wgpu_device: &wgpu::Device, wgpu_queue: &wgpu::Queue) {
        let video_job_vec: Vec<_> = self
            .video_streams
            .iter_mut()
            .map(|(name, loader)| {
                let buf = loader.present();
                (buf, name.clone(), loader.width(), loader.height())
            })
            .collect();

        for (buf, name, width, height) in video_job_vec {
            if let Some(buf) = buf {
                let desc = tweak_shader::TextureDesc {
                    width,
                    height,
                    stride: None,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    data: &buf.lock().unwrap(),
                };
                self.current_shader_mut()
                    .load_texture(name, desc, wgpu_device, wgpu_queue);
            }
        }
    }

    fn update_letterbox(
        &mut self,
        width: u32,
        height: u32,
        win_width: u32,
        win_height: u32,
    ) -> Option<()> {
        self.letter_box
            .get_input_mut("output_height")?
            .as_float()?
            .current = win_height as f32;

        self.letter_box
            .get_input_mut("output_width")?
            .as_float()?
            .current = win_width as f32;

        self.letter_box
            .get_input_mut("aspect_ratio")?
            .as_float()?
            .current = width as f32 / height as f32;

        Some(())
    }

    pub fn render(
        &mut self,
        wgpu_device: &wgpu::Device,
        wgpu_queue: &wgpu::Queue,
        screen_tex: &wgpu::Texture,
        window: &egui_winit::winit::window::Window,
    ) {
        let mut wgpu_encoder =
            wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // This is here because it's
        // a good place to catch the wgpu device
        if self.recompile_scheduled {
            self.recompile_shader(wgpu_device, wgpu_queue);
        }

        let size = window.inner_size();

        while let Some((name, rgba_8_data, width, height)) = self.texture_jobs.pop() {
            let desc = tweak_shader::TextureDesc {
                data: &rgba_8_data,
                width,
                height,
                stride: None,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
            };

            self.current_shader_mut()
                .load_texture(name, desc, wgpu_device, wgpu_queue)
        }

        if self.must_update_render_targets {
            self.update_render_targets(wgpu_device, size.width, size.height);
            self.must_update_render_targets = false;
        }

        if !self.pipeline_invalid {
            self.update_stream_textures(wgpu_device, wgpu_queue);

            let h = self.output_texture.height();
            let w = self.output_texture.width();

            self.update_letterbox(w, h, size.width, size.height)
                .unwrap();

            self.current_shader_mut()
                .update_resolution([w as f32, h as f32]);

            let view = self.output_texture.create_view(&Default::default());
            self.current_shader_mut().render(
                wgpu_queue,
                wgpu_device,
                &mut wgpu_encoder,
                // this texture was shared with `letter_box`  in init
                view,
                w,
                h,
            );

            self.letter_box.render(
                wgpu_queue,
                wgpu_device,
                &mut wgpu_encoder,
                screen_tex.create_view(&Default::default()),
                size.width,
                size.height,
            );

            if let Some(path) = self.ui_state.screen_shot_scheduled.as_ref().cloned() {
                let (mut vec, w, h) = if self.ui_state.options.use_screen_size_for_screenshots {
                    let vec = self.letter_box.render_to_vec(
                        wgpu_queue,
                        wgpu_device,
                        size.width,
                        size.height,
                    );
                    (vec, size.width, size.height)
                } else {
                    let vec =
                        self.current_shader_mut()
                            .render_to_vec(wgpu_queue, wgpu_device, w, h);
                    (vec, w, h)
                };

                for chunk in vec.chunks_exact_mut(4) {
                    // Swap the red and blue channels
                    chunk.swap(0, 2);
                }

                let dynamic_image =
                    image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, vec).unwrap());

                dynamic_image.save(path).unwrap();
                self.ui_state.screen_shot_scheduled = None;
            }
        }

        // Below this comment is all UI logic
        // as a note to people thinking about integrating the winit runner:
        // Any toy runner using winit is responsible for
        // maintaining it's own secondary render passes
        let raw_input = self.gui_context.egui_state.take_egui_input(window);

        let status = &mut self.status;
        let ui_state = &mut self.ui_state;

        let output = self.gui_context.egui_context.run(raw_input, |ctx| {
            ui::toasts(ui_state, ctx);
            match status {
                AppStatus::Ok { runner } => {
                    crate::ui::side_panel(runner, &self.message_sender, ui_state, ctx);
                }
                AppStatus::CompilerError { err_string, .. } => {
                    crate::ui::diagnostic_message(ctx, err_string)
                }
            }
        });

        for (id, delta) in &output.textures_delta.set {
            self.gui_context
                .egui_renderer
                .update_texture(wgpu_device, wgpu_queue, *id, delta);
        }

        for id in &output.textures_delta.free {
            self.gui_context.egui_renderer.free_texture(id);
        }

        self.gui_context
            .egui_state
            .handle_platform_output(window, output.platform_output);

        let prims = self
            .gui_context
            .egui_context
            .tessellate(output.shapes, window.scale_factor() as f32);

        self.gui_context.egui_renderer.update_buffers(
            wgpu_device,
            wgpu_queue,
            &mut wgpu_encoder,
            &prims,
            &self.gui_context.egui_screen_desc,
        );

        {
            let view = &screen_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let mut render_pass = wgpu_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Gui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.gui_context.egui_renderer.render(
                &mut render_pass,
                &prims,
                &self.gui_context.egui_screen_desc,
            );
        }

        wgpu_queue.submit(Some(wgpu_encoder.finish()));
    }

    pub fn update_render_targets(
        &mut self,
        device: &wgpu::Device,
        out_width: u32,
        out_height: u32,
    ) {
        self.current_shader_mut()
            .update_resolution([out_width as f32, out_height as f32]);

        let [width, height] = self
            .ui_state
            .options
            .lock_aspect_ratio
            .unwrap_or([out_width, out_height]);

        if self.output_texture.width() != width || self.output_texture.height() != height {
            let output_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1, // crunch crunch
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            self.letter_box
                .load_shared_texture(&output_texture, "image");

            self.output_texture = output_texture.into();
        }
    }

    fn update_frame_timing(&mut self) {
        let time_delta = self.last_frame.elapsed();
        self.last_frame = std::time::Instant::now();

        if self.ui_state.options.paused {
            self.start_time += time_delta;
            return;
        }

        let year = self.local.year() as f32;
        let month = self.local.month() as f32;
        let day = self.local.day() as f32;
        let seconds_since_midnight =
            (self.local.hour() * 3600 + self.local.minute() * 60 + self.local.second()) as f32;

        self.frame_ct += 1;
        let frame_ct = self.frame_ct;
        let time = self.start_time.elapsed().as_secs_f32();
        let ctx = self.current_shader_mut();

        ctx.update_datetime([year, month, day, seconds_since_midnight]);
        ctx.update_frame_count(frame_ct);
        ctx.update_time(time);
        ctx.update_delta(time_delta.as_secs_f32());
    }

    fn current_shader_mut(&mut self) -> &mut RenderContext {
        match self.status {
            AppStatus::Ok { ref mut runner } => runner,
            AppStatus::CompilerError {
                ref mut old_shader, ..
            } => old_shader.as_mut().unwrap(),
        }
    }

    fn recompile_shader(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Ok(source) = read_file(&self.shader_path) else {
            self.temp_isf_ctx.set(Some(Err(RunnerError::MissingFile)));
            return;
        };

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let temp_context =
            RenderContext::new(source, wgpu::TextureFormat::Rgba8Unorm, device, queue)
                .map_err(RunnerError::Shader);
        let err_fut = device.pop_error_scope();
        let err = pollster::block_on(err_fut);

        if let Some(e) = err {
            self.temp_isf_ctx
                .set(Some(Err(RunnerError::Validation(format!("{e}")))));
        } else {
            self.temp_isf_ctx.set(Some(temp_context));
        }
    }

    pub fn update_gui(
        &mut self,
        event: &egui_winit::winit::event::WindowEvent,
        window: &egui_winit::winit::window::Window,
    ) {
        let _ = self.gui_context.egui_state.on_window_event(window, event);
        match event {
            egui_winit::winit::event::WindowEvent::Resized(size) => {
                self.gui_context.egui_screen_desc.size_in_pixels = [size.width, size.height];
                self.gui_context.egui_painter.on_window_resized(
                    self.gui_context.egui_context.viewport_id(),
                    size.width.try_into().unwrap(),
                    size.height.try_into().unwrap(),
                );
            }
            egui_winit::winit::event::WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.gui_context.egui_screen_desc.pixels_per_point = *scale_factor as f32;
            }
            _ => {}
        }
    }

    pub fn queue_message(&mut self, message: RunnerMessage) {
        let _ = self.message_sender.send(message);
    }

    pub fn process_messages(&mut self) {
        while let Ok(message) = self.messages.try_recv() {
            self.update(message);
        }
    }

    fn update(&mut self, message: RunnerMessage) {
        match message {
            RunnerMessage::ScreenShot(p) => {
                self.ui_state.screen_shot_scheduled = Some(p);
            }
            RunnerMessage::AspectChanged => {
                self.must_update_render_targets = true;
            }
            RunnerMessage::ToggleTweakMenu => {
                self.ui_state.input_panel_hidden = !self.ui_state.input_panel_hidden
            }
            RunnerMessage::Resized { width, height } => {
                self.current_shader_mut().update_resolution([width, height]);
                if self.ui_state.options.lock_aspect_ratio.is_none() {
                    self.must_update_render_targets = true;
                }
            }
            RunnerMessage::UnloadImage { var } => {
                self.video_streams.remove(&var);
                self.current_shader_mut().remove_texture(&var);
            }
            RunnerMessage::LoadImage { var, path } => {
                match load_image_to_rgba8(&path) {
                    Ok(LoadedImage::Image {
                        rgba8_data,
                        height,
                        width,
                    }) => {
                        self.texture_jobs
                            .push((var.clone(), rgba8_data, width, height));

                        if let Some(file) = path.file_name() {
                            self.ui_state
                                .current_loaded_files
                                .insert(var, file.to_string_lossy().to_string());
                        }
                    }
                    Ok(LoadedImage::Video(loader)) => {
                        self.video_streams.insert(var.clone(), loader);

                        if let Some(file) = path.file_name() {
                            self.ui_state
                                .current_loaded_files
                                .insert(var, file.to_string_lossy().to_string());
                        }
                    }
                    Err(e) => self.queue_message(RunnerMessage::PrintEphemralError {
                        error: format!("{e}"),
                    }),
                };
            }
            RunnerMessage::RenderFinished => {
                self.update_frame_timing();
            }
            RunnerMessage::MouseDown => {
                if !self.gui_context.egui_context.is_pointer_over_area() {
                    self.current_shader_mut().set_mouse_down();
                }
            }
            RunnerMessage::MouseUp => {
                self.current_shader_mut().set_mouse_up();
            }
            RunnerMessage::PrintEphemralError { error } => self.ui_state.notifications.push(error),
            RunnerMessage::MouseMove { x, y, w, h } => {
                if !self.gui_context.egui_context.is_using_pointer() {
                    if let Some([tex_w, tex_h]) = self.ui_state.options.lock_aspect_ratio {
                        let window_w = w as f32;
                        let window_h = h as f32;
                        let h_scale = tex_h as f32 / window_h;
                        let w_scale = tex_w as f32 / window_w;

                        let window_aspect_ratio = w as f32 / h as f32;
                        let letterbox_aspect_ratio = tex_w as f32 / tex_h as f32;

                        if window_aspect_ratio > letterbox_aspect_ratio {
                            let content_width = window_h * letterbox_aspect_ratio;
                            let margin = (window_w - content_width) / 2.0;
                            let mapped =
                                ((x as f32 - margin) / (window_w - margin * 2.0)) * tex_w as f32;
                            self.current_shader_mut()
                                .set_mouse_input([mapped, h_scale * y as f32]);
                        } else {
                            let content_height = window_w / letterbox_aspect_ratio;
                            let margin = (window_h - content_height) / 2.0;
                            let mapped =
                                ((y as f32 - margin) / (window_h - margin * 2.0)) * tex_h as f32;
                            self.current_shader_mut()
                                .set_mouse_input([w_scale * x as f32, mapped]);
                        }
                    } else {
                        self.current_shader_mut()
                            .set_mouse_input([x as f32, y as f32]);
                    }
                }
            }
            RunnerMessage::WatchedFileChanged { .. } => {
                if !self.ui_state.options.halt_recompilation {
                    self.recompile_scheduled = true
                }
            }
            RunnerMessage::TogglePause => {
                self.ui_state.options.paused = !self.ui_state.options.paused;
            }
            RunnerMessage::WatchedFileDeleted { .. } => {
                self.queue_message(RunnerMessage::PrintEphemralError {
                    error: "Dude Wheres my Shader? : Shader file deleted.".to_string(),
                });
            }
            RunnerMessage::ValidationError(e) => {
                self.temp_isf_ctx
                    .set(Some(Err(RunnerError::Validation(e.clone()))));
                self.pipeline_invalid = true;
                self.queue_message(RunnerMessage::PrintEphemralError {
                    error: format!("Validation Error: {e}"),
                });
            }
        }
    }

    pub fn update_pipeline(&mut self, wgpu_device: &wgpu::Device, wgpu_queue: &wgpu::Queue) {
        // if the pipeline has been update the temp pipeline will be some
        // otherwise we don't have any recompiled shader versions to handle
        let Some(temp_ctx) = self.temp_isf_ctx.take() else {
            return;
        };

        self.recompile_scheduled = false;
        let current = self.current_shader_mut();

        match temp_ctx {
            Ok(mut runner) => {
                current.copy_resources_into(&mut runner, wgpu_device, wgpu_queue);
                self.status = AppStatus::Ok { runner }
            }
            Err(e) => {
                let new_err_string = format!("{e}");

                let mut temp = AppStatus::CompilerError {
                    old_shader: None,
                    err_string: String::new(),
                };

                std::mem::swap(&mut self.status, &mut temp);

                match temp {
                    AppStatus::Ok { runner } => {
                        self.status = AppStatus::CompilerError {
                            old_shader: Some(runner),
                            err_string: new_err_string,
                        }
                    }
                    AppStatus::CompilerError { old_shader, .. } => {
                        self.status = AppStatus::CompilerError {
                            old_shader: old_shader.or_else(|| {
                                Some(RenderContext::error_state(
                                    wgpu_device,
                                    wgpu_queue,
                                    wgpu::TextureFormat::Rgba8Unorm,
                                ))
                            }),
                            err_string: new_err_string,
                        }
                    }
                }
            }
        }
    }
}

enum LoadedImage {
    Video(VideoLoader),
    Image {
        height: u32,
        width: u32,
        rgba8_data: Vec<u8>,
    },
}

fn load_image_to_rgba8<P: AsRef<Path>>(file_path: P) -> Result<LoadedImage, RunnerError> {
    let path = file_path.as_ref();

    let supported_video_exensions = &["mov", "webm", "mp4", "avi", "wmv"];

    let is_video = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| supported_video_exensions.contains(&s))
        .unwrap_or(false);

    if is_video {
        let loader = VideoLoader::init(file_path).map_err(RunnerError::Video)?;
        Ok(LoadedImage::Video(loader))
    } else {
        let img = image::io::Reader::open(path)
            .map_err(|_| RunnerError::Image("Image Load Failed".to_owned()))?;

        let dyn_image = img
            .decode()
            .map_err(|_| RunnerError::Image("Decoding failed".into()))?;

        let rgba8 = dyn_image.to_rgba8().into_raw();

        let res = LoadedImage::Image {
            height: dyn_image.height(),
            width: dyn_image.width(),
            rgba8_data: rgba8,
        };

        Ok(res)
    }
}
