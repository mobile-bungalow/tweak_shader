use egui_wgpu::{
    renderer::ScreenDescriptor,
    wgpu::{self, CompositeAlphaMode},
};
use egui_winit::{
    egui::{Color32, FontData, FontDefinitions},
    winit::{
        event_loop::{EventLoop, EventLoopBuilder},
        window::Window,
    },
    State,
};
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;

use crate::app::RunnerMessage;

// All the meta data necessary to manage EGUI
pub struct GuiContext {
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_context: egui_winit::egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_painter: egui_wgpu::winit::Painter,
    pub egui_screen_desc: ScreenDescriptor,
}

// Custom error type for initialization errors
#[derive(Debug)]
pub enum InitializationError {
    Window,
    AdapterRequestError,
    DeviceCreationError,
    WatcherSetupError,
    EguiPainterSetupError,
}

impl std::fmt::Display for InitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            InitializationError::Window => {
                write!(f, "Could Not Init Window.")
            }
            InitializationError::AdapterRequestError => {
                write!(f, "Failed to find an appropriate adapter")
            }
            InitializationError::DeviceCreationError => {
                write!(f, "Failed to create device")
            }
            InitializationError::WatcherSetupError => {
                write!(f, "Could not set up file watcher")
            }
            InitializationError::EguiPainterSetupError => {
                write!(f, "Could not set up egui painter")
            }
        }
    }
}

impl std::error::Error for InitializationError {}

pub struct Resources {
    pub event_loop: EventLoop<RunnerMessage>,
    pub window: Window,
    pub file_watcher: RecommendedWatcher,
    pub wgpu_surface: wgpu::Surface,
    pub wgpu_surface_config: wgpu::SurfaceConfiguration,
    pub wgpu_adapter: wgpu::Adapter,
    pub wgpu_device: wgpu::Device,
    pub wgpu_queue: wgpu::Queue,
    pub gui_context: GuiContext,
}

fn create_window(event_loop: &EventLoop<RunnerMessage>) -> Result<Window, InitializationError> {
    egui_winit::winit::window::WindowBuilder::new()
        .with_transparent(true)
        .with_title("Tweak Shader Runner")
        .build(event_loop)
        .map_err(|_| InitializationError::Window)
}

fn request_adapter(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface,
) -> Result<wgpu::Adapter, InitializationError> {
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(surface),
    }))
    .ok_or(InitializationError::AdapterRequestError)
}

fn create_device(
    adapter: &wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), InitializationError> {
    let mut limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    limits.max_push_constant_size = 256;
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::PUSH_CONSTANTS,
            limits,
        },
        None,
    ))
    .map_err(|_| InitializationError::DeviceCreationError)
}

fn configure_surface(
    surface: &wgpu::Surface,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    size: egui_winit::winit::dpi::PhysicalSize<u32>,
) -> wgpu::SurfaceConfiguration {
    let swapchain_capabilities = surface.get_capabilities(adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let alpha_mode = if swapchain_capabilities
        .alpha_modes
        .contains(&CompositeAlphaMode::PostMultiplied)
    {
        CompositeAlphaMode::PostMultiplied
    } else {
        CompositeAlphaMode::Auto
    };

    let wgpu_surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode,
        view_formats: vec![],
    };

    surface.configure(device, &wgpu_surface_config);
    wgpu_surface_config
}

fn setup_file_watcher(
    path: &Path,
    event_loop: &EventLoop<RunnerMessage>,
) -> Result<RecommendedWatcher, InitializationError> {
    let proxy = event_loop.create_proxy();
    let mut watcher =
        notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| match res {
            Ok(event) => match event.kind {
                EventKind::Modify(_) => {
                    let _ = proxy.send_event(RunnerMessage::WatchedFileChanged {
                        path: event.paths.first().unwrap().to_owned(),
                    });
                }
                EventKind::Remove(_) => {
                    let _ = proxy.send_event(RunnerMessage::WatchedFileDeleted {
                        path: event.paths.first().unwrap().to_owned(),
                    });
                }
                _ => {}
            },
            Err(e) => eprintln!("{e}"),
        })
        .map_err(|_| InitializationError::WatcherSetupError)?;

    // Configure the watcher
    if watcher
        .configure(
            Config::default()
                .with_compare_contents(true)
                .with_poll_interval(std::time::Duration::from_secs(4)),
        )
        .is_err()
    {
        return Err(InitializationError::WatcherSetupError);
    }

    if watcher.watch(path, RecursiveMode::NonRecursive).is_err() {
        return Err(InitializationError::WatcherSetupError);
    }

    Ok(watcher)
}

fn setup_egui(
    window: &Window,
    device: &wgpu::Device,
    swapchain_format: &wgpu::TextureFormat,
) -> Result<GuiContext, InitializationError> {
    let egui_context = egui_winit::egui::Context::default();

    let egui_state = State::new(egui_context.viewport_id(), window, None, None);

    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "Roboto".to_owned(),
        FontData::from_static(include_bytes!("../resources/Roboto-Regular.ttf")),
    );
    fonts
        .families
        .entry(egui_winit::egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "Roboto".to_owned());
    fonts
        .families
        .entry(egui_winit::egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "Roboto".to_owned());
    egui_context.set_fonts(fonts);

    egui_context.set_visuals(egui_winit::egui::Visuals {
        override_text_color: Some(Color32::from_gray(240)),
        panel_fill: Color32::from_rgba_unmultiplied(100, 100, 100, 190),
        ..egui_winit::egui::Visuals::dark()
    });

    let mut egui_painter =
        egui_wgpu::winit::Painter::new(egui_wgpu::WgpuConfiguration::default(), 1, None, true);

    pollster::block_on(egui_painter.set_window(egui_context.viewport_id(), Some(window)))
        .map_err(|_| InitializationError::EguiPainterSetupError)?;

    let egui_screen_desc = ScreenDescriptor {
        size_in_pixels: [window.inner_size().width, window.inner_size().height],
        pixels_per_point: window.scale_factor() as f32,
    };

    let egui_renderer = egui_wgpu::Renderer::new(device, *swapchain_format, None, 1);

    Ok(GuiContext {
        egui_renderer,
        egui_context,
        egui_state,
        egui_painter,
        egui_screen_desc,
    })
}

pub fn initialize(path: &Path) -> Result<Resources, InitializationError> {
    let event_loop: EventLoop<RunnerMessage> = EventLoopBuilder::with_user_event().build();
    let window = create_window(&event_loop)?;
    let instance = wgpu::Instance::default();
    let surface =
        unsafe { instance.create_surface(&window) }.map_err(|_| InitializationError::Window)?;

    let adapter = request_adapter(&instance, &surface)?;
    let (device, queue) = create_device(&adapter)?;

    let error_proxy = event_loop.create_proxy();
    device.on_uncaptured_error(Box::new(move |e| match e {
        wgpu::Error::OutOfMemory { .. } => {
            panic!("Out Of GPU Memory! bailing");
        }
        wgpu::Error::Validation { description, .. } => {
            let _ = error_proxy.send_event(RunnerMessage::ValidationError(description.clone()));
        }
    }));

    let size = window.inner_size();
    let wgpu_surface_config = configure_surface(&surface, &adapter, &device, size);

    let file_watcher = setup_file_watcher(path, &event_loop)?;
    let gui_context = setup_egui(&window, &device, &wgpu_surface_config.format)?;

    Ok(Resources {
        event_loop,
        window,
        file_watcher,
        wgpu_surface_config,
        wgpu_surface: surface,
        wgpu_adapter: adapter,
        wgpu_device: device,
        wgpu_queue: queue,
        gui_context,
    })
}
