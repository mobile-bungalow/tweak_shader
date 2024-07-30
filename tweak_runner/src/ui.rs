use crate::RunnerMessage;

use tweak_shader::input_type::{InputVariant, MutInputInt};

use egui_plot::{Plot, PlotPoint, PlotPoints, Points};
use egui_winit::egui::Color32;

use egui_winit::egui::RichText;
use egui_winit::egui::ScrollArea;
use egui_winit::egui::Slider;

use egui_winit::egui::Ui;

use std::cmp::Ord;
use std::collections::BTreeMap;
use std::sync::mpsc;

use egui_notify::Toasts;

pub struct UiOptions {
    pub paused: bool,
    pub halt_recompilation: bool,
    pub lock_aspect_ratio: Option<[u32; 2]>,
    pub use_screen_size_for_screenshots: bool,
    pub resize_debounce: Option<std::time::Instant>,
}

impl Default for UiOptions {
    fn default() -> Self {
        Self {
            resize_debounce: None,
            paused: false,
            use_screen_size_for_screenshots: true,
            halt_recompilation: false,
            lock_aspect_ratio: Some([640, 480]),
        }
    }
}

#[derive(Default)]
pub struct UiState {
    pub options: UiOptions,
    pub show_options: bool,
    pub input_panel_hidden: bool,
    pub screen_shot_scheduled: Option<std::path::PathBuf>,
    // map of variable names to file names
    pub current_loaded_files: BTreeMap<String, String>,
    pub notifications: Vec<String>,
    pub toasts: Toasts,
}

impl UiState {
    pub fn new() -> Self {
        Default::default()
    }
}

pub fn side_panel(
    isf_ctx: &mut tweak_shader::RenderContext,
    message_sender: &mpsc::Sender<RunnerMessage>,
    ui_state: &mut UiState,
    ctx: &egui_winit::egui::Context,
) {
    egui_winit::egui::SidePanel::new(egui_winit::egui::panel::Side::Left, "User Inputs")
        .show_animated(ctx, !ui_state.input_panel_hidden, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                // Label and close button.
                ui.vertical_centered_justified(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("User Inputs").size(15.0));
                        if ui.button("Options").clicked() {
                            ui_state.show_options = !ui_state.show_options;
                        };
                        if ui.button("<< [Esc]").clicked() {
                            ui_state.input_panel_hidden = true;
                        }
                    });

                    ui.separator();
                    if ui_state.show_options {
                        option_panel(ui_state, message_sender, ui);
                    }
                });

                let mut inputs = isf_ctx.iter_inputs_mut().collect::<Vec<_>>();

                inputs.sort_by(|(_, a), (_, b)| (a.variant() as u32).cmp(&(b.variant() as u32)));

                let mut last_variant = inputs
                    .first()
                    .map(|(_, val)| val.variant())
                    .unwrap_or(InputVariant::Point);

                for (name, mut val) in inputs {
                    let variant = val.variant();

                    if last_variant != variant {
                        last_variant = variant;
                        ui.separator();
                    };

                    input_widget(name, &mut val, ui_state, message_sender, ui);
                }
            });
        });
}

fn option_panel(ui_state: &mut UiState, message_sender: &mpsc::Sender<RunnerMessage>, ui: &mut Ui) {
    ui.vertical_centered_justified(|ui| {
        ui.horizontal(|ui| {
            if ui
                .radio(ui_state.options.halt_recompilation, "Pause Recompilation")
                .clicked()
            {
                ui_state.options.halt_recompilation = !ui_state.options.halt_recompilation;
            }
            if ui.radio(ui_state.options.paused, "Pause").clicked() {
                ui_state.options.paused = !ui_state.options.paused
            }
        });

        if ui
            .radio(
                ui_state.options.use_screen_size_for_screenshots,
                "Use screen aspect for screenshots",
            )
            .clicked()
        {
            ui_state.options.use_screen_size_for_screenshots =
                !ui_state.options.use_screen_size_for_screenshots
        }

        if ui.button("take screenshot").clicked() {
            launch_screenshot_dialog(message_sender.clone());
        }

        ui.horizontal(|ui| {
            if ui
                .radio(
                    ui_state.options.lock_aspect_ratio.is_some(),
                    "Lock Aspect Ratio",
                )
                .clicked()
            {
                if ui_state.options.lock_aspect_ratio.is_none() {
                    ui_state.options.lock_aspect_ratio = Some([640, 480]);
                } else {
                    ui_state.options.lock_aspect_ratio = None;
                }
                let _ = message_sender.send(RunnerMessage::AspectChanged);
            }

            if let Some([ref mut w, ref mut h]) = ui_state.options.lock_aspect_ratio.as_mut() {
                let (pre_w, pre_h) = (*w, *h);
                ui.add(egui_winit::egui::DragValue::new(w));
                ui.add(egui_winit::egui::DragValue::new(h));

                *w = (*w).max(1);
                *h = (*h).max(1);

                if (pre_w != *w) || (pre_h != *h) {
                    ui_state.options.resize_debounce = Some(std::time::Instant::now());
                }

                if let Some(last_call) = ui_state.options.resize_debounce {
                    if last_call.elapsed() > std::time::Duration::from_millis(250) {
                        ui_state.options.resize_debounce = None;
                        let _ = message_sender.send(RunnerMessage::AspectChanged);
                    }
                }
            } else {
                let mut placeholder = 0.0;
                ui.add(
                    egui_winit::egui::DragValue::new(&mut placeholder)
                        .custom_formatter(|_, _| "—".into()),
                );
                ui.add(
                    egui_winit::egui::DragValue::new(&mut placeholder)
                        .custom_formatter(|_, _| "—".into()),
                );
            }
        });
    });
    ui.separator();
}

pub fn toasts(ui_state: &mut UiState, ctx: &egui_winit::egui::Context) {
    while let Some(notification) = ui_state.notifications.pop() {
        ui_state
            .toasts
            .error(notification)
            .set_duration(Some(std::time::Duration::from_secs(5)));
    }
    ui_state.toasts.show(ctx);
}

fn input_widget(
    name: &str,
    val: &mut tweak_shader::input_type::MutInput,
    ui_state: &mut UiState,
    message_sender: &mpsc::Sender<RunnerMessage>,
    ui: &mut Ui,
) {
    match val.variant() {
        InputVariant::Image => {
            file_selector(ui, val, name, ui_state, message_sender.clone());
        }
        InputVariant::Float => {
            let v = val.as_float().unwrap();
            ui.add(Slider::new(&mut v.current, v.min..=v.max).text(name));
            ui.add_space(10.0);
        }
        InputVariant::Bool => {
            let v = val.as_bool().unwrap();
            if ui.radio(v.current.is_true(), name).clicked() {
                if v.current.is_true() {
                    v.current = tweak_shader::input_type::ShaderBool::False;
                } else {
                    v.current = tweak_shader::input_type::ShaderBool::True;
                }
            }
        }
        InputVariant::Color => {
            let v = val.as_color().unwrap();
            let _ = ui.horizontal(|ui| {
                ui.color_edit_button_rgba_unmultiplied(&mut v.current);
                ui.label(name);
            });
        }
        InputVariant::Int => {
            let MutInputInt { value: v, labels } = val.as_int().unwrap();
            if let Some(list) = labels {
                let current =
                    list.iter()
                        .find_map(|(str, val)| if v.current == *val { Some(str) } else { None });

                egui_winit::egui::ComboBox::from_label(name)
                    .selected_text(current.unwrap())
                    .show_ui(ui, |ui| {
                        for opt in list {
                            ui.selectable_value(&mut v.current, opt.1, &opt.0);
                        }
                    });
            } else {
                ui.add(Slider::new(&mut v.current, v.min..=v.max).text(name));
                ui.add_space(10.0);
            }
        }
        InputVariant::Point => {
            let val = val.as_point().unwrap();
            point_selector(ui, name, val);
        }
        _ => {}
    };
}

fn file_selector(
    ui: &mut Ui,
    val: &mut tweak_shader::input_type::MutInput,
    name: &str,
    ui_state: &mut UiState,
    sender: std::sync::mpsc::Sender<RunnerMessage>,
) {
    let meta = val.texture_status().unwrap();
    ui.horizontal(|ui| {
        ui.label(name);
        if let tweak_shader::input_type::TextureStatus::Loaded { .. } = meta {
            if let Some(path) = ui_state.current_loaded_files.get(name) {
                if path.len() > 20 {
                    let path = format!("...{}", &path[path.len().saturating_sub(20)..]);
                    ui.label(&path);
                } else {
                    ui.label(path);
                };
            } else {
                ui.label("[ERROR]");
            }

            if ui.button("X").clicked() {
                if val.variant() == InputVariant::Image {
                    let _ = sender.send(RunnerMessage::UnloadImage {
                        var: name.to_owned(),
                    });
                }
                ui_state.current_loaded_files.remove(name);
            }
        } else if ui.button("Select File").clicked() && val.variant() == InputVariant::Image {
            launch_image_or_video_dialog(sender, name.to_owned());
        }
    });
}

fn point_selector(
    ui: &mut Ui,
    name: &str,
    input: &mut tweak_shader::input_type::BoundedInput<[f32; 2]>,
) {
    ui.horizontal(|ui| {
        ui.label(name);
        ui.label("X");
        ui.add(
            egui_winit::egui::DragValue::new(&mut input.current[0])
                .range(input.min[0]..=input.max[0]),
        );
        ui.label("Y");
        ui.add(
            egui_winit::egui::DragValue::new(&mut input.current[1])
                .range(input.min[1]..=input.max[1]),
        );
    });

    Plot::new(name)
        .include_x(input.max[0])
        .include_x(input.min[0])
        .include_y(input.max[1])
        .include_y(input.min[1])
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_double_click_reset(false)
        .view_aspect(2.0)
        .show(ui, |plot_ui| {
            if plot_ui.response().clicked()
                || plot_ui.pointer_coordinate_drag_delta() != egui_winit::egui::Vec2::ZERO
            {
                if let Some(p) = plot_ui.pointer_coordinate() {
                    input.current = [
                        p.x.clamp(input.min[0] as f64, input.max[0] as f64) as f32,
                        p.y.clamp(input.min[1] as f64, input.max[1] as f64) as f32,
                    ];
                }
            }
            let point = Points::new(PlotPoints::Owned(vec![PlotPoint::new(
                input.current[0],
                input.current[1],
            )]))
            .radius(5.0);
            plot_ui.points(point);
        });
}

pub fn diagnostic_message(ctx: &egui_winit::egui::Context, e: &str) {
    egui_winit::egui::TopBottomPanel::bottom("")
        .min_height(3.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(15.0);
                ui.label(
                    RichText::new(e)
                        .color(Color32::from_rgb(255, 10, 0))
                        .background_color(Color32::BLACK),
                );
            });
        });
}

fn launch_screenshot_dialog(sender: std::sync::mpsc::Sender<RunnerMessage>) {
    std::thread::spawn(move || {
        let file_path = tinyfiledialogs::save_file_dialog("select screen shot location", "/");

        if let Some(path) = file_path {
            let mut buf: std::path::PathBuf = path.into();
            if buf.extension().is_none() {
                buf.set_extension("png");
            }
            let _ = sender.send(RunnerMessage::ScreenShot(buf));
        }
    });
}

fn launch_image_or_video_dialog(sender: std::sync::mpsc::Sender<RunnerMessage>, var: String) {
    std::thread::spawn(move || {
        let file_path = tinyfiledialogs::open_file_dialog("Load an Image or Video", "/", None);

        if let Some(file_path) = file_path {
            let _ = sender.send(RunnerMessage::LoadImage {
                path: std::path::PathBuf::from(file_path),
                var,
            });
        }
    });
}
