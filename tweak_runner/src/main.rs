use app::{App, RunnerMessage};
use egui_winit::winit::event::{ElementState, KeyEvent, MouseButton};
use egui_winit::winit::keyboard::{Key, NamedKey};
use egui_winit::winit::{
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
};
use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::process::exit;

mod app;
mod initialization;
mod ui;
mod video;

fn main() {
    let (path, shader_source) = parse_commands();

    let initialization::Resources {
        event_loop,
        window,
        file_watcher,
        wgpu_surface,
        mut wgpu_surface_config,
        wgpu_adapter,
        wgpu_device,
        wgpu_queue,
        gui_context,
        ..
    } = match initialization::initialize(&path) {
        Ok(r) => r,
        Err(error) => {
            eprintln!("Initialization error: {}", error);
            std::process::exit(1);
        }
    };

    let mut app = match App::init(
        &shader_source,
        &path,
        &wgpu_device,
        &wgpu_queue,
        wgpu_surface_config.format,
        gui_context,
        file_watcher,
    ) {
        Ok(app) => app,
        Err(e) => {
            eprintln!("Failed to set up app: {e:?}");
            exit(1)
        }
    };

    let target_frame_duration = std::time::Duration::from_millis(1000 / 60);
    let mut last_frame_time = std::time::Instant::now();
    let mut resized = false;

    event_loop
        .run(move |event, control_flow| {
            let _ = (&wgpu_adapter, &wgpu_surface_config, &app);
            if last_frame_time.elapsed() > target_frame_duration {
                window.request_redraw();
            }
            let cf = ControlFlow::WaitUntil(std::time::Instant::now() + target_frame_duration);
            control_flow.set_control_flow(cf);
            match event {
                // The file watcher and wgpu can push errors
                Event::UserEvent(m) => app.queue_message(m),
                Event::AboutToWait => {
                    app.process_messages();
                    app.update_pipeline(&wgpu_device, &wgpu_queue);
                }
                Event::WindowEvent { event, .. } => {
                    app.update_gui(&event, &window);
                    match event {
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    logical_key: key,
                                    state: ElementState::Released,
                                    ..
                                },
                            ..
                        } => match key {
                            Key::Named(NamedKey::Escape) => {
                                app.queue_message(RunnerMessage::ToggleTweakMenu);
                            }
                            Key::Named(NamedKey::Space) => {
                                app.queue_message(RunnerMessage::TogglePause);
                            }
                            _ => {}
                        },
                        WindowEvent::MouseInput {
                            state: mouse_state,
                            button: MouseButton::Left,
                            ..
                        } => {
                            if mouse_state == ElementState::Pressed {
                                app.queue_message(RunnerMessage::MouseDown);
                            } else {
                                app.queue_message(RunnerMessage::MouseUp);
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            let size = window.inner_size();
                            app.queue_message(RunnerMessage::MouseMove {
                                x: position.x,
                                y: position.y,
                                w: size.width as f64,
                                h: size.height as f64,
                            });
                        }
                        WindowEvent::CloseRequested => {
                            // using control flow to exit causes
                            // panics in winit 0.28
                            process::exit(0);
                        }

                        WindowEvent::Resized(size) => {
                            wgpu_surface_config.width = size.width;
                            wgpu_surface_config.height = size.height;
                            resized = true;
                            app.queue_message(RunnerMessage::Resized {
                                width: wgpu_surface_config.width as f32,
                                height: wgpu_surface_config.height as f32,
                            });
                        }
                        WindowEvent::RedrawRequested => {
                            if resized {
                                wgpu_surface.configure(&wgpu_device, &wgpu_surface_config);
                                resized = false;
                            }
                            let frame = wgpu_surface
                                .get_current_texture()
                                .expect("Failed to acquire next swap chain texture");

                            app.render(&wgpu_device, &wgpu_queue, &frame.texture, &window);
                            app.queue_message(RunnerMessage::RenderFinished);

                            frame.present();
                            last_frame_time = std::time::Instant::now();
                        }
                        _ => {}
                    };
                }
                _ => {}
            }
        })
        .unwrap();
}

fn read_file<P: AsRef<std::path::Path>>(file_path: P) -> io::Result<String> {
    let mut file = File::open(file_path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

use std::path::PathBuf;
use std::process;

fn parse_commands() -> (PathBuf, String) {
    let args: Vec<String> = env::args().collect();

    let mut file_path: Option<String> = None;
    let mut no_fork = false;

    let mut index = 1; // skip exe name
    while index < args.len() {
        match args[index].as_str() {
            "--file" => {
                // Check if the next argument exists
                if let Some(file_arg) = args.get(index + 1) {
                    file_path = Some(file_arg.clone());
                    index += 2;
                } else {
                    eprintln!("Error: Please specify a file with --file <path_to_shader>");
                    process::exit(1);
                }
            }
            "--no-fork" => {
                no_fork = true;
                index += 1;
            }
            "--help" | "-h" => {
                println!(
                    "The Tweak Shader Runner: \n
     flags:\n
        > -h/--help : Show this message.\n
        > --no-fork : Don't disown process from the terminal.\n
        > --file=<file_path>: Specify the shader file to load and run.\n"
                );
                process::exit(0);
            }
            _ => {
                eprintln!("Error: Unknown argument: {}", args[index]);
                process::exit(1);
            }
        }
    }

    let Some(path) = file_path else {
        eprintln!("Error: Please specify a file with --file <path_to_shader>");
        process::exit(1);
    };

    if !std::path::PathBuf::from(&path).exists() {
        eprintln!("{path} does not exist!");
        process::exit(1);
    }

    // ifn't ain't no fork.
    // fork it with no fork.
    // I sure hope this doesn't crash someones computer.
    if !no_fork {
        if let Ok(current_exe) = env::current_exe() {
            let mut command = std::process::Command::new(current_exe);
            command.stdout(std::process::Stdio::null());
            command.stderr(std::process::Stdio::null());

            // Get the command-line arguments, skipping the first argument (the executable name).
            let args: Vec<String> = env::args().collect();
            if args.len() > 1 {
                command.args(&args[1..]); // Skip the first argument and add the rest.
            }

            command.arg("--no-fork");

            match command.spawn() {
                Ok(_) => {
                    exit(0);
                }
                Err(err) => {
                    eprintln!("Error spawning child process: {}", err);
                    exit(1);
                }
            }
        }
    }

    match read_file(&path) {
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
        Ok(res) => (path.into(), res),
    }
}
