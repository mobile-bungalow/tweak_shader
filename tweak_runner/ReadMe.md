# Tweak Shader Runner 

A simple demo application for the tweak shader library. It watches changes to a tweak shader file and displays errors as they arise. A few examples are available under `tweak_shader_examples` in the root directory.

The ui is powered by Egui.

## Usage

```
runner --file <tweak_shader_file>

### Building


```bash
cargo build
```
and 

```bash
cargo run -- file tweak_shader_examples/<demo_file.fs>
```
To run the demo application.

if you want to use audio or video then the runner application requires the ffmpeg library installed due to the [ffmpeg_next](https://github.com/zmwangx/rust-ffmpeg) crate. Please 
see the platform specific build instructions for ffmpeg_next. 

```bash
cargo build --features audio video
```
