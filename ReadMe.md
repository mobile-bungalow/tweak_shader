
## Lib Tweak Shader

[![Documentation](https://docs.rs/tweak_shader/badge.svg)](https://docs.rs/tweak_shader)
[![Crates.io](https://img.shields.io/crates/v/tweak_shader.svg)](https://crates.io/crates/tweak_shader)

 <div style="display: flex; flex-direction: row; justify-content: space-between;">
  <img width=250; src="media/sc1.png" alt="screenshot of use" style="flex: 1; max-width: 100%; height: auto;">
  <img width=250; src="media/sc2.png" alt="screenshot of use" style="flex: 1; max-width: 100%; height: auto;">
  <img width=250; src="media/sc3.png" alt="screenshot of use" style="flex: 1; max-width: 100%; height: auto;">
</div>

### Description

The tweak shader library provides a [wgpu](https://github.com/gfx-rs/wgpu) rendering and bookkeeping context for an interative screen shader format, it currently supports fragment and compute shaders.
It allows users to create shaders reminiscent of ShaderToy or ISF shaders with any number custom uniforms, textures, buffers, and renderpasses that can be tweaked at runtime. This can be used for 
composable post processing effects, generative art, reactive visuals and animation, it is intended for inclusion in other wgpu based creative software.


### Usage

You can try the [web demo here](https://mobile-bungalow.github.io/tweak_shader_web/)

Run any of the examples under `tweak_shader_examples` with

```bash
cargo run -- --file tweak_shader_examples/<file_name>
```

To include the library in your own WGPU based project, simply add it to your cargo.toml and use it like so:

```Rust 
 use tweak_shader::RenderContext;
 use wgpu::TextureFormat;

 let src =  r#"
#version 450

layout(location = 0) out vec4 out_color;

#pragma input(float, name="foo", default=0.0, min=0.0, max=1.0)
#pragma input(float, name="bar")
#pragma input(float, name="baz", default=0.5)
layout(set = 0, binding = 0) uniform Inputs {
    float foo;
    float bar;
    float baz;
};

void main()
{
    out_color = vec4(foo, bar, baz, 1.0);
}
"#;

 let format = TextureFormat::Rgba8UnormSrgb;
 let device = // your wgpu::Device here;
 let queue = // your wgpu::Queue here;

 let render_context = RenderContext::new(src, format, &device, &queue).unwrap();

 let input = render_context.get_input_as<f32>("foo")?;
 input = 0.5;

 // congratulations! you now have a 255x255 pink square.
 let output = render_context.render_to_vec(&queue, &device, 255, 255);

```

See the documentation on creates.io or in the tweak_shader subdirectory for more info.
