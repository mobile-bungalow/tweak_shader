
## Lib Tweak Shader

[![Documentation](https://docs.rs/tweak_shader/badge.svg)](https://docs.rs/tweak_shader)
[![Crates.io](https://img.shields.io/crates/v/tweak_shader.svg)](https://crates.io/crates/tweak_shader)

 <div style="display: flex; flex-direction: row; justify-content: space-between;">
  <img width=250; src="media/sc1.png" alt="screenshot of use" style="flex: 1; max-width: 100%; height: auto;">
  <img width=250; src="media/sc2.png" alt="screenshot of use" style="flex: 1; max-width: 100%; height: auto;">
  <img width=250; src="media/sc3.png" alt="screenshot of use" style="flex: 1; max-width: 100%; height: auto;">
</div>

### Description

The tweak shader library provides a [wgpu](https://github.com/gfx-rs/wgpu) rendering and bookkeeping context for an interative screen shader format.
It allows users to create shaders reminiscent of ShaderToy or ISF shaders with custom uniforms that can be tweaked at runtime. This can be used for 
composable post processing effects, generative art, reactive visuals and animation, it is intended for inclusion in other wgpu based creative software.
This Library and it's features were modeled after [ISF](https://github.com/mrRay/ISF_Spec)

This has some notable differences from other glsl based screen shader environments. 
* It is vulkan-like, UV's and FragCoords increase going down the screen.
* It is built on wgpu and naga. This means it enforced uniformity of control flow and will throw a validation error if you try to do things
like access a texture in a conditional block.

### Usage

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
 let surface = // your surface texture here;

 let render_context = RenderContext::new(src, format, &device, &queue).unwrap();

 let input = render_context.get_input_as<f32>("foo")?;
 input = 0.5;

 let encoder = device.create_command_encoder(&Default::default());

 render_context.render(&queue, &device, &encoder, &surface, 255, 255);

 queue.submit(Some(encoder.finish()));
 // congratulations! you now have a 255x255 pink square.
```

See the documentation on creates.io or in the tweak_shader subdirectory for more info.
