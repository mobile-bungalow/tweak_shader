use tweak_shader::RenderContext;

const TEST_RENDER_DIM: u32 = 256;

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

macro_rules! png_pixels {
    ($file_path:literal) => {{
        // Load the PNG data using include_bytes!
        let png_bytes = include_bytes!($file_path);

        // Decode the PNG bytes into an image.
        let decoded_image = image::load_from_memory(png_bytes).expect("Failed to decode PNG image");

        // Convert the decoded image to an ImageBuffer in Rgba format.
        let rgba_image: ImageBuffer<Rgba<u8>, Vec<u8>> = decoded_image.to_rgba8();

        // Convert the ImageBuffer to a vector of pixels (raw pixel values).
        rgba_image.into_raw()
    }};
}

#[test]
fn error_state() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    RenderContext::error_state(&device, &queue, wgpu::TextureFormat::Bgra8UnormSrgb);
}

const BASIC_SRC: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(set = 0, binding = 0) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate; 
    uint frame_index;  
    vec4 mouse;       
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

#pragma input(float, name="foo", default=0.0)
layout(set = 1, binding = 0) uniform Ecco {
    float foo;
};

layout(location = 0) out vec4 out_color; 


void main()
{
    vec2 frag_flip = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
    vec2 st = (frag_flip.xy / resolution.xy);
    out_color = vec4(foo, sin(time), st.x, 1.0);
}
"#;

#[test]
fn basic_frag() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut basic = RenderContext::new(
        BASIC_SRC,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    basic.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let time_0_bytes = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    //write_texture_to_png(&time_0_bytes, "basic.png").unwrap();
    assert!(approximately_equivalent(
        &time_0_bytes,
        &png_pixels!("./resources/basic.png")
    ));

    basic.update_time(1.0);
    let time_1_bytes = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    assert!(approximately_equivalent(
        &time_1_bytes,
        &png_pixels!("./resources/basic_time_1.png")
    ));
}

#[test]
fn basic_frag_target_tex() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut basic = RenderContext::new(
        BASIC_SRC,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let out_tex = device.create_texture(&target_desc(TEST_RENDER_DIM, TEST_RENDER_DIM));

    basic.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);

    let mut enc = device.create_command_encoder(&Default::default());
    basic.render(
        &queue,
        &device,
        &mut enc,
        out_tex.create_view(&Default::default()),
        TEST_RENDER_DIM,
        TEST_RENDER_DIM,
    );
    queue.submit(Some(enc.finish()));

    let mut time_0_bytes = vec![0u8; TEST_RENDER_DIM as usize * TEST_RENDER_DIM as usize * 4_usize];

    read_texture_contents_to_slice(
        &device,
        &queue,
        &out_tex,
        TEST_RENDER_DIM,
        TEST_RENDER_DIM,
        &mut time_0_bytes,
    );

    //write_texture_to_png(&time_0_bytes, "basic.png").unwrap();
    assert!(approximately_equivalent(
        time_0_bytes.as_slice(),
        &png_pixels!("./resources/basic.png")
    ));

    basic.update_time(1.0);

    let mut enc = device.create_command_encoder(&Default::default());
    basic.render(
        &queue,
        &device,
        &mut enc,
        out_tex.create_view(&Default::default()),
        TEST_RENDER_DIM,
        TEST_RENDER_DIM,
    );
    queue.submit(Some(enc.finish()));

    read_texture_contents_to_slice(
        &device,
        &queue,
        &out_tex,
        TEST_RENDER_DIM,
        TEST_RENDER_DIM,
        &mut time_0_bytes,
    );

    assert!(approximately_equivalent(
        &time_0_bytes,
        &png_pixels!("./resources/basic_time_1.png")
    ));
}

// reading and writing to non256 aligned textures should not panic
#[test]
fn misaligned() {
    let (device, queue) = set_up_wgpu();
    let mut basic = RenderContext::new(
        BASIC_SRC,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    basic.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let _time_0_bytes =
        basic.render_to_vec(&queue, &device, TEST_RENDER_DIM + 30, TEST_RENDER_DIM + 30);

    basic.update_time(1.0);
    let _time_1_bytes =
        basic.render_to_vec(&queue, &device, TEST_RENDER_DIM - 30, TEST_RENDER_DIM - 30);
}

const PUSH_CONSTANT_ALIGNMENT_SRC: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)


#pragma input(float, name="time", default=0.0)
#pragma input(float, name="foo", default=0.0)
layout(push_constant) uniform ShaderInputs {
    vec3 mess_up;
    float time;       
    float foo;
    mat4 brogan;
};

layout(location = 0) out vec4 out_color; 


void main()
{
    vec2 frag_flip = vec2(gl_FragCoord.x, 256.0 - gl_FragCoord.y);
    vec2 st = (frag_flip.xy / vec2(256.0));
    out_color = vec4(foo, sin(time), st.x, 1.0);
}
"#;

#[test]
fn push_constant_alignment() {
    let (device, queue) = set_up_wgpu();
    let mut basic = RenderContext::new(
        PUSH_CONSTANT_ALIGNMENT_SRC,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let time_0_bytes = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    //write_texture_to_png(&time_0_bytes, "basic.png").unwrap();
    assert!(approximately_equivalent(
        &time_0_bytes,
        &png_pixels!("./resources/basic.png")
    ));

    basic
        .get_input_mut("time")
        .unwrap()
        .as_float()
        .unwrap()
        .current = 1.0;
    let time_1_bytes = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    //write_texture_to_png(&time_1_bytes, "basic_time_1.png").unwrap();
    assert!(approximately_equivalent(
        &time_1_bytes,
        &png_pixels!("./resources/basic_time_1.png")
    ));
}

const PUSH_CONSTANTS: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate; 
    uint frame_index;  
    vec4 mouse;       
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

#pragma input(float, name="foo", default=0.0)
layout(set = 1, binding = 0) uniform Ecco {
    float foo;
};

layout(location = 0) out vec4 out_color; 


void main()
{
    vec2 frag_flip = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
    vec2 st = (frag_flip.xy / resolution.xy);
    out_color = vec4(foo, sin(time), st.x, 1.0);
}
"#;

#[test]
fn push_constants() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut basic = RenderContext::new(
        PUSH_CONSTANTS,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    basic.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let time_0_bytes = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    //write_texture_to_png(&time_0_bytes, "basic.png").unwrap();
    assert!(approximately_equivalent(
        &time_0_bytes,
        &png_pixels!("./resources/basic.png")
    ));

    basic.update_time(1.0);
    let time_1_bytes = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    //write_texture_to_png(time_1_bytes, "basic_time_1.png");
    assert!(approximately_equivalent(
        &time_1_bytes,
        &png_pixels!("./resources/basic_time_1.png")
    ));
}

const TOO_MANY_PUSH_CONSTANTS: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate; 
    uint frame_index;  
    vec4 mouse;       
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

layout(push_constant) uniform Doofus {
    uint no;
};

layout(location = 0) out vec4 out_color; 

void main()
{
    out_color = vec4(1.0, sin(time), 1.0, 1.0);
}
"#;

#[test]
fn too_many_push_constants() {
    let (device, queue) = set_up_wgpu();
    let basic = RenderContext::new(
        TOO_MANY_PUSH_CONSTANTS,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    );

    assert!(matches!(basic, Err(tweak_shader::Error::UniformError(_))))
}

const PERSISTENT_SRC: &str = r#"
#version 450
#pragma tweak_shader(version="1.0")

#pragma utility_block(ShaderInputs)
layout(set = 0, binding = 0) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate;
    uint frame_index;
    vec4 mouse;    
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

#pragma input(color, name=data, default = [0.95, 0.25, 0, 1])
layout(set = 0, binding = 3) uniform Data {
  vec4 data;
};

#pragma pass(0, persistent, target="dataHistory", height=1)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D dataHistory;

layout(location = 0) out vec4 out_color; 


void main()	{
  ivec2 size = textureSize(sampler2D(dataHistory, default_sampler), 0);
  vec2 fsize = vec2(float(size.x), float(size.y));
	vec2	loc = gl_FragCoord.xy;
	vec4	inputPixelColor = vec4(0.0, 0.0, 0.0, 1.0);
    if (pass_index == 0) {
	    inputPixelColor = texture(sampler2D(dataHistory, default_sampler), vec2(loc.x - 1.0, 0.0) / fsize);
	    if (floor(loc.x) == 0.0)	{
	    	inputPixelColor = data;
	    } 
    } else {
	    vec4 val = texture(sampler2D(dataHistory, default_sampler), loc / fsize);
	    inputPixelColor = val;
  }
	out_color = inputPixelColor;
}
"#;

#[test]
fn persistent_frag() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut persistent = RenderContext::new(
        PERSISTENT_SRC,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let mut stripe_1 = vec![];

    for _ in 0..4 {
        stripe_1 = persistent.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    }

    //write_texture_to_png(stripe_1.as_slice(), "persistent_1.png").unwrap();
    assert!(approximately_equivalent(
        &stripe_1,
        &png_pixels!("./resources/persistent_1.png")
    ));

    let mut stripe_2 = vec![];

    let mut data = persistent.get_input_mut("data").unwrap();
    let color = data.as_color().unwrap();
    color.current = [0.0, 1.0, 0.0, 1.0];
    for _ in 0..4 {
        stripe_2 = persistent.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    }

    //write_texture_to_png(stripe_2, "persistent_2.png").unwrap();
    assert!(approximately_equivalent(
        &stripe_2,
        &png_pixels!("./resources/persistent_2.png")
    ));

    let mut stripe_3 = vec![];

    let mut data = persistent.get_input_mut("data").unwrap();
    let color = data.as_color().unwrap();
    color.current = [0.0, 0.0, 1.0, 1.0];
    for _ in 0..4 {
        stripe_3 = persistent.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    }

    //write_texture_to_png(stripe_3, "persistent_3.png").unwrap();
    assert!(stripe_3 == png_pixels!("./resources/persistent_3.png"));
}

const SHRIMPLE_TEXTURE_LOAD: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(set = 1, binding = 0) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate; 
    uint frame_index;  
    vec4 mouse;       
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

#pragma input(image, name="input_image")
layout(set=1, binding=1) uniform sampler default_sampler;
layout(set=1, binding=2) uniform texture2D input_image;

layout(location = 0) out vec4 out_color; 

void main() {
	vec2 uv = gl_FragCoord.xy / resolution.xy;
	out_color = texture(sampler2D(input_image, default_sampler), uv);
}
"#;

#[test]
// test a texture identity shader
fn shrimple_texture_direct() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let shrimple_bytes = png_pixels!("./resources/shrimple_tex.png");
    tx_load.load_texture(
        "input_image",
        tweak_shader::TextureDesc {
            width: TEST_RENDER_DIM,
            height: TEST_RENDER_DIM,
            stride: None,
            data: &shrimple_bytes,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        },
        &device,
        &queue,
    );
    tx_load.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);

    let output = tx_load.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    let mut tx_load_2 = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    tx_load.copy_resources_into(&mut tx_load_2, &device, &queue);
    tx_load_2.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let output_2 = tx_load_2.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    assert!(approximately_equivalent(&shrimple_bytes, &output));
    assert!(approximately_equivalent(&shrimple_bytes, &output_2));
}

#[test]
// test a texture identity shader
fn shrimple_texture_load_view() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let shrimple_bytes = png_pixels!("./resources/shrimple_tex.png");
    let tex = device.create_texture_with_data(
        &queue,
        &target_desc(TEST_RENDER_DIM, TEST_RENDER_DIM),
        Default::default(),
        &shrimple_bytes,
    );

    tx_load.load_texture_as_view(&tex, "input_image");

    tx_load.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let desc = tweak_shader::TextureDesc {
        width: TEST_RENDER_DIM,
        height: TEST_RENDER_DIM,
        data: &shrimple_bytes,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        stride: None,
    };
    tx_load.load_texture("input_image", desc, &device, &queue);

    let output = tx_load.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    assert!(approximately_equivalent(&shrimple_bytes, &output));
}

#[test]
// test a texture identity shader
fn shrimple_texture_load() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let shrimple_bytes = png_pixels!("./resources/shrimple_tex.png");

    tx_load.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);

    let desc = tweak_shader::TextureDesc {
        width: TEST_RENDER_DIM,
        height: TEST_RENDER_DIM,
        data: &shrimple_bytes,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        stride: None,
    };
    tx_load.load_texture("input_image", desc, &device, &queue);

    let output = tx_load.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    assert!(approximately_equivalent(&shrimple_bytes, &output));
}

#[test]
// test a texture identity shader
fn float_texture_load() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba32Float,
        &device,
        &queue,
    )
    .unwrap();

    let shrimple_bytes = png_pixels!("./resources/shrimple_tex.png");

    tx_load.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let desc = tweak_shader::TextureDesc {
        width: TEST_RENDER_DIM,
        height: TEST_RENDER_DIM,
        data: &shrimple_bytes,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        stride: None,
    };
    tx_load.load_texture("input_image", desc, &device, &queue);

    let _ = tx_load.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
}

#[test]
fn unaligned_texture() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    // know from file.
    let width = 500;
    let height = 350;

    let zac_bytes = png_pixels!("./resources/zac.png");

    tx_load.update_resolution([width as f32, height as f32]);
    let desc = tweak_shader::TextureDesc {
        width,
        height,
        data: &zac_bytes,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        stride: None,
    };
    tx_load.load_texture("input_image", desc, &device, &queue);

    let output = tx_load.render_to_vec(&queue, &device, width, height);

    assert!(approximately_equivalent(&zac_bytes, &output));
}

#[test]
fn unaligned_texture_from_slice() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    // know from file.
    let width = 500;
    let height = 350;

    let zac_bytes = png_pixels!("./resources/zac.png");

    tx_load.update_resolution([width as f32, height as f32]);

    let desc = tweak_shader::TextureDesc {
        width,
        height,
        data: &zac_bytes,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        stride: None,
    };
    tx_load.load_texture("input_image", desc, &device, &queue);

    let mut vec = vec![0u8; (width * height * 4) as usize];
    tx_load.render_to_slice(
        &queue,
        &device,
        width,
        height,
        vec.as_mut_slice(),
        Some(500 * 4),
    );

    assert!(approximately_equivalent(&zac_bytes, vec.as_slice()));
}

const INPUTS_ITER: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)

layout(location = 0) out vec4 out_color; 

#pragma input(color, name=topColor, default=[0.0, 0.0, 0.0, 1.0])
#pragma input(color, name=bottomColor, default=[0.0, 0.5, 0.8999, 1.0])
#pragma input(color, name=strokeColor, default=[0.25, 0.25, 0.25, 1.0])
#pragma input(float, name=minRange, min=0.0, max=1.0, default=0.1)
#pragma input(float, name=maxRange, min=0.0, max=1.0, default=0.50)
#pragma input(float, name=gain, min=0.0, max=1.0, default=0.13)
#pragma input(float, name=strokeSize, min=0.0, max=0.25, default=0.050)

layout(set=0, binding=0) uniform CustomInput {
  vec4  topColor;
  vec4  bottomColor;
  vec4  strokeColor;
  float minRange;
  float maxRange;
  float strokeSize;
  float gain;
};

#pragma input(color, name=pushtopColor, default=[0.0, 0.0, 0.0, 1.0])
#pragma input(color, name=pushbottomColor, default=[0.0, 0.5, 0.8999, 1.0])
#pragma input(color, name=pushstrokeColor, default=[0.25, 0.25, 0.25, 1.0])
#pragma input(float, name=pushminRange, min=0.0, max=1.0, default=0.1)
#pragma input(float, name=pushmaxRange, min=0.0, max=1.0, default=0.50)
#pragma input(float, name=pushgain, min=0.0, max=1.0, default=0.13)
#pragma input(float, name=pushstrokeSize, min=0.0, max=0.25, default=0.050)
layout(push_constant) uniform PushCustomInput {
  vec4  pushtopColor;
  vec4  pushbottomColor;
  vec4  pushstrokeColor;
  float pushminRange;
  float pushmaxRange;
  float pushstrokeSize;
  float pushgain;
};

layout(set=0, binding=1) uniform sampler default_sampler;

void main() {}
"#;

#[test]
fn inputs_iter() {
    let (device, queue) = set_up_wgpu();
    // this will panic if the pipeline can't be set up.
    let mut inputs_test = RenderContext::new(
        INPUTS_ITER,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    inputs_test.get_input_as::<[f32; 4]>("topColor").unwrap();
    inputs_test.get_input_as::<[f32; 4]>("bottomColor").unwrap();

    inputs_test
        .get_input_mut("strokeColor")
        .unwrap()
        .as_color()
        .unwrap();

    assert!(inputs_test
        .get_input_mut("minRange")
        .unwrap()
        .as_color()
        .is_none());

    let names = [
        "topColor",
        "bottomColor",
        "strokeColor",
        "minRange",
        "maxRange",
        "gain",
        "pushtopColor",
        "pushbottomColor",
        "pushstrokeColor",
        "pushminRange",
        "pushmaxRange",
        "pushgain",
    ];

    let refs = inputs_test.iter_inputs_mut().collect::<Vec<_>>();

    for name in names {
        assert!(refs.iter().any(|(s, _)| name == *s))
    }

    let im_refs = inputs_test.iter_inputs().collect::<Vec<_>>();

    for name in names {
        assert!(im_refs.iter().any(|(s, _)| name == *s))
    }
}

const COMPUTE_TARGETS: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)
#pragma stage(compute)

layout(set=0, binding=1) uniform sampler default_sampler;

#pragma target(name=output_image)
layout(rgba8, set=0, binding=2) uniform writeonly image2D output_image;

#pragma target(name=scratch_buffer, persistent, width=20)
layout(rgba8, set=0, binding=3) uniform writeonly image2D scratch_buffer;


layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
}
"#;

#[test]
fn compute_targets() {
    let (device, queue) = set_up_wgpu();

    let inputs_test = RenderContext::new(
        COMPUTE_TARGETS,
        wgpu::TextureFormat::Rgba8Unorm,
        &device,
        &queue,
    )
    .unwrap();

    assert!(inputs_test.is_compute());

    let inputs_test = RenderContext::new(
        COMPUTE_TARGETS,
        wgpu::TextureFormat::Rgba16Float,
        &device,
        &queue,
    );

    assert!(inputs_test.is_err());
}

const COMPUTE_BASIC: &str = r#"
#version 450
#pragma tweak_shader(version="1.0")
#pragma stage(compute)

#pragma input(float, name=blue, default=0.0, min=0.0, max=1.0)
layout(set=1, binding=0) uniform custom_inputs {
    float blue;
};

#pragma target(name="output_image")
layout(rgba8, set=0, binding=1) uniform writeonly image2D output_image;

#pragma target(name="screen_two")
layout(rgba8, set=0, binding=2) uniform writeonly image2D screen_two;

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(output_image);
    
    vec2 normalized_coords = vec2(pixel_coords) / vec2(image_size);
    
    vec4 color = vec4(normalized_coords.x, normalized_coords.y, blue, 1.0);
    
    vec4 inverse = vec4(vec3(1.0) - color.xyz, 1.0);
    imageStore(output_image, pixel_coords, color);
    imageStore(screen_two, pixel_coords, inverse);
}
"#;

#[test]
fn compute_basic() {
    let (device, queue) = set_up_wgpu();

    let mut basic = RenderContext::new(
        COMPUTE_BASIC,
        wgpu::TextureFormat::Rgba8Unorm,
        &device,
        &queue,
    )
    .unwrap();

    let grad_1 = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    basic.set_compute_target("screen_two").unwrap();

    let grad_2 = basic.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    assert!(approximately_equivalent(
        &grad_1,
        &png_pixels!("./resources/basic_compute.png")
    ));

    assert!(approximately_equivalent(
        &grad_2,
        &png_pixels!("./resources/basic_compute_grad_2.png")
    ));
}

const COMPUTE_MULTIPASS: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)
#pragma stage(compute)

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate; 
    uint frame_index;  
    vec4 mouse;       
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

#pragma target(name="output_image", screen)
layout(rgba8, set=0, binding=1) uniform writeonly image2D output_image;

#pragma pass(0)
#pragma relay(name="relay", target="relay_target")
layout(rgba8, set=0, binding=2) uniform writeonly image2D relay;
layout(set=0, binding=3) uniform texture2D relay_target;

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(output_image);
    
    vec2 normalized_coords = vec2(pixel_coords) / vec2(image_size);
    
    vec4 color = vec4(normalized_coords.x, normalized_coords.y, 1.0, 1.0);
    vec4 mask = texelFetch(relay_target, pixel_coords, 0);
    float center_circle = step(length(normalized_coords - vec2(0.5)), 0.2);

    if (pass_index == 0) {
        imageStore(relay, pixel_coords, vec4(center_circle));
    } else {
        imageStore(output_image, pixel_coords, mask * color);
    }
}
"#;

#[test]
fn compute_multipass() {
    let (device, queue) = set_up_wgpu();

    let mut multipass = RenderContext::new(
        COMPUTE_MULTIPASS,
        wgpu::TextureFormat::Rgba8Unorm,
        &device,
        &queue,
    )
    .unwrap();

    let img = multipass.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);

    assert!(approximately_equivalent(
        &img,
        &png_pixels!("./resources/multipass_compute.png")
    ));
}

const NO_EXCESS: &str = r#"
#version 450

#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(set = 0, binding = 0) uniform ShaderInputs {
    float time;       
    float time_delta; 
    float frame_rate; 
    uint frame_index;  
    vec4 mouse;       
    vec4 date;        
    vec3 resolution;  
    uint pass_index;   
};

#pragma input(float, name="foo", default=0.0)
layout(set = 1, binding = 0) uniform Ecco {
    // unmapped
    float bar;
    float bar;
    float bar;
    float bar;
    float foo;
};

layout(location = 0) out vec4 out_color; 


void main()
{
    vec2 frag_flip = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
    vec2 st = (frag_flip.xy / resolution.xy);
    out_color = vec4(foo, sin(time), st.x, 1.0);
}
"#;

#[test]
fn unmapped_bindings() {
    let (device, queue) = set_up_wgpu();

    let mut unmapped = RenderContext::new(
        NO_EXCESS,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    assert_eq!(1, unmapped.iter_inputs().count());
    assert!(unmapped.get_input("nope").is_none());
    assert!(matches!(
        unmapped.get_input("bar").unwrap(),
        tweak_shader::input_type::InputType::RawBytes(_)
    ));

    assert_eq!(1, unmapped.iter_inputs_mut().count());
    assert!(unmapped.get_input_mut("nope").is_none());
    assert!(matches!(
        unmapped.get_input_mut("bar").unwrap().variant(),
        tweak_shader::input_type::InputVariant::Bytes
    ));
}

const LETTERBOX: &str = "
#version 450

#pragma input(image, name=image)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D image;
layout(location = 0) out vec4 out_color; 

void main() {
	vec2 uv = gl_FragCoord.xy / vec2(256.0, 256.0);
    float targetAspectRatio = 16.0 / 9.0;

    // Letterbox on the top and bottom.
    float scaledHeight = 1.0 / targetAspectRatio;
    vec2 new_uv = vec2(uv.x, (uv.y - (1.0 - scaledHeight) / 2.0) / scaledHeight);
    out_color = texture(sampler2D(image, default_sampler), new_uv);

    bool in_bounds = new_uv.x >= 0.0 && new_uv.x <= 1.0 && new_uv.y >= 0.0 && new_uv.y <= 1.0;

    if (!in_bounds) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0); // Color the out-of-bounds pixels black
    }

}
";

#[test]
fn letterboxed_shrimple_texture_load() {
    let (device, queue) = set_up_wgpu();

    let mut tx_load = RenderContext::new(
        SHRIMPLE_TEXTURE_LOAD,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let mut letterbox = RenderContext::new(
        LETTERBOX,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &device,
        &queue,
    )
    .unwrap();

    let shrimple_bytes = png_pixels!("./resources/shrimple_tex.png");

    tx_load.update_resolution([TEST_RENDER_DIM as f32, TEST_RENDER_DIM as f32]);
    let desc = tweak_shader::TextureDesc {
        width: TEST_RENDER_DIM,
        height: TEST_RENDER_DIM,
        data: &shrimple_bytes,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        stride: None,
    };
    tx_load.load_texture("input_image", desc, &device, &queue);

    // create 256 x 256 rgba texture, load into letter box, render to it with shrimp
    // render output again with letterbox
    let shared_tex = device.create_texture(&target_desc(TEST_RENDER_DIM, TEST_RENDER_DIM));

    let mut desc = DEFAULT_VIEW;
    desc.format = Some(shared_tex.format());

    assert!(letterbox.load_texture_as_view(&shared_tex, "image"));

    assert!(!letterbox.load_texture_as_view(&shared_tex, "shrimp"));

    let mut enc = device.create_command_encoder(&Default::default());
    tx_load.render(
        &queue,
        &device,
        &mut enc,
        shared_tex.create_view(&Default::default()),
        TEST_RENDER_DIM,
        TEST_RENDER_DIM,
    );
    queue.submit(Some(enc.finish()));

    let out = letterbox.render_to_vec(&queue, &device, TEST_RENDER_DIM, TEST_RENDER_DIM);
    //write_texture_to_png(out.as_slice(), "letterbox.png").unwrap();
    let png_bytes = png_pixels!("./resources/letterbox.png");
    assert!(approximately_equivalent(&out, &png_bytes));
}

fn set_up_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = if cfg!(windows) {
        let desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        };

        wgpu::Instance::new(desc)
    } else {
        wgpu::Instance::default()
    };

    let adapter = pollster::block_on(async {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter")
    });
    let mut required_limits = wgpu::Limits::default().using_resolution(adapter.limits());
    required_limits.max_push_constant_size = 128;

    let (d, q) = pollster::block_on(async {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    memory_hints: wgpu::MemoryHints::Performance,
                    label: None,
                    required_features: wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::CLEAR_TEXTURE,
                    required_limits,
                },
                None,
            )
            .await
            .expect("Failed to create device")
    });

    d.on_uncaptured_error(Box::new(|e| match e {
        wgpu::Error::Internal {
            source,
            description,
        } => {
            panic!("internal error: {source}, {description}");
        }
        wgpu::Error::OutOfMemory { .. } => {
            panic!("Out Of GPU Memory! bailing");
        }
        wgpu::Error::Validation {
            description,
            source,
        } => {
            panic!("{description} : {source}");
        }
    }));
    (d, q)
}
use image::{ImageBuffer, ImageFormat, Rgba};
use wgpu::util::DeviceExt;

fn approximately_equivalent(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .map(|(a, b)| if a < b { b - a } else { a - b })
            .enumerate()
            .all(|(idx, abs_diff)| {
                let pixel = idx / 4;
                if abs_diff > 3 {
                    panic!("images differ at pixel {pixel}")
                } else {
                    true
                }
            })
}

fn target_desc(width: u32, height: u32) -> wgpu::TextureDescriptor<'static> {
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
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }
}

#[allow(dead_code)]
// use this function when adding validation tests
fn write_texture_to_png(
    data: &[u8],
    file_path: &str,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let texture: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_vec(width, height, data.to_owned())
        .ok_or("Failed to create ImageBuffer")?;

    // Write the texture to a PNG file.
    texture.save_with_format(file_path, ImageFormat::Png)?;
    Ok(())
}

fn read_texture_contents_to_slice(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    height: u32,
    width: u32,
    slice: &mut [u8],
) {
    let block_size = texture
        .format()
        .block_copy_size(Some(wgpu::TextureAspect::All))
        .expect("It seems like you are trying to render to a Depth Stencil. Stop that.");

    let row_byte_ct = block_size * width;
    let padded_row_byte_ct = (row_byte_ct + 255) & !255;

    // Create a buffer to store the texture data
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Texture Read Buffer"),
        size: (height * padded_row_byte_ct) as u64,
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
                bytes_per_row: Some(padded_row_byte_ct),
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

    {
        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| r.unwrap());
        device.poll(wgpu::Maintain::Wait);

        let gpu_slice = buffer_slice.get_mapped_range();
        let gpu_chunks = gpu_slice.chunks(padded_row_byte_ct as usize);
        let slice_chunks = slice.chunks_mut(row_byte_ct as usize);
        let iter = slice_chunks.zip(gpu_chunks);

        for (output_chunk, gpu_chunk) in iter {
            output_chunk.copy_from_slice(&gpu_chunk[..row_byte_ct as usize]);
        }
    };

    buffer.unmap();
}
