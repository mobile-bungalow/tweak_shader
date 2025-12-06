use tweak_shader::{input_type::InputType, RenderContext};
use wgpu::{ExperimentalFeatures, MemoryHints};

const TEST_NO_INPUTS: &str = r"
#version 450
void main() {

}
";

#[test]
fn test_isf_context_new() {
    let (device, queue) = set_up_wgpu();
    let context_result = RenderContext::new(
        TEST_NO_INPUTS,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        &device,
        &queue,
    );

    context_result.unwrap();
}

const UTIL_BLOCK: &str = r"
#version 450

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

void main() {
    float v = time;
}
";

#[test]
fn test_isf_context_new_with_util() {
    let (device, queue) = set_up_wgpu();
    let context_result = RenderContext::new(
        UTIL_BLOCK,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        &device,
        &queue,
    );

    context_result.unwrap();
}

const MALFORMED_UTIL_BLOCK: &str = r"
// should not match the expected shader layout
#version 450
#pragma utility_block(ShaderInputs)
layout(set = 0, binding = 0) uniform ShaderInputs {
    vec4 wrong;
};

void main() {

}
";

#[test]
fn test_isf_context_malformed_util() {
    let (device, queue) = set_up_wgpu();
    let context_result = RenderContext::new(
        MALFORMED_UTIL_BLOCK,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        &device,
        &queue,
    );

    assert!(context_result.is_err());
}

const MISSING_UTIL_BLOCK: &str = r"
// should not match the expected shader layout
#version 450
#pragma utility_block(ShaderInputs)

void main() {

}
";

#[test]
fn test_missing_util() {
    let (device, queue) = set_up_wgpu();
    let context_result = RenderContext::new(
        MISSING_UTIL_BLOCK,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        &device,
        &queue,
    );

    assert!(context_result.is_err());
}

const INPUTS: &str = r"
#version 450
#pragma input(float, name=foo)
#pragma input(point, name=bar)
#pragma input(color, name=baz)
#pragma input(bool, name=qux)
layout(set = 0, binding = 0) uniform ShaderInputs {
    float foo;
    vec2 bar;
    vec4 baz;
    int qux;
    int qux_1;
};

void main() {
    // NOTE: this test FAILS without this line!
    // i'm 99% sure this is a bug in naga that accidentally
    // omits unused uniforms from reflection! bad!
   float ugh = foo;
}
";

#[test]
fn all_input_types() {
    let (device, queue) = set_up_wgpu();
    let context_result =
        RenderContext::new(INPUTS, wgpu::TextureFormat::Bgra8UnormSrgb, &device, &queue);

    let res = context_result.unwrap();

    assert!(matches!(res.get_input("foo"), Some(InputType::Float(_))));
    assert!(matches!(res.get_input("bar"), Some(InputType::Point(_))));
    assert!(matches!(res.get_input("baz"), Some(InputType::Color(_))));
    assert!(matches!(res.get_input("qux"), Some(InputType::Bool(_))));
}

const MISSING_INPUTS: &str = r"
#version 450
#pragma input(float, name=foo)
layout(set = 0, binding = 0) uniform ShaderInputs {
    int qux_1;
};

void main() {
   int test = qux_1;
}
";

#[test]
fn missing_inputs() {
    let (device, queue) = set_up_wgpu();
    let context_result = RenderContext::new(
        MISSING_INPUTS,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        &device,
        &queue,
    );

    assert!(context_result.is_err());
}

fn set_up_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = if cfg!(windows) {
        let desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        };

        wgpu::Instance::new(&desc)
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

    pollster::block_on(async {
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: MemoryHints::Performance,
                experimental_features: ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device")
    })
}
