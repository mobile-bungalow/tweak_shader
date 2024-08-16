#version 450
#pragma tweak_shader(version="1.0")

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;       // shader playback time (in seconds)
    float time_delta; // elapsed time since last frame in secs
    float frame_rate; // number of frames per second estimates
    uint frame_index; // frame count
    vec4 mouse;       // xy is last mouse down position,  abs(zw) is current mouse, sign(z) > 0.0 is mouse_down, sign(w) > 0.0 is click_down event
    vec4 date;        // [year, month, day, seconds]
    vec3 resolution;  // viewport resolution in pixels, [w, h, w/h]
    uint pass_index;  // updated to reflect render pass
};

layout(location = 0) out vec4 out_color; 

#pragma sampler(name="default_sampler", linear)
layout(set=0, binding=1) uniform sampler default_sampler;

#pragma input(image, name="input_image")
layout(set=0, binding=2) uniform texture2D input_image;

void main() {
	vec2 uv = gl_FragCoord.xy / resolution.xy;
	out_color = texture(sampler2D(input_image, default_sampler), uv);
}
