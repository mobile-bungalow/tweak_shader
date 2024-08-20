#version 450
#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;       // shader playback time (in seconds)
    float time_delta; // elapsed time since last frame in secs
    float frame_rate; // number of frames per second estimates
    uint frame_index;  // frame count
    vec4 mouse;       // xy is last mouse down position,  abs(zw) is current mouse, sign(z) > 0.0 is mouse_down, sign(w) > 0.0 is click_down event
    vec4 date;        // [year, month, day, seconds]
    vec3 resolution;  // viewport resolution in pixels, [w, h, w/h]
    uint pass_index;   // updated to reflect render pass
};

layout(location = 0) out vec4 out_color; 

#pragma input(float, name="foo", default=0.0, min=0.0, max=1.0)
layout(set = 1, binding = 0) uniform Ecco {
    float foo;
};

#pragma input(float, name="john", default=0.0, min=0.0, max=1.0)
layout(set=0, binding=5) uniform float john;


void main()
{
    vec2 st = (gl_FragCoord.xy / resolution.xy);
    out_color = vec4(foo + john, 0.0, st.x, 1.0);
}
