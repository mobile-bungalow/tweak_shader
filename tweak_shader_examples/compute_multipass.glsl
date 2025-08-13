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
layout(rgba8, set = 0, binding = 1) uniform writeonly image2D output_image;

#pragma pass(0)
#pragma relay(name="relay", target="relay_target")
layout(rgba8, set = 0, binding = 2) uniform writeonly image2D relay;
layout(set = 0, binding = 3) uniform texture2D relay_target;

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(output_image);

    vec2 normalized_coords = vec2(pixel_coords) / vec2(image_size);

    vec4 color = vec4(normalized_coords.x, normalized_coords.y, 1.0, 1.0);
    vec4 mask = texelFetch(relay_target, pixel_coords, 0);
    float center_circle = step(length(normalized_coords - (vec2(0.5) + sin(time) * 0.1)), 0.2);

    if (pass_index == 0 && center_circle > 0.0) {
        imageStore(relay, pixel_coords, vec4(center_circle));
    } else {
        imageStore(output_image, pixel_coords, mask * color);
    }
}
