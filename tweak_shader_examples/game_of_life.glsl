#version 450
#pragma tweak_shader(version="1.0")
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
#pragma relay(name="current_state", target="previous_state", persistent)
layout(rgba8, set = 0, binding = 2) uniform writeonly image2D current_state;
layout(set = 0, binding = 3) uniform texture2D previous_state;

#pragma input(image, name="input_image")
layout(set = 0, binding = 4) uniform texture2D input_image;

float random(in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

int getNeighborCount(ivec2 pixel_coords, ivec2 image_size) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            ivec2 neighbor = pixel_coords + ivec2(i, j);
            neighbor = (neighbor + image_size) % image_size; // Wrap around
            vec4 cell = texelFetch(previous_state, neighbor, 0);
            count += int(luma(cell.rgb) > 0.5);
        }
    }
    return count;
}

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = textureSize(previous_state, 0);
    vec2 normalized_coords = vec2(pixel_coords) / vec2(image_size);

    ivec2 input_image_size = textureSize(input_image, 0);
    bool use_input_image = input_image_size.x >= 10 && input_image_size.y >= 10;

    if (pass_index == 0) {
        vec4 new_cell;

        if (frame_index % 1000 == 0) {
            float initial_state;
            if (use_input_image) {
                vec4 input_color = texelFetch(input_image, pixel_coords, 0);
                float pixel_luma = luma(input_color.rgb);
                float rand_value = random(normalized_coords);
                initial_state = step(1.0 - pixel_luma, rand_value);
            } else {
                initial_state = step(0.2, random(normalized_coords));
            }
            new_cell = vec4(vec3(initial_state), 1.0);
        } else {
            vec4 current_cell = texelFetch(previous_state, pixel_coords, 0);
            int neighbor_count = getNeighborCount(pixel_coords, image_size);
            float new_state = current_cell.r;

            if (current_cell.r > 0.5) {
                if (neighbor_count < 2 || neighbor_count > 3) {
                    new_state = 0.0; // Dies from under/overpopulation
                }
            } else {
                if (neighbor_count == 3) {
                    new_state = 1.0; // Becomes alive from reproduction
                }
            }
            new_cell = vec4(vec3(new_state), 1.0);
        }

        imageStore(current_state, pixel_coords, new_cell);
    } else {
        // Pass 1 (main): Copy the result to the screen
        vec4 cell = texelFetch(previous_state, pixel_coords, 0);
        imageStore(output_image, pixel_coords, cell);
    }
}
