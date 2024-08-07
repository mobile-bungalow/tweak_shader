#version 450
#pragma tweak_shader(version="1.0")
#pragma stage(compute)
#pragma input(float, name=blue, default=0.0, min=0.0, max=1.0)

layout(set=1, binding=0) uniform custom_inputs {
    float blue;
};

#pragma target(name="output_image", screen)
layout(rgba8, set=0, binding=1) uniform writeonly image2D output_image;

#pragma target(name="screen_two", screen)
layout(rgba8, set=0, binding=2) uniform writeonly image2D screen_two;

#pragma target(name="screen_three", screen)
layout(rgba8, set=0, binding=4) uniform writeonly image2D screen_three;

#pragma input(image, name="input_image")
layout(set=0, binding=3) uniform texture2D input_image;

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(output_image);
    
    vec2 normalized_coords = vec2(pixel_coords) / vec2(image_size);
    
    vec4 color = vec4(normalized_coords.x, normalized_coords.y, blue, 1.0);
    
    vec4 inverse = vec4(vec3(1.0) - color.xyz, 1.0);
    imageStore(output_image, pixel_coords, color);
    imageStore(screen_two, pixel_coords, inverse);

    // Texture sampling is forbidden (!!!) in compute shaders
    // so use texelfetch.
    vec4 col = texelFetch(input_image, pixel_coords, 0);
    imageStore(screen_three, pixel_coords, col);
}
