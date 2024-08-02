#version 450

#pragma tweak_shader(version="1.0")
#pragma stage(compute)


#pragma input(float, name=blur, default=0.0, min=0.0, max=1.0)
layout(set=1, binding=0) uniform custom_inputs {
    float blur;
};

#pragma target(name="output_image", screen)
layout(rgba8, set=0, binding=1) uniform writeonly image2D output_image;

// You have access to these built in globals
//in uvec3 gl_NumWorkGroups;
//in uvec3 gl_WorkGroupID;
//in uvec3 gl_LocalInvocationID;
//in uvec3 gl_GlobalInvocationID;
//in uint  gl_LocalInvocationIndex;

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(output_image);
    
    vec2 normalized_coords = vec2(pixel_coords) / vec2(image_size);
    
    vec4 color = vec4( normalized_coords.x,
        normalized_coords.y,
        blur,
        1.0
    );
    
    imageStore(output_image, pixel_coords, color);
}
