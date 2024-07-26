#version 450

#pragma tweak_shader(version="1.0")
#pragma stage(compute)

#pragma target(0, name="output_image")
layout(rgba32f, set=0, binding=3) uniform writeonly image2D output_image;

// You have access to these built in globals
//in uvec3 gl_NumWorkGroups;
//in uvec3 gl_WorkGroupID;
//in uvec3 gl_LocalInvocationID;
//in uvec3 gl_GlobalInvocationID;
//in uint  gl_LocalInvocationIndex;

layout(local_size_x = 16, local_size_y = 16) in;
void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec4 color = vec4(1.0);

    imageStore(output_image, pixel_coords, color);
}
