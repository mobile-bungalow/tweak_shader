@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    var uv : vec2<u32> = vec2<u32>(
        (in_vertex_index << 1u) & 2u,
        in_vertex_index & 2u
    );
    
    var uv_f32 : vec2<f32> = vec2<f32>(f32(uv.x), f32(uv.y));
    
    return vec4<f32>(
        uv_f32 * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0),
        0.0,
        1.0
    );
}
