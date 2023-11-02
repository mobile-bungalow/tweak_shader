@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32((in_vertex_index << 1u) & 2u));
    let y = f32(i32(in_vertex_index & 2u));
    return vec4<f32>((x * 2.0) - 1.0, (y * 2.0) - 1.0, 0.0, 1.0);
}