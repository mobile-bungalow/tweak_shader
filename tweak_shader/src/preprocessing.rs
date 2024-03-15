use std::collections::HashMap;
use wgpu::naga::{self, Expression, Handle, Span, Statement};

const U15_ARGB_ENTRYPOINT: &str = "
#import include as Inc

@fragment
fn fragment(
    @builtin(position) frag_coord: vec4<f32>,
) -> @location(0) vec4<u16>  {
    const out: vec4 = Inc::main(frag_coord).argb;
    return vec4u(out);
}
";

const UNORM_ARGB_ENTRYPOINT: &str = "
#import include as Inc

@fragment
fn fragment(
    @builtin(position) frag_coord: vec4<f32>,
) -> @location(0) vec4<f32>  {
    return Inc::main(frag_coord).argb;
}
";

pub fn convert_output_to_ae_format(
    module: naga::Module,
    fmt: wgpu::TextureFormat,
) -> Result<wgpu::naga::Module, Box<dyn std::error::Error>> {
    // Wraps the texture lookups in functions that move the channels into the right positions.
    let new_module = wrap_texture_lookups(module)?;
    // Wraps the entry point into a function the realligns the channels.
    let res = wrap_entrypoint(&new_module, fmt)?;
    Ok(res)
}

fn wrap_texture_lookups(
    mut module: naga::Module,
) -> Result<wgpu::naga::Module, Box<dyn std::error::Error>> {
    for (_, func) in module.functions.iter_mut() {
        let mut swizzles = vec![];

        visit_all_expressions(&mut func.body, &mut |expr| {
            if let naga::Expression::ImageSample { .. } = func.expressions.get_mut(*expr) {
                swizzles.push(naga::Expression::Swizzle {
                    size: naga::VectorSize::Quad,
                    vector: expr.clone(),
                    pattern: [
                        naga::SwizzleComponent::W,
                        naga::SwizzleComponent::X,
                        naga::SwizzleComponent::Y,
                        naga::SwizzleComponent::Z,
                    ],
                });
            }
        });

        let swiz_map: std::collections::HashMap<_, _> = swizzles
            .into_iter()
            .filter_map(|swiz| {
                let naga::Expression::Swizzle {
                    vector: og_expr, ..
                } = swiz
                else {
                    unreachable!()
                };
                let new_handle = func.expressions.fetch_or_append(swiz, Span::UNDEFINED);
                Some((og_expr, new_handle))
            })
            .collect();

        visit_all_expressions(&mut func.body, &mut |b| {
            if let Some(new_expr) = swiz_map.get(b) {
                *b = *new_expr;
            }
        });

        splice_emission_ranges(&mut func.body, swiz_map);
    }

    Ok(module)
}

fn splice_emission_ranges(
    func: &mut naga::Block,
    points_to_replace: HashMap<Handle<Expression>, Handle<Expression>>,
) {
    for (original, target) in points_to_replace {
        // emissions that contain the original expression
        let containing_emissions: Vec<_> = func
            .iter()
            .filter_map(|b| match b {
                naga::Statement::Emit(range) => {
                    if range.clone().into_iter().any(|e| e == original) {
                        Some(b.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        for range in containing_emissions {
            let naga::Statement::Emit(emission) = range else {
                continue;
            };

            if !emission.clone().into_iter().any(|e| e == original) {
                continue;
            }

            //
            let new_chunk = naga::Range::new_from_bounds(target, target);

            let range = emission.first_and_last();

            // index of the emission span in function body
            let Some(index) = func
                .iter()
                .position(|e| matches!(e, Statement::Emit(i) if i.first_and_last() == range))
            else {
                continue;
            };

            match range {
                // replace whole range (one item)
                Some((start, end)) if start == original && end == original => {
                    let new_block = naga::Block::from_vec(vec![naga::Statement::Emit(new_chunk)]);
                    func.splice(index..index + 1, new_block);
                }
                // prepend single item
                Some((start, _)) if start == original => {
                    let vec: Vec<_> = emission.clone().into_iter().collect();

                    if let Some([new_end, _]) =
                        vec.as_slice().windows(2).find(|slice| match slice {
                            &[removed, _] => *removed == original,
                            _ => false,
                        })
                    {
                        let first_chunk = naga::Range::new_from_bounds(start, *new_end);

                        let new_block = naga::Block::from_vec(vec![
                            naga::Statement::Emit(first_chunk),
                            naga::Statement::Emit(new_chunk),
                        ]);

                        func.splice(index..index + 1, new_block);
                    };
                }
                // append single item
                Some((start, end)) if end == original => {
                    let vec: Vec<_> = emission.clone().into_iter().collect();

                    if let Some([new_end, _]) =
                        vec.as_slice().windows(2).find(|slice| match slice {
                            &[_, removed] => *removed == original,
                            _ => false,
                        })
                    {
                        let first_chunk = naga::Range::new_from_bounds(start, *new_end);

                        let new_block = naga::Block::from_vec(vec![
                            naga::Statement::Emit(first_chunk),
                            naga::Statement::Emit(new_chunk),
                        ]);

                        func.splice(index..index + 1, new_block);
                    };
                }
                // split range and insert between
                Some((start, end)) => {
                    let vec: Vec<_> = emission.clone().into_iter().collect();
                    // the only way to get handles is by iterating.
                    if let Some([new_end, _, new_next]) =
                        vec.as_slice().windows(3).find(|slice| match slice {
                            &[_, removed, _] => *removed == original,
                            _ => false,
                        })
                    {
                        let first_chunk = naga::Range::new_from_bounds(start, *new_end);
                        let second_chunk = naga::Range::new_from_bounds(*new_next, end);

                        let new_block = naga::Block::from_vec(vec![
                            naga::Statement::Emit(first_chunk),
                            naga::Statement::Emit(new_chunk),
                            naga::Statement::Emit(second_chunk),
                        ]);

                        func.splice(index..index + 1, new_block);
                    };
                }
                _ => {}
            }
        }
    }
}

fn visit_all_expressions<F>(func: &mut naga::Block, cb: &mut F)
where
    F: FnMut(&mut Handle<naga::Expression>),
{
    for block in func.iter_mut() {
        match block {
            naga::Statement::Emit(_) => {
                // handled in emission range code
            }
            naga::Statement::Block(inner_block) => visit_all_expressions(inner_block, cb),
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                cb(condition);
                visit_all_expressions(accept, cb);
                visit_all_expressions(reject, cb);
            }
            naga::Statement::Switch { selector, cases } => {
                cb(selector);
                for case in cases.iter_mut() {
                    visit_all_expressions(&mut case.body, cb);
                }
            }
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                visit_all_expressions(body, cb);
                visit_all_expressions(continuing, cb);
                if let Some(condition) = break_if {
                    cb(condition);
                }
            }
            naga::Statement::Return { value } => {
                if let Some(expr) = value {
                    cb(expr);
                }
            }
            naga::Statement::Store { pointer, value } => {
                cb(pointer);
                cb(value);
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                cb(image);
                cb(value);
                cb(coordinate);
                if let Some(array_index) = array_index {
                    cb(array_index);
                }
            }
            naga::Statement::Atomic {
                pointer,
                value,
                result,
                ..
            } => {
                cb(pointer);
                cb(value);
                cb(result);
            }
            naga::Statement::WorkGroupUniformLoad { pointer, result } => {
                cb(pointer);
                cb(result);
            }
            naga::Statement::Call {
                arguments, result, ..
            } => {
                for expr in arguments.iter_mut() {
                    cb(expr);
                }
                if let Some(expr) = result {
                    cb(expr);
                }
            }
            naga::Statement::RayQuery { query, .. } => cb(query),
            naga::Statement::Barrier(_)
            | naga::Statement::Kill
            | naga::Statement::Break
            | naga::Statement::Continue => {}
        }
    }
}

fn wrap_entrypoint(
    module: &wgpu::naga::Module,
    fmt: wgpu::TextureFormat,
) -> Result<wgpu::naga::Module, Box<dyn std::error::Error>> {
    let wrapper = if let wgpu::TextureFormat::Rgba16Unorm = fmt {
        U15_ARGB_ENTRYPOINT
    } else {
        UNORM_ARGB_ENTRYPOINT
    };
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;
    use pretty_assertions::assert_eq;

    const NO_SAMPLE: &str = "
#version 450

#pragma input(image, name=image)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D image;

layout(location = 0) out vec4 out_color; 

void main() {
  out_color = vec4(1.0);
}
";

    #[test]
    fn no_sample() {
        let mut frontend = naga::front::glsl::Frontend::default();

        let options = naga::front::glsl::Options::from(naga::ShaderStage::Fragment);
        let naga_mod = frontend.parse(&options, &NO_SAMPLE).unwrap();

        let new_mod = wrap_texture_lookups(naga_mod.clone()).unwrap();

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&new_mod)
        .unwrap();

        let modded_wgsl = naga::back::wgsl::write_string(
            &new_mod,
            &info,
            naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
        )
        .unwrap();

        let vanilla_wgsl = naga::back::wgsl::write_string(
            &naga_mod,
            &info,
            naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
        )
        .unwrap();

        assert_eq!(modded_wgsl, vanilla_wgsl);
    }

    const FUNC_NO_SAMPLERS: &str = "
#version 450

#pragma input(image, name=image)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D image;

 vec4 global_var = vec4(0.0);

void complexFunction() {
  global_var.x = 32.0 * sin(global_var.z);
  return;
}

layout(location = 0) out vec4 out_color; 

void main() {
  complexFunction();
  out_color = global_var;
}
";

    #[test]
    fn complex_function_texture_sample() {
        let mut frontend = naga::front::glsl::Frontend::default();

        let options = naga::front::glsl::Options::from(naga::ShaderStage::Fragment);
        let naga_mod = frontend.parse(&options, &FUNC_NO_SAMPLERS).unwrap();

        let new_mod = wrap_texture_lookups(naga_mod.clone()).unwrap();

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&new_mod)
        .unwrap();

        let modded_wgsl = naga::back::wgsl::write_string(
            &new_mod,
            &info,
            naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
        )
        .unwrap();

        let vanilla_wgsl = naga::back::wgsl::write_string(
            &naga_mod,
            &info,
            naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
        )
        .unwrap();

        assert_eq!(modded_wgsl, vanilla_wgsl);
    }

    const BASIC: &str = "
#version 450

#pragma input(image, name=image)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D image;

layout(location = 0) out vec4 out_color; 

void main() {
  out_color = texture(sampler2D(image, default_sampler), vec2(0.0));
}
";

    const BASIC_OUT: &str = "struct FragmentOutput {
    @location(0) out_color: vec4<f32>,
}

@group(0) @binding(1) 
var default_sampler: sampler;
@group(0) @binding(2) 
var image: texture_2d<f32>;
var<private> out_color: vec4<f32>;

fn main_1() {
    out_color = textureSample(image, default_sampler, vec2(0f)).wxyz;
    return;
}

@fragment 
fn main() -> FragmentOutput {
    main_1();
    let _e7: vec4<f32> = out_color;
    return FragmentOutput(_e7);
}
";

    #[test]
    fn basic_shader() {
        let mut frontend = naga::front::glsl::Frontend::default();

        let options = naga::front::glsl::Options::from(naga::ShaderStage::Fragment);
        let naga_mod = frontend.parse(&options, &BASIC).unwrap();

        let new_mod = wrap_texture_lookups(naga_mod).unwrap();

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&new_mod)
        .unwrap();

        let wgsl = naga::back::wgsl::write_string(
            &new_mod,
            &info,
            naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
        )
        .unwrap();

        assert_eq!(&wgsl, BASIC_OUT);
    }
}
