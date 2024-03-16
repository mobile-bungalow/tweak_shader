use glsl::parser::Parse;
use glsl::syntax::{Expr, ExprStatement, ShaderStage, SingleDeclaration, Statement};
use glsl::visitor::HostMut;
use glsl::visitor::VisitorMut;
use std::iter::FromIterator;
use wgpu::naga;

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

const TEXTURE_SAMPLING_FUNCTIONS: [&str; 7] = [
    "texture",
    "textureOffset",
    "textureProj",
    "textureProjOffset",
    "textureLod",
    "textureLodOffset",
    "textureGrad",
];

struct EntryPointExitSwizzler {}

impl EntryPointExitSwizzler {
    pub fn new() -> Self {
        Self {}
    }
}

struct FormatSwizzler {}

impl FormatSwizzler {
    pub fn new() -> Self {
        Self {}
    }
}

impl VisitorMut for FormatSwizzler {
    fn visit_expr(&mut self, e: &mut Expr) -> glsl::visitor::Visit {
        match e {
            Expr::FunCall(id, args) => {
                let mut string = String::new();

                for expr in args.iter_mut() {
                    expr.visit_mut(self);
                }

                glsl::transpiler::glsl::show_function_identifier(&mut string, &id);
                if TEXTURE_SAMPLING_FUNCTIONS.contains(&string.as_str()) {
                    let clone = e.clone();
                    let swizzed = Expr::Dot(clone.into(), glsl::syntax::Identifier("argb".into()));
                    *e = swizzed;
                }

                return glsl::visitor::Visit::Parent;
            }
            _ => {}
        }

        glsl::visitor::Visit::Children
    }
}

pub fn convert_output_to_ae_format(
    module: &String,
    fmt: wgpu::TextureFormat,
) -> Result<naga::Module, Box<dyn std::error::Error>> {
    let tu = ShaderStage::parse(module);
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;
    use pretty_assertions::assert_eq;
    #[test]
    fn visit_single_expression() {
        let pre = "vec3 test = vec3(1.);\n";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::Statement::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_statement(&mut string, &expr);
        assert_eq!(pre, &string);
    }

    #[test]
    fn visit_sampler_simple() {
        let pre = "vec3 test = texture(sampler2D(image, sampler), vec2(1., 1.));\n";
        let post = "vec3 test = texture(sampler2D(image, sampler), vec2(1., 1.)).argb;\n";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::Statement::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_statement(&mut string, &expr);
        assert_eq!(post, &string);
    }

    #[test]
    fn visit_sampler() {
        let pre = "vec3 test = texture(sampler2D(image, sampler), vec2(1., 1.)).x;\n";
        let post = "vec3 test = texture(sampler2D(image, sampler), vec2(1., 1.)).argb.x;\n";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::Statement::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_statement(&mut string, &expr);
        assert_eq!(post, &string);
    }

    #[test]
    fn two_levels_deep() {
        let pre = "vec3 test = texture(sampler2D(image, sampler), textureLod(sampler2D(image_2, sampler), vec2(1., 1.)).xy).x;\n";
        let post = "vec3 test = texture(sampler2D(image, sampler), textureLod(sampler2D(image_2, sampler), vec2(1., 1.)).argb.xy).argb.x;\n";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::Statement::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_statement(&mut string, &expr);
        assert_eq!(post, &string);
    }

    #[test]
    fn full_shader() {
        let pre =
            "layout(location = 0) out vec4 out_color; void main() { out_color = vec4(1.); } \n";
        let post= "layout (location = 0) out vec4 out_color;\nvoid main() {\nout_color = vec4(1.);\nout_color = out_color.argb;\n} \n";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::TranslationUnit::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_translation_unit(&mut string, &expr);
        assert_eq!(post, &string);
    }
}
