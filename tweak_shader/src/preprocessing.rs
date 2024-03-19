use glsl::parser::Parse;
use glsl::syntax::{
    Declaration, Expr, ExternalDeclaration, FullySpecifiedType, FunctionPrototype,
    InitDeclaratorList, SimpleStatement, SingleDeclaration, TypeQualifier, TypeQualifierSpec,
    TypeSpecifier, TypeSpecifierNonArray,
};
use glsl::visitor::HostMut;
use glsl::visitor::VisitorMut;
use naga::front::glsl::{Frontend, Options};
use wgpu::naga;

pub fn convert_output_to_argb_format(
    module: &String,
) -> Result<naga::Module, Vec<naga::front::glsl::Error>> {
    let mut swiz = FormatSwizzler::new();
    let mut expr = glsl::syntax::TranslationUnit::parse(module).unwrap();
    expr.visit_mut(&mut swiz);

    let mut output = String::new();
    glsl::transpiler::glsl::show_translation_unit(&mut output, &expr);

    let mut options = Options::from(naga::ShaderStage::Fragment);
    options
        .defines
        .insert("TWEAK_SHADER".to_owned(), "1".to_owned());

    let mut frontend = Frontend::default();
    frontend.parse(&options, &output)
}

const TEXTURE_SAMPLING_FUNCTIONS: [&str; 7] = [
    "texture",
    "textureOffset",
    "textureProj",
    "textureProjOffset",
    "textureLod",
    "textureLodOffset",
    "textureGrad",
];

struct EntryPointExitSwizzler {
    pub out_var: String,
}

// identify the end of the top level block, and find every block that ends in return;
// inserting a swizzle statement at the end of it.
impl EntryPointExitSwizzler {
    pub fn new(out_var: String) -> Self {
        Self { out_var }
    }
}

impl VisitorMut for EntryPointExitSwizzler {
    fn visit_compound_statement(
        &mut self,
        block: &mut glsl::syntax::CompoundStatement,
    ) -> glsl::visitor::Visit {
        let mut ret_index = None;
        for (index, statement) in block.statement_list.iter().enumerate() {
            if let glsl::syntax::Statement::Simple(simple) = statement {
                if let SimpleStatement::Jump(glsl::syntax::JumpStatement::Return(_)) = **simple {
                    ret_index = Some(index);
                }
            }
        }
        if let Some(index) = ret_index {
            let out_var = self.out_var.clone();
            let ender = format!("{out_var} = {out_var}.argb;");
            let new_stmnt = glsl::syntax::Statement::parse(ender);
            if let Ok(end_stmnt) = new_stmnt {
                block
                    .statement_list
                    .insert(index.saturating_sub(1), end_stmnt);
            }
        }

        glsl::visitor::Visit::Children
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
                for expr in args.iter_mut() {
                    expr.visit_mut(self);
                }

                let mut string = String::new();
                glsl::transpiler::glsl::show_function_identifier(&mut string, id);
                if TEXTURE_SAMPLING_FUNCTIONS.contains(&string.as_str()) {
                    let clone = e.clone();
                    let swizzed = Expr::Dot(clone.into(), glsl::syntax::Identifier("gbar".into()));
                    *e = swizzed;
                }

                return glsl::visitor::Visit::Parent;
            }
            _ => {}
        }

        glsl::visitor::Visit::Children
    }

    fn visit_translation_unit(
        &mut self,
        tu: &mut glsl::syntax::TranslationUnit,
    ) -> glsl::visitor::Visit {
        // add the swizzle to the end of the compound statement.
        // changing the output variable to argb;

        // then implement the return point checker, which goes through every block in the
        // funcition def compound statement and inserts the swizzle before each return statement;

        let mut exit_swiz = None;

        for item in &mut tu.0 {
            // 'ello Giza - just check if we can get an "out vec4"
            if let ExternalDeclaration::Declaration(Declaration::InitDeclaratorList(
                InitDeclaratorList {
                    head:
                        SingleDeclaration {
                            ty:
                                FullySpecifiedType {
                                    qualifier: Some(TypeQualifier { qualifiers }),
                                    ty:
                                        TypeSpecifier {
                                            ty: TypeSpecifierNonArray::Vec4,
                                            array_specifier: None,
                                        },
                                },
                            name: Some(name),
                            array_specifier: None,
                            ..
                        },
                    ..
                },
            )) = item
            {
                if qualifiers.0.contains(&TypeQualifierSpec::Storage(
                    glsl::syntax::StorageQualifier::Out,
                )) {
                    let mut string = String::new();
                    glsl::transpiler::glsl::show_identifier(&mut string, name);
                    exit_swiz = Some(EntryPointExitSwizzler::new(string));
                    break;
                }
            }
        }

        if let Some(mut swizzler) = exit_swiz {
            for item in &mut tu.0 {
                if let ExternalDeclaration::FunctionDefinition(glsl::syntax::FunctionDefinition {
                    prototype: FunctionPrototype { name, .. },
                    statement,
                }) = item
                {
                    let mut string = String::new();
                    glsl::transpiler::glsl::show_identifier(&mut string, name);
                    if string == "main" {
                        statement.visit_mut(&mut swizzler);
                        let out_var = swizzler.out_var;

                        let ender = format!("{out_var} = {out_var}.argb;");
                        let new_stmnt = glsl::syntax::Statement::parse(ender);
                        if let Ok(end_stmnt) = new_stmnt {
                            statement.statement_list.push(end_stmnt);
                        }
                        break;
                    }
                }
            }
        }

        glsl::visitor::Visit::Children
    }
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
        let post = "vec3 test = texture(sampler2D(image, sampler), vec2(1., 1.)).gbar;\n";
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
        let post = "vec3 test = texture(sampler2D(image, sampler), vec2(1., 1.)).gbar.x;\n";
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
        let post = "vec3 test = texture(sampler2D(image, sampler), textureLod(sampler2D(image_2, sampler), vec2(1., 1.)).gbar.xy).gbar.x;\n";
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
        let post= "layout (location = 0) out vec4 out_color;\nvoid main() {\nout_color = vec4(1.);\nout_color = out_color.argb;\n}\n";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::TranslationUnit::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_translation_unit(&mut string, &expr);
        assert_eq!(post, &string);
    }

    #[test]
    fn full_shader_early_return() {
        let pre = "
    layout(location = 0) out vec4 out_color; 

    void main() { 
        if ( 1.0 == 1.0) {
            return;
        } else {
            return;
        }
        out_color = vec4(1.);
    }";

        let post = "layout (location = 0) out vec4 out_color;
void main() {
if (1.==1.) {
{
out_color = out_color.argb;
return ;
}
} else {
out_color = out_color.argb;
return ;
}
out_color = vec4(1.);
out_color = out_color.argb;
}
";
        let mut swiz = FormatSwizzler::new();
        let mut expr = glsl::syntax::TranslationUnit::parse(pre).unwrap();
        expr.visit_mut(&mut swiz);
        let mut string = String::new();
        glsl::transpiler::glsl::show_translation_unit(&mut string, &expr);
        assert_eq!(post, &string);
    }
}
