use naga_oil::{compose, redirect};
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

const ARGB_SAMPLER_MODULE: &str = "
  #define_import_path samplers 

  vec4 sampleTextureRGBA(sampler2D tex, vec2 texCoords) {
      return texture(tex, texCoords).gbra;
  }
  
  vec4 sampleTextureRGBALod(sampler2D tex, vec2 texCoords, float lod) {
      return textureLod(tex, texCoords, lod).gbra;
  }
  
  vec4 sampleTextureRGBAGrad(sampler2D tex, vec2 texCoords, vec2 dPdx, vec2 dPdy) {
      return textureGrad(tex, texCoords, dPdx, dPdy).gbra;
  }

  vec4 texelFetchRGBA(sampler2D tex, ivec2 texCoords, int lod) {
      return texelFetchRGBA(tex, texCoords, lod).gbra;
  }
  
  vec4 sampleTextureOffsetRGBA(sampler2D tex, vec2 texCoords, ivec2 offset) {
      return textureOffset(tex, texCoords, offset).gbra;
  }
  
  vec4 sampleTextureProjRGBA(sampler2D tex, vec3 texCoords) {
      return textureProj(tex, texCoords).gbra;
  }
";

pub fn convert_output_to_ae_format(
    module: &str,
    fmt: wgpu::TextureFormat,
) -> Result<wgpu::naga::Module, Box<dyn std::error::Error>> {
    let new_module_source = wrap_texture_lookups(module)?;
    let res = wrap_entrypoint(&new_module_source, fmt)?;
    Ok(res)
}

fn wrap_texture_lookups(module: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut combined = compose::Composer::default();

    let mut defined = std::collections::HashMap::new();
    defined.insert(
        "TWEAK_SHADER".to_owned(),
        compose::ShaderDefValue::Bool(true),
    );
    defined.insert(
        "AFTER_EFFECTS".to_owned(),
        compose::ShaderDefValue::Bool(true),
    );

    combined.add_composable_module(compose::ComposableModuleDescriptor {
        source: module,
        file_path: "main.glsl",
        language: compose::ShaderLanguage::Glsl,
        as_name: None,
        additional_imports: &[],
        shader_defs: defined.clone(),
    });

    combined.add_composable_module(compose::ComposableModuleDescriptor {
        source: ARGB_SAMPLER_MODULE,
        file_path: "samplers.glsl",
        language: compose::ShaderLanguage::Glsl,
        as_name: Some("samplers".to_owned()),
        additional_imports: &[],
        shader_defs: defined.clone(),
    });

    let new_mod = combined.make_naga_module(compose::NagaModuleDescriptor {
        source: todo!(),
        file_path: todo!(),
        shader_type: todo!(),
        shader_defs: todo!(),
        additional_imports: todo!(),
    })?;

    let re = naga_oil::redirect::Redirector::new(new_mod);

    let none = &Default::default();
    re.redirect_function("texture", "sampleTextureRGBA", none)?;
    //re.redirect_function("textureLod", "sampleTextureLodRGBA", none)?;
    //re.redirect_function("textureGrad", "sampleTextureGradRGBA", none)?;
    //re.redirect_function("textureOffset", "sampleTextureOffsetRGBA", none)?;
    //re.redirect_function("textureProj", "sampleTextureProjRGBA", none)?;
    //re.redirect_function("texelFetch", "texelFetchRGBA", none)?;
    let new = re.into_module()?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&new)?;

    let wgsl =
        naga::back::wgsl::write_string(&new, &info, naga::back::wgsl::WriterFlags::EXPLICIT_TYPES)?;
    Ok(wgsl)
}

fn wrap_entrypoint(
    module: &str,
    fmt: wgpu::TextureFormat,
) -> Result<wgpu::naga::Module, Box<dyn std::error::Error>> {
    let wrapper = if let wgpu::TextureFormat::Rgba16Unorm = fmt {
        U15_ARGB_ENTRYPOINT
    } else {
        UNORM_ARGB_ENTRYPOINT
    };
    todo!()
}
