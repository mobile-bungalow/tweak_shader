#version 450

#pragma input(float, name="aspect_ratio", default=1.777)
#pragma input(float, name="output_height")
#pragma input(float, name="output_width")
layout(push_constant) uniform TexInfo {
  float output_height;
  float output_width;
  float aspect_ratio;
};

#pragma input(image, name=image)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D image;

layout(location = 0) out vec4 out_color; 

void main() {

  ivec2 i_size = textureSize(sampler2D(image, default_sampler), 0);
  vec2 tex_size = vec2(float(i_size.x), float(i_size.y));
  vec2 screen_size = vec2(output_width, output_height);
  float screen_aspect = output_width / output_height; 

  vec2 scale = vec2(1.0, 1.0);

  // texture wider than target
  // scale down x
  if (aspect_ratio < screen_aspect) {
    scale.x = aspect_ratio / screen_aspect;
  }

  // target wider that texture
  // scale down y 
  if (aspect_ratio > screen_aspect) {
    scale.y = screen_aspect / aspect_ratio ;
  }
 
  // offset to center
  vec2 offset = vec2(0.5) - (scale * 0.5);

  //  screen pixel coord
  vec2 uv = gl_FragCoord.xy / screen_size;

  // scale down UV's 
  uv = (uv - offset) / scale;

  out_color = texture(sampler2D(image, default_sampler), uv);

  // black bars for OOB 
  if ( any(lessThan(uv, vec2(0.0))) ||  any(greaterThan(uv, vec2(1.0))) ) {
    out_color = vec4(0.0, 0.0, 0.0, 1.0); 
  }
}



