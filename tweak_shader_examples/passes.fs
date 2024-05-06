#version 450
#pragma tweak_shader(version="1.0")

// This shader builds up four rectangles in each corner of the 
// screen in a series of passes
#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;       // shader playback time (in seconds)
    float time_delta; // elapsed time since last frame in secs
    float frame_rate; // number of frames per second estimates
    uint frame_index;  // frame count
    vec4 mouse;       // xy is last mouse down position,  abs(zw) is current mouse, sign(z) > 0.0 is mouse_down, sign(w) > 0.0 is click_down event
    vec4 date;        // [year, month, day, seconds]
    vec3 resolution;  // viewport resolution in pixels, [w, h, w/h]
    uint pass_index;   // updated to reflect render pass
};

#pragma pass(0, target="dataHistory")
#pragma pass(1, target="dataHistory")
#pragma pass(2, target="dataHistory")
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D dataHistory;

layout(location = 0) out vec4 out_color; 

void main()	{
  ivec2 size = textureSize(sampler2D(dataHistory, default_sampler), 0);
  vec2 fsize = vec2(float(size.x), float(size.y));
	vec2	loc = gl_FragCoord.xy;
	vec4 val = texture(sampler2D(dataHistory, default_sampler), loc / fsize);
	vec4	inputPixelColor = vec4(0.0, 0.0, 0.0, 1.0);

  inputPixelColor = val;
  inputPixelColor.a = 1.0;

  if (pass_index == 0 && (loc.x / fsize.x) > 0.5 && (loc.y / fsize.y) < 0.5) {
    inputPixelColor.rgb = vec3(1.0, 0.0, 0.0);
  }

  if (pass_index == 1 && (loc.x / fsize.x) < 0.5 && (loc.y / fsize.y) < 0.5) {
    inputPixelColor.rgb = vec3(0.0, 1.0, 0.0);
  }

  if (pass_index == 2 && (loc.x / fsize.x) < 0.5 && (loc.y / fsize.y) > 0.5) {
    inputPixelColor.rgb = vec3(0.0, 0.0, 1.0);
  }

  if (pass_index == 3 && (loc.x / fsize.x) > 0.5 && (loc.y / fsize.y) > 0.5) {
    inputPixelColor.rgb = vec3(1.0, 1.0, 0.0);
  }

	out_color = inputPixelColor;
}

