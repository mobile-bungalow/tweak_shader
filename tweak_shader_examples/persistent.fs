#version 450
#pragma tweak_shader(version="1.0")

// Adapted from the VidVox "data" ISF shader.

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

#pragma input(color, name=data, default = [0.95, 0.35, 0, 1])
#pragma input(int, name=displayMode, default=0, values=[0,1], labels=["block", "lines"])
layout(set = 0, binding = 3) uniform Data {
  vec4 data;
  int displayMode;
};

#pragma pass(0, persistent, target="dataHistory", height=1)
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D dataHistory;

layout(location = 0) out vec4 out_color; 

void main()	{
  ivec2 size = textureSize(sampler2D(dataHistory, default_sampler), 0);
  vec2 fsize = vec2(float(size.x), float(size.y));
	vec2	loc = gl_FragCoord.xy;
	vec4	inputPixelColor = vec4(0.0, 0.0, 0.0, 1.0);
    if (pass_index == 0) {
	    inputPixelColor = texture(sampler2D(dataHistory, default_sampler), vec2(loc.x - 1.0, 0.0) / fsize);
	    if (floor(loc.x) == 0.0)	{
	    	inputPixelColor = data;
	    } 
    } else {
		    vec4 val = texture(sampler2D(dataHistory, default_sampler), loc / fsize);
      if (displayMode == 0) {
		  	  inputPixelColor = val;
      } else if (displayMode == 1)	{
			float	tmp = floor(val.r * resolution.y);
			inputPixelColor.a = val.a;
			if (abs((resolution.y - loc.y) - tmp) < 5.0)	{
				inputPixelColor.r = 1.0;
				inputPixelColor.a = 1.0;
			}
			tmp = floor(val.g * resolution.y);
			if (abs((resolution.y - loc.y) - tmp) < 5.0)	{
				inputPixelColor.g = 1.0;
				inputPixelColor.a = 1.0;
			}
			tmp = floor(val.b * resolution.y);
			if (abs((resolution.y - loc.y) - tmp) < 5.0)	{
				inputPixelColor.b = 1.0;
				inputPixelColor.a = 1.0;
			}
		}
  }
	out_color = inputPixelColor;
}

