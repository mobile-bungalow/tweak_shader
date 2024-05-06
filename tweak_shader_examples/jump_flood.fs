#version 450
#pragma tweak_shader(version="1.0")

// an implementation of the jump flood algorithm, you can do something
// with it's data in the final pass 

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

//TODO: some make a way to generalize this huge stack 

// init
#pragma pass(0, target="distance_field")
// jump flood steps
#pragma pass(1, target="distance_field")
#pragma pass(2, target="distance_field")
#pragma pass(3, target="distance_field")
#pragma pass(4, target="distance_field")
#pragma pass(5, target="distance_field")
#pragma pass(6, target="distance_field")
#pragma pass(7, target="distance_field")
#pragma pass(8, target="distance_field")
#pragma pass(9, target="distance_field")
#pragma pass(10, target="distance_field")
#pragma pass(11, target="distance_field")
#pragma pass(12, target="distance_field")
// Finish
#pragma pass(13, target="distance_field")
layout(set=0, binding=1) uniform sampler default_sampler;
layout(set=0, binding=2) uniform texture2D distance_field;
layout(set=0, binding=3) uniform texture2D matte;


layout(location = 0) out vec4 out_color; 

float TAU = 6.28318530718;

// get alpha from matte texture
float alpha(vec4 color) {
  // wrong but fine for now
  return color.r;
}

void main()	{
  vec2 fsize = resolution.xy; 
	vec2	loc = gl_FragCoord.xy;
	vec4 dist = texture(sampler2D(distance_field, default_sampler), loc / fsize);
  ivec2 d_size = textureSize(distance_field, 0);
  ivec2 matte_size = textureSize(matte, 0);
  vec2 fmatte_size = vec2(float(matte_size.x), float(matte_size.y));

  vec2 aspect = 1.0 / matte_size;
  vec2 uv = loc / fmatte_size;
	vec4 matte_r = texture(sampler2D(matte, default_sampler), uv);
	vec4	inputPixelColor = vec4(0.0, 0.0, 0.0, 1.0);

  // init outer ring
  if (pass_index == 0) {

    if (alpha(matte_r) < 1.0) {

      vec2 dir_to_border = vec2(0.0);
      bool out_of_radius = true;
      for (float i = 0.0; i < TAU; i += TAU / 16.0) {

          vec2 dir = vec2(sin(i), cos(i));
          vec2 new_uv = uv - dir * aspect * 2.0;
          vec4 col = texture(sampler2D(matte, default_sampler), new_uv);
          col.r = step(0.1, 1.0 - col.r);

          if (col.r <= alpha(matte_r)) {
              out_of_radius = false;
              dir_to_border += dir;
          } 
      }

      if (!out_of_radius) {

       float d = 2.0;
       float move = 2.0 ;
       dir_to_border = normalize(dir_to_border);
       vec2 nuv = vec2(0.0);

       for (int i = 0; i < 8 && !out_of_radius; i++) {
           vec2 new_uv = uv + (dir_to_border * aspect * d);
           vec4 col = texture(sampler2D(matte, default_sampler), new_uv);
           col.r = 1.0 - col.r;
           d += col.r > 0.0 ? -move : move; // -4.0 is a weird fudge
           move /= 2.0;
           nuv = new_uv;
       }

       out_color.rg = nuv;
       out_color.a = 1.0;
      }
    }
  } else if (pass_index < 14 && pass_index >= 1) {

          float level = clamp(pass_index - 1.0, 0.0, 13.0);
          int stepwidth = int(exp2(13.0 - level));

          float best_dist = 1000000.0;
          vec2 best_coord = vec2(0.0);
          for (int y = -1; y <= 1; ++y) {
             for (int x = -1; x <= 1; ++x) {
                 vec2 fc = (gl_FragCoord.xy / resolution.xy) + vec2(x,y)*stepwidth*0.0002;
	               vec4 ntc = texture(sampler2D(distance_field, default_sampler), fc);
                 float d = length(ntc.xy - fc);
                 if ((ntc.x != 0.0) && (ntc.y != 0.0) && (d < best_dist)) {
                     best_dist = d;
                     best_coord = ntc.xy;
                 }
             }
          }      
	    out_color = vec4(best_coord, 0.0, 1.0);
  } else if (pass_index == 14) {
    // finish
	  out_color =  mix(matte_r, dist, dist.a);
  } 
}

