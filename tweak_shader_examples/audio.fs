#version 450
#pragma tweak_shader(version=1.0)

// Credit to VidVox

#pragma utility_block(shaderinputs)
layout(push_constant) uniform shaderinputs {
    float time;       // shader playback time (in seconds)
    float time_delta; // elapsed time since last frame in secs
    float frame_rate; // number of frames per second estimates
    uint frame_index; // frame count
    vec4 mouse;       // xy is last mouse down position,  abs(zw) is current mouse, sign(z) > 0.0 is mouse_down, sign(w) > 0.0 is click_down event
    vec4 date;        // [year, month, day, seconds]
    vec3 resolution;  // viewport resolution in pixels, [w, h, w/h]
    uint pass_index;  // updated to reflect render pass
};

#pragma input(color, name=topColor, default=[0.0, 0.0, 0.0, 1.0])
#pragma input(color, name=bottomColor, default=[0.0, 0.5, 0.8999, 1.0])
#pragma input(color, name=strokeColor, default=[0.25, 0.25, 0.25, 1.0])
#pragma input(float, name=minRange, min=0.0, max=1.0, default=0.1)
#pragma input(float, name=maxRange, min=0.0, max=1.0, default=0.50)
#pragma input(float, name=gain, min=0.0, max=1.0, default=0.13)
#pragma input(float, name=strokeSize, min=0.0, max=0.25, default=0.050)

layout(set = 1, binding = 0) uniform CustomInput {
  vec4  topColor;
  vec4  bottomColor;
  vec4  strokeColor;
  float minRange;
  float maxRange;
  float strokeSize;
  float gain;
};

#pragma input(audiofft, name="audioFFT", path="./audio.mp3")
layout(set=1, binding=1) uniform sampler default_sampler;
layout(set=1, binding=2) uniform texture2D audioFFT;

layout(location = 0) out vec4 out_color; 

void main() {
	
	vec2 loc = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y)  / resolution.xy;
	
	//	the fftImage is 256 steps
	loc.x = loc.x * abs(maxRange - minRange) + minRange;
	
	vec4 fft = texture(sampler2D(audioFFT, default_sampler), vec2(loc.x,0.5));

	float fftVal = gain * (fft.r + fft.g + fft.b) / 3.0;
	if (loc.y > fftVal) {
		fft = topColor;
	} else {
		fft = bottomColor;
  }
	if ((strokeSize > 0.0) && (abs(fftVal - loc.y) < strokeSize))	{
		fft = mix(strokeColor, fft, abs(fftVal - loc.y) / strokeSize);
	}
	
	out_color= fft;
}
