#version 450
#pragma tweak_shader(version="1.0")

// Credit to enslow: https://www.shadertoy.com/user/enslow

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


layout(location = 0) out vec4 out_color; 

#pragma input(float, name=seed, default=0.1, min=0.0, max=3.0)
#pragma input(float, name=seed2, default=0.1, min=0.0, max=3.0)
#pragma input(float, name=radius, default=0.2, min=0.0)
#pragma input(float, name=speed, default = 1.0, min=0.0)
#pragma input(float, name=max_dist, default = 1.0, min=0.0)
#pragma input(color, name=line_color, default=[0.0, 1.0, 0.0, 1.0])
#pragma input(color, name=point_color, default=[0.0, 1.0, 0.0, 1.0])
#pragma input(point, name=offset, max=[1.0, 1.0], min=[-1.0, -1.0])
layout(set = 0, binding = 1) uniform CustomInput {
    float seed;       
    float seed2;       
    float radius;       
    float max_dist;       
    float speed;       
    vec4 line_color;       
    vec4 point_color;       
    vec2 offset;
};

#define num 10

//Segment function credit: https://www.shadertoy.com/view/MlcGDB
//User: gPlatl

struct circle
{
    float r;
	vec2 p;
};
    
float segment(vec2 P, vec2 A, vec2 B, float r) 
{
    vec2 g = B - A;
    vec2 h = P - A;
    float d = length(h - g * clamp(dot(g, h) / dot(g,g), 0.0, 1.0));
	return smoothstep(r, 0.5*r, d);
}
    
    
float hash1(int x)
{
    return sign(sin(float(x)*432.))*fract(sin(float(x)*seed));
}
float hash2(int x)
{
    return sign(sin(float(x)*273.))*fract(sin(float(x)*seed2));
}

void main()
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = 5.*(gl_FragCoord.xy-0.5*resolution.xy)/resolution.y;
    
    vec3 col = vec3(0.);    
    float t = time*speed/3.;
    
    
    circle v[num];
    for (int i=0;i<num;i++)
    {
        v[i].r = radius;
        v[i].p = vec2( cos(t+float(i*24)),sin(t+float((i)*32)) ) + vec2(hash1(i),hash2(i)) + vec2(offset.x, -offset.y);
        for (int j=0;j<i;j++)
        {
            float d = distance(v[i].p,v[j].p);
            if (d > max_dist)
            {
                continue;
            }
            float intensity = segment(uv,v[i].p,v[j].p,0.01)*(-exp(d-max_dist)+1.);
            col = col+(line_color.rgb * intensity);
        }
        if (length(v[i].p-uv) < v[i].r)
        {
            col = col+point_color.rgb;
        }
    }
    
    
    // Output to screen
    out_color = vec4(col, length(col));
}
