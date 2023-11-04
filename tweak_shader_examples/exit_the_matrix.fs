#version 450

// Original shader by Kali: https://www.shadertoy.com/user/Kali
// Slightly modified for tweakability.

#pragma tweak_shader(version=1.0)

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float iTime;       
    float iTimeDelta; 
    float iFrameRate;
    uint iFrameIndex; 
    vec4 iMouse;       
    vec4 iDate;       
    vec3 iResolution;  
    uint iPassIndex;   
};

#pragma input(color, name=cube_tint, default=[0.8, 0.3, 0.0, 1.0])
#pragma input(color, name=bg_tint, default=[0.341, 0.77, 0.341, 1.0])
#pragma input(int, name=steps, default=80, min=0, max=200)
#pragma input(float, name=time_scale, default=7, min=0.1, max=30)
layout(binding=0) uniform custom_inputs {
    int steps;
    float time_scale;
    vec4 cube_tint;
    vec4 bg_tint;
};


layout(location = 0) out vec4 out_color; 

float det=.001, t;
vec3 adv;

float hash(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}


mat2 rot(float a)
{
    float s=sin(a), c=cos(a);
    return mat2(c,s,-s,c);
}

vec3 path(float t)
    {
    vec3 p=vec3(vec2(sin(t*.1),cos(t*.05))*10.,t);
    p.x+=smoothstep(.0,.5,abs(.5-fract(t*.02)))*10.;
    return p;
}

float fractal(vec2 p)
{
    p=abs(5.-mod(p*.2,10.))-5.;
    float ot=1000.;
    for (int i=0; i<7; i++)
    {
        p=abs(p)/clamp(p.x*p.y,.25,2.)-1.;
        if(i>0)ot=min(ot,abs(p.x)+.7*fract(abs(p.y)*.05+t*.05+float(i)*.3));
        
    }
    ot=exp(-10.*ot);
    return ot;
}

float box(vec3 p, vec3 l)
{
    vec3 c=abs(p)-l;
    return length(max(vec3(0.),c))+min(0.,max(c.x,max(c.y,c.z)));
}

float de(vec3 p, inout float boxhit, inout vec3 boxp)
{
    boxhit=0.;
    vec3 p2=p-adv;
    p2.xz*=rot(t*.2);
    p2.xy*=rot(t*.1);
    p2.yz*=rot(t*.15);
    float b=box(p2,vec3(1.));
    p.xy-=path(p.z).xy;
    float s=sign(p.y);
    p.y=-abs(p.y)-3.;
    p.z=mod(p.z,20.)-10.;
    for (int i=0; i<5; i++)
    {
        p=abs(p)-1.;
        p.xz*=rot(radians(s*-45.));
        p.yz*=rot(radians(90.));
    }
    float f=-box(p,vec3(5.,5.,10.));
    float d=min(f,b);
    if (d==b) { boxp=p2; boxhit=1.; };
    return d*.7;
}


vec3 march(vec3 from, vec3 dir)
{
    vec2 fragCoord = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
    vec3 p,n,g=vec3(0.);
    float d, td=0.;
    float boxhit = 0.0;
    vec3 boxp = vec3(0.); 
    for (int i=0; i<steps; i++)
    {
        p=from+td*dir;
        d=de(p, boxhit, boxp)*(1.-hash(fragCoord.xy+t)*.3);
        if (d<det && boxhit<.5) break;
        td+=max(det,abs(d));
        float f=fractal(p.xy)+fractal(p.xz)+fractal(p.yz);
        //boxp*=.5;
        float b=fractal(boxp.xy)+fractal(boxp.xz)+fractal(boxp.yz);
        vec3 colf=vec3(f*f,f,f*f*f);
        vec3 colb=vec3(b+.1,b*b+.05,0.);
        g+=(length(colf)*bg_tint.rgb)/(3.+d*d*2.)*exp(-.0015*td*td)*step(5.,td)/2.*(1.-boxhit);
        g+=(length(colb))*cube_tint.rgb/(10.+d*d*20.)*boxhit*.5;
    }
    return g;
}

mat3 lookat(vec3 dir, vec3 up) 
{
	dir=normalize(dir);vec3 rt=normalize(cross(dir,normalize(up)));
    return mat3(rt,cross(rt,dir),dir);
}


void main()
{
    vec2 fragCoord = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
    vec2 uv = (fragCoord-iResolution.xy*.5)/iResolution.y;
    t=iTime*time_scale;
    vec3 from=path(t);
    adv=path(t+6.+sin(t*.1)*3.);
    vec3 dir=normalize(vec3(uv,.7));
    dir=lookat(adv-from,vec3(0.,1.,0.))*dir;
    vec3 col=march(from, dir);
    out_color=vec4(col,1.0);
}
