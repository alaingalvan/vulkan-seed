#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Attributes
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inColor;

// Varyings
layout (location = 0) out vec3 outColor;

out gl_PerVertex 
{
    vec4 gl_Position;   
};


void main() 
{
	outColor = inColor;
	gl_Position = .25 * vec4(inPos.xyz, 1.0);
}
