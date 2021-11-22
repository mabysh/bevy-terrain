#version 450

layout(location = 0) in vec2 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;

layout(location = 0) out vec4 v_Color;

layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
};
layout(set = 1, binding = 0) uniform Transform {
    mat4 Model;
};

layout(set = 2, binding = 0) uniform texture2D HeightMapTexture_texture;
layout(set = 2, binding = 1) uniform sampler HeightMapTexture_texture_sampler;
layout(set = 2, binding = 2) uniform HeightMapTexture_color {
    vec4 color;
};
layout(set = 2, binding = 3) uniform HeightMapTexture_scale {
    float scale;
};
layout(set = 2, binding = 4) uniform HeightMapTexture_middle {
    vec3 middle;
};

void main() {
    vec2 xz = (Model * vec4(Vertex_Position.x, 1.0, Vertex_Position.y, 1.0)).xz;
    vec2 xz_snapped = floor(xz / scale) * scale;

    vec2 uv = (xz_snapped + 0.5) / 2048.0;
    vec4 rgba = texture(sampler2D(HeightMapTexture_texture, HeightMapTexture_texture_sampler), uv);
    float z = ((rgba.g * 65280.0) + (rgba.r * 255.0)) * 0.005;
    // float c = z / 256.0;
    // v_Color = vec4(c, c, c, 1.0);
    v_Color = color;

    gl_Position = ViewProj * vec4(xz_snapped.x, z, xz_snapped.y, 1.0);
}