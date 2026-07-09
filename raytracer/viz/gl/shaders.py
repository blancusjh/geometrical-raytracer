"""GLSL sources for the HDR ray-accumulation pipeline.

Rays are screen-space quads with a Gaussian cross-section, blended
additively (ONE, ONE) into a float framebuffer in *linear* radiometric
units. A fullscreen tone-mapping pass then compresses the accumulated
energy for display. The view enters only through uniforms, so pan/zoom
never re-tessellates geometry, and the perpendicular is computed in pixel
space, making stroke width isotropic for every orientation.
"""

_PREAMBLE = """
#ifdef GL_ES
precision highp float;
precision mediump int;
#endif
"""

RAY_VERT = _PREAMBLE + """
uniform vec2  u_viewport;      // physical pixels
uniform vec2  u_view_center;   // world (scene-relative)
uniform float u_ppw;           // pixels per world unit
uniform float u_width_px;
uniform float u_sigma_px;

attribute vec2  a_start;       // world, scene-relative
attribute vec2  a_end;
attribute vec2  a_corner;      // (t along segment {0,1}, side {-1,+1})
attribute vec4  a_color;       // rgb + per-ray weight in alpha
attribute float a_intensity;

varying vec2  v_start_px;
varying vec2  v_end_px;
varying vec2  v_pos_px;
varying vec4  v_color;
varying float v_intensity;

void main() {
    vec2 half_vp = 0.5 * u_viewport;
    vec2 s = (a_start - u_view_center) * u_ppw + half_vp;
    vec2 e = (a_end   - u_view_center) * u_ppw + half_vp;

    vec2 dir = e - s;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 perp = vec2(-dir.y, dir.x);

    // Expand to the full Gaussian support (4 sigma) so the tail is never
    // clipped by the quad; +1 px guard for antialiasing.
    float extent = max(0.5 * u_width_px, 4.0 * u_sigma_px) + 1.0;
    vec2 pos = mix(s, e, a_corner.x)
             + perp * (a_corner.y * extent)
             + dir * ((a_corner.x * 2.0 - 1.0) * extent);

    gl_Position = vec4(pos / u_viewport * 2.0 - 1.0, 0.0, 1.0);
    v_start_px = s;
    v_end_px = e;
    v_pos_px = pos;
    v_color = a_color;
    v_intensity = a_intensity;
}
"""

RAY_FRAG = _PREAMBLE + """
uniform float u_sigma_px;
uniform float u_weight_scale;

varying vec2  v_start_px;
varying vec2  v_end_px;
varying vec2  v_pos_px;
varying vec4  v_color;
varying float v_intensity;

float point_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float denom = dot(ba, ba);
    if (denom < 1e-12) return length(pa);
    float t = clamp(dot(pa, ba) / denom, 0.0, 1.0);
    return length(pa - t * ba);
}

void main() {
    float dist = point_to_segment(v_pos_px, v_start_px, v_end_px);
    float sigma = max(u_sigma_px, 1e-6);
    float profile = exp(-0.5 * dist * dist / (sigma * sigma));
    float energy = v_intensity * v_color.a * u_weight_scale * profile;
    if (energy < 1e-7) discard;
    // Linear radiometric accumulation: rgb premultiplied by energy.
    gl_FragColor = vec4(v_color.rgb * energy, energy);
}
"""

RAY_SOLID_FRAG = _PREAMBLE + """
uniform float u_width_px;
uniform float u_weight_scale;

varying vec2  v_start_px;
varying vec2  v_end_px;
varying vec2  v_pos_px;
varying vec4  v_color;
varying float v_intensity;

float point_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float denom = dot(ba, ba);
    if (denom < 1e-12) return length(pa);
    float t = clamp(dot(pa, ba) / denom, 0.0, 1.0);
    return length(pa - t * ba);
}

void main() {
    float dist = point_to_segment(v_pos_px, v_start_px, v_end_px);
    float edge = 1.0 - smoothstep(0.5 * u_width_px - 0.75, 0.5 * u_width_px + 0.75, dist);
    float alpha = clamp(edge * v_color.a * v_intensity * u_weight_scale, 0.0, 1.0);
    if (alpha < 1e-3) discard;
    // Plain (non-premultiplied) color, standard alpha blend, no HDR
    // accumulation: each ray reads as a crisp, non-glowing physical line.
    gl_FragColor = vec4(v_color.rgb, alpha);
}
"""

TONEMAP_VERT = _PREAMBLE + """
attribute vec2 a_position;     // fullscreen triangle in NDC
varying vec2 v_uv;
void main() {
    v_uv = 0.5 * (a_position + 1.0);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

TONEMAP_FRAG = _PREAMBLE + """
uniform sampler2D u_accum;
uniform float u_exposure;
uniform int   u_mode;          // 0 exponential, 1 reinhard, 2 linear
uniform vec3  u_background;

varying vec2 v_uv;

void main() {
    vec3 c = texture2D(u_accum, v_uv).rgb * u_exposure;
    vec3 mapped;
    if (u_mode == 0) {
        mapped = 1.0 - exp(-c);
    } else if (u_mode == 1) {
        mapped = c / (1.0 + c);
    } else {
        mapped = clamp(c, 0.0, 1.0);
    }
    mapped = pow(mapped, vec3(1.0 / 2.2));
    gl_FragColor = vec4(u_background + mapped, 1.0);
}
"""

LINE_VERT = _PREAMBLE + """
uniform vec2  u_viewport;
uniform vec2  u_view_center;
uniform float u_ppw;
uniform float u_width_px;

attribute vec2 a_start;
attribute vec2 a_end;
attribute vec2 a_corner;

varying vec2 v_start_px;
varying vec2 v_end_px;
varying vec2 v_pos_px;

void main() {
    vec2 half_vp = 0.5 * u_viewport;
    vec2 s = (a_start - u_view_center) * u_ppw + half_vp;
    vec2 e = (a_end   - u_view_center) * u_ppw + half_vp;
    vec2 dir = e - s;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 perp = vec2(-dir.y, dir.x);
    float extent = 0.5 * u_width_px + 1.0;
    vec2 pos = mix(s, e, a_corner.x)
             + perp * (a_corner.y * extent)
             + dir * ((a_corner.x * 2.0 - 1.0) * extent);
    gl_Position = vec4(pos / u_viewport * 2.0 - 1.0, 0.0, 1.0);
    v_start_px = s;
    v_end_px = e;
    v_pos_px = pos;
}
"""

LINE_FRAG = _PREAMBLE + """
uniform vec4  u_color;
uniform float u_width_px;

varying vec2 v_start_px;
varying vec2 v_end_px;
varying vec2 v_pos_px;

float point_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float denom = dot(ba, ba);
    if (denom < 1e-12) return length(pa);
    float t = clamp(dot(pa, ba) / denom, 0.0, 1.0);
    return length(pa - t * ba);
}

void main() {
    float dist = point_to_segment(v_pos_px, v_start_px, v_end_px);
    float alpha = 1.0 - smoothstep(0.5 * u_width_px - 0.75, 0.5 * u_width_px + 0.75, dist);
    if (alpha < 1e-3) discard;
    gl_FragColor = vec4(u_color.rgb, u_color.a * alpha);
}
"""

MARKER_VERT = _PREAMBLE + """
uniform vec2  u_viewport;
uniform vec2  u_view_center;
uniform float u_ppw;
uniform float u_size_px;

attribute vec2 a_position;

void main() {
    vec2 half_vp = 0.5 * u_viewport;
    vec2 pos = (a_position - u_view_center) * u_ppw + half_vp;
    gl_Position = vec4(pos / u_viewport * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = u_size_px;
}
"""

MARKER_FRAG = _PREAMBLE + """
uniform vec4 u_color;

void main() {
    vec2 offset = gl_PointCoord - vec2(0.5);
    if (dot(offset, offset) > 0.25) discard;
    gl_FragColor = u_color;
}
"""

FILL_VERT = _PREAMBLE + """
uniform vec2  u_viewport;
uniform vec2  u_view_center;
uniform float u_ppw;

attribute vec2 a_position;     // world, scene-relative

void main() {
    vec2 half_vp = 0.5 * u_viewport;
    vec2 pos = (a_position - u_view_center) * u_ppw + half_vp;
    gl_Position = vec4(pos / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}
"""

FILL_FRAG = _PREAMBLE + """
uniform vec4 u_color;          // rgb + translucent alpha

void main() {
    gl_FragColor = u_color;
}
"""

__all__ = [
    "RAY_VERT",
    "RAY_FRAG",
    "RAY_SOLID_FRAG",
    "TONEMAP_VERT",
    "TONEMAP_FRAG",
    "LINE_VERT",
    "LINE_FRAG",
    "MARKER_VERT",
    "MARKER_FRAG",
    "FILL_VERT",
    "FILL_FRAG",
]
