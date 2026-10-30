"""GLSL sources of the 2-D viewer, and of the 3-D viewer's display pass.

Rays are screen-space quads with a Gaussian cross-section, blended additively (ONE, ONE) into a float
framebuffer in linear radiometric units; a fullscreen pass then tone-maps the energy for display. The view
enters only through uniforms, so pan and zoom never re-tessellate, and the quad's perpendicular is taken in
pixels, so a stroke has the same width at every orientation.
"""

_PREAMBLE = """
#ifdef GL_ES
precision highp float;
precision mediump int;
#endif
"""

# A segment widened into a quad in pixel space, (t along it, side) given per corner.
_QUAD = """
uniform vec2  u_viewport;      // physical pixels
uniform vec2  u_view_center;   // world (scene-relative)
uniform float u_ppw;           // pixels per world unit
uniform float u_width_px;

attribute vec2 a_start;        // world, scene-relative
attribute vec2 a_end;
attribute vec2 a_corner;       // (t along segment {0,1}, side {-1,+1})

varying vec2 v_start_px;
varying vec2 v_end_px;
varying vec2 v_pos_px;

void place_quad(float extent) {
    vec2 half_vp = 0.5 * u_viewport;
    vec2 s = (a_start - u_view_center) * u_ppw + half_vp;
    vec2 e = (a_end   - u_view_center) * u_ppw + half_vp;
    vec2 dir = e - s;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 perp = vec2(-dir.y, dir.x);
    vec2 pos = mix(s, e, a_corner.x) + perp * (a_corner.y * extent) + dir * ((a_corner.x * 2.0 - 1.0) * extent);
    gl_Position = vec4(pos / u_viewport * 2.0 - 1.0, 0.0, 1.0);
    v_start_px = s;
    v_end_px = e;
    v_pos_px = pos;
}
"""

_DISTANCE = """
varying vec2 v_start_px;
varying vec2 v_end_px;
varying vec2 v_pos_px;

float distance_to_segment() {
    vec2 pa = v_pos_px - v_start_px;
    vec2 ba = v_end_px - v_start_px;
    float denom = dot(ba, ba);
    if (denom < 1e-12) return length(pa);
    return length(pa - clamp(dot(pa, ba) / denom, 0.0, 1.0) * ba);
}
"""

RAY_VERT = _PREAMBLE + _QUAD + """
uniform float u_sigma_px;
attribute vec4  a_color;       // rgb + per-ray weight in alpha
attribute float a_intensity;
varying vec4  v_color;
varying float v_intensity;

void main() {
    // the full Gaussian support (4 sigma) plus a pixel, so the tail is never clipped
    place_quad(max(0.5 * u_width_px, 4.0 * u_sigma_px) + 1.0);
    v_color = a_color;
    v_intensity = a_intensity;
}
"""

RAY_FRAG = _PREAMBLE + _DISTANCE + """
uniform float u_sigma_px;
uniform float u_weight_scale;
varying vec4  v_color;
varying float v_intensity;

void main() {
    float dist = distance_to_segment();
    float sigma = max(u_sigma_px, 1e-6);
    float energy = v_intensity * v_color.a * u_weight_scale * exp(-0.5 * dist * dist / (sigma * sigma));
    if (energy < 1e-7) discard;
    gl_FragColor = vec4(v_color.rgb * energy, energy);  // linear energy, premultiplied
}
"""

# Crisp rays: plain colour, standard alpha blending, no accumulation.
RAY_SOLID_FRAG = _PREAMBLE + _DISTANCE + """
uniform float u_width_px;
uniform float u_weight_scale;
varying vec4  v_color;
varying float v_intensity;

void main() {
    float edge = 1.0 - smoothstep(0.5 * u_width_px - 0.75, 0.5 * u_width_px + 0.75, distance_to_segment());
    float alpha = clamp(edge * v_color.a * v_intensity * u_weight_scale, 0.0, 1.0);
    if (alpha < 1e-3) discard;
    gl_FragColor = vec4(v_color.rgb, alpha);
}
"""

LINE_VERT = _PREAMBLE + _QUAD + """
void main() {
    place_quad(0.5 * u_width_px + 1.0);
}
"""

LINE_FRAG = _PREAMBLE + _DISTANCE + """
uniform vec4  u_color;
uniform float u_width_px;

void main() {
    float alpha = 1.0 - smoothstep(0.5 * u_width_px - 0.75, 0.5 * u_width_px + 0.75, distance_to_segment());
    if (alpha < 1e-3) discard;
    gl_FragColor = vec4(u_color.rgb, u_color.a * alpha);
}
"""

# A world point placed in the view (markers, filled triangles).
_PLACE = """
uniform vec2  u_viewport;
uniform vec2  u_view_center;
uniform float u_ppw;
attribute vec2 a_position;     // world, scene-relative

vec4 place() {
    vec2 pos = (a_position - u_view_center) * u_ppw + 0.5 * u_viewport;
    return vec4(pos / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}
"""

MARKER_VERT = _PREAMBLE + _PLACE + """
uniform float u_size_px;

void main() {
    gl_Position = place();
    gl_PointSize = u_size_px;
}
"""

FILL_VERT = _PREAMBLE + _PLACE + """
void main() {
    gl_Position = place();
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

FILL_FRAG = _PREAMBLE + """
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
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
    vec3 mapped = u_mode == 0 ? 1.0 - exp(-c) : (u_mode == 1 ? c / (1.0 + c) : clamp(c, 0.0, 1.0));
    gl_FragColor = vec4(u_background + pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
"""

# The 3-D viewer's display map: identity below the knee K (lenses and beam bodies keep their linear look,
# brightness proportional to ray count), smooth compression above (foci glow without flattening).
SOFT_KNEE_FRAG = _PREAMBLE + """
uniform sampler2D u_tex;
varying vec2 v_uv;
const float K = 0.9;

void main() {
    vec3 c = texture2D(u_tex, v_uv).rgb;
    vec3 soft = K + (1.0 - K) * (1.0 - exp(-max(c - K, 0.0) / (1.0 - K)));
    gl_FragColor = vec4(mix(c, soft, step(K, c)), 1.0);
}
"""
