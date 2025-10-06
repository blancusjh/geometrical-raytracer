"""Custom AGG line visual with controllable alpha falloff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from vispy import gloo
from vispy.color import ColorArray
from vispy.scene.visuals import create_visual_node
from vispy.visuals.visual import CompoundVisual, Visual

from vispy.visuals.line.dash_atlas import DashAtlas

VERT_SHADER = """
const float THETA = 15.0 * 3.14159265358979323846264/180.0;

float cross(in vec2 v1, in vec2 v2) {
    return v1.x*v2.y - v1.y*v2.x;
}

float signed_distance(in vec2 v1, in vec2 v2, in vec2 v3) {
    return cross(v2-v1,v1-v3) / length(v2-v1);
}

void rotate( in vec2 v, in float alpha, out vec2 result ) {
    float c = cos(alpha);
    float s = sin(alpha);
    result = vec2( c*v.x - s*v.y,
                   s*v.x + c*v.y );
}

vec2 transform_vector(vec2 x, vec2 base) {
    vec4 o = $transform(vec4(base, 0, 1));
    return ($transform(vec4(base+x, 0, 1)) - o).xy;
}

attribute vec4 color;
uniform float linewidth;
uniform float antialias;
uniform vec2 linecaps;
uniform float linejoin;
uniform float miter_limit;
attribute float alength;
uniform float dash_phase;
uniform float dash_period;
uniform float dash_index;
uniform vec2 dash_caps;
uniform float closed;

attribute vec2 a_position;
attribute vec4 a_tangents;
attribute vec2 a_segment;
attribute vec2 a_angles;
attribute vec2 a_texcoord;

varying vec4  v_color;
varying vec2  v_segment;
varying vec2  v_angles;
varying vec2  v_linecaps;
varying vec2  v_texcoord;
varying vec2  v_miter;
varying float v_miter_limit;
varying float v_length;
varying float v_linejoin;
varying float v_linewidth;
varying float v_antialias;
varying float v_dash_phase;
varying float v_dash_period;
varying float v_dash_index;
varying vec2  v_dash_caps;
varying float v_closed;

void main()
{
    v_color = color;

    v_linewidth = linewidth;
    v_antialias = antialias;
    v_linecaps  = linecaps;

    v_linejoin    = linejoin;
    v_miter_limit = miter_limit;
    v_length      = alength;
    v_dash_phase  = dash_phase;

    v_dash_period = dash_period;
    v_dash_index  = dash_index;
    v_dash_caps   = dash_caps;

    v_closed = closed;
    bool closed = (v_closed > 0.0);

    v_angles  = a_angles;
    v_segment = a_segment;

    v_color.a = min(v_linewidth, v_color.a);
    v_linewidth = max(v_linewidth, 1.0);

    float w = ceil(1.25*v_antialias+v_linewidth)/2.0;

    vec4 doc_pos = $transform(vec4(a_position,0.,1.));
    vec2 position = doc_pos.xy;

    vec2 t1 = normalize(transform_vector(a_tangents.xy, a_position));
    vec2 t2 = normalize(transform_vector(a_tangents.zw, a_position));
    float u = a_texcoord.x;
    float v = a_texcoord.y;
    vec2 o1 = vec2( +t1.y, -t1.x);
    vec2 o2 = vec2( +t2.y, -t2.x);

    if( t1 != t2 ) {
        float angle  = atan (t1.x*t2.y-t1.y*t2.x, t1.x*t2.x+t1.y*t2.y);
        vec2 t  = normalize(t1+t2);
        vec2 o  = vec2( + t.y, - t.x);

        if ( v_dash_index > 0.0 ) {
            if( (abs(angle) > THETA) ) {
                position += v * w * o / cos(angle/2.0);
                if( angle < 0.0 ) {
                    if( u == +1.0 ) {
                        u = v_segment.y + v * w * tan(angle/2.0);
                        if( v == 1.0 ) {
                            position -= 2.0 * w * t1 / sin(angle);
                            u -= 2.0 * w / sin(angle);
                        }
                    } else {
                        u = v_segment.x - v * w * tan(angle/2.0);
                        if( v == 1.0 ) {
                            position += 2.0 * w * t2 / sin(angle);
                            u += 2.0*w / sin(angle);
                        }
                    }
                } else {
                    if( u == +1.0 ) {
                        u = v_segment.y + v * w * tan(angle/2.0);
                        if( v == -1.0 ) {
                            position += 2.0 * w * t1 / sin(angle);
                            u += 2.0 * w / sin(angle);
                        }
                    } else {
                        u = v_segment.x - v * w * tan(angle/2.0);
                        if( v == -1.0 ) {
                            position -= 2.0 * w * t2 / sin(angle);
                            u -= 2.0*w / sin(angle);
                        }
                    }
                }
            } else {
                position += v * w * o / cos(angle/2.0);
                if( u == +1.0 ) u = v_segment.y;
                else            u = v_segment.x;
            }
        }
        else {
            position.xy += v * w * o / cos(angle/2.0);
            if( angle < 0.0 ) {
                if( u == +1.0 ) {
                    u = v_segment.y + v * w * tan(angle/2.0);
                } else {
                    u = v_segment.x - v * w * tan(angle/2.0);
                }
            } else {
                if( u == +1.0 ) {
                    u = v_segment.y + v * w * tan(angle/2.0);
                } else {
                    u = v_segment.x - v * w * tan(angle/2.0);
                }
            }
        }
    } else {
        position += v * w * o1;
        if( u == -1.0 ) {
            u = v_segment.x - w;
            position -=  w * t1;
        } else {
            u = v_segment.y + w;
            position +=  w * t2;
        }
    }

    vec2 t;
    vec2 curr = $transform(vec4(a_position,0.,1.)).xy;
    if( a_texcoord.x < 0.0 ) {
        vec2 next = curr + t2*(v_segment.y-v_segment.x);

        rotate( t1, +a_angles.x/2.0, t);
        v_miter.x = signed_distance(curr, curr+t, position);

        rotate( t2, +a_angles.y/2.0, t);
        v_miter.y = signed_distance(next, next+t, position);
    } else {
        vec2 prev = curr - t1*(v_segment.y-v_segment.x);

        rotate( t1, -a_angles.x/2.0,t);
        v_miter.x = signed_distance(prev, prev+t, position);

        rotate( t2, -a_angles.y/2.0,t);
        v_miter.y = signed_distance(curr, curr+t, position);
    }

    if (!closed && v_segment.x <= 0.0) {
        v_miter.x = 1e10;
    }
    if (!closed && v_segment.y >= v_length)
    {
        v_miter.y = 1e10;
    }

    v_texcoord = vec2( u, v*w );

    gl_Position = $px_ndc_transform($doc_px_transform(vec4(position, doc_pos.z, 1.0)));
}
"""

FRAG_SHADER = """
#include "math/constants.glsl"

const float THETA = 15.0 * 3.14159265358979323846264/180.0;

float cap(int type, float dx, float dy, float t)
{
    float d = 0.0;
    dx = abs(dx);
    dy = abs(dy);

    if (type == 0) discard;
    else if (type == 1)  d = sqrt(dx*dx+dy*dy);
    else if (type == 3)  d = (dx+abs(dy));
    else if (type == 2)  d = max(abs(dy),(t+dx-abs(dy)));
    else if (type == 4)  d = max(dx,dy);
    else if (type == 5)  d = max(dx+t,dy);

    return d;
}

float join(int type, float d, vec2 segment, vec2 texcoord,
           vec2 miter, float miter_limit, float linewidth)
{
    float dx = texcoord.x;

    if (type == 1) {
        if (dx < segment.x) {
            d = max(d,length(texcoord - vec2(segment.x,0.0)));
        } else if (dx > segment.y) {
            d = max(d,length(texcoord - vec2(segment.y,0.0)));
        }
    } else if (type == 2) {
        if ((dx < segment.x) ||  (dx > segment.y))
            d = max(d, min(abs(miter.x),abs(miter.y)));
    }

    if ((dx < segment.x) ||  (dx > segment.y)) {
        d = max(d, min(abs(miter.x),
                       abs(miter.y)) - miter_limit*linewidth/2.0);
    }

    return d;
}

uniform sampler2D u_dash_atlas;
uniform float alpha_floor;
uniform float alpha_softness;

varying vec4  v_color;
varying vec2  v_segment;
varying vec2  v_angles;
varying vec2  v_linecaps;
varying vec2  v_texcoord;
varying vec2  v_miter;
varying float v_miter_limit;
varying float v_length;
varying float v_linejoin;
varying float v_linewidth;
varying float v_antialias;
varying float v_dash_phase;
varying float v_dash_period;
varying float v_dash_index;
varying vec2  v_dash_caps;
varying float v_closed;

void main()
{
    if (v_color.a <= 0.0) {
        discard;
    }

    float dx = v_texcoord.x;
    float dy = v_texcoord.y;
    float t = v_linewidth/2.0-v_antialias;
    float d = abs(dy);

    if ((v_closed <= 0.0) && (dx < 0.0)) {
        d = cap(int(v_linecaps.x), abs(dx), abs(dy), t);
    } else if ((v_closed <= 0.0) && (dx > v_length)) {
        d = cap(int(v_linecaps.y), abs(dx)-v_length, abs(dy), t);
    } else {
        d = join(int(v_linejoin), d, v_segment, v_texcoord,
                 v_miter, v_miter_limit, v_linewidth);
    }

    d = d - t;
    float alpha;
    if (d < 0.0) {
        alpha = v_color.a;
    } else {
        if (v_antialias <= 0.0) {
            alpha = 0.0;
        } else {
            float falloff = exp(- (d / v_antialias) * (d / v_antialias));
            float softness = clamp(alpha_softness, 0.0, 1.0);
            alpha = mix(v_color.a, falloff * v_color.a, softness);
        }
    }

    alpha = max(alpha, alpha_floor);
    if (alpha <= 0.0) {
        discard;
    }
    gl_FragColor = vec4(v_color.rgb, alpha);
}
"""

JOINS = {"miter": 0, "round": 1, "bevel": 2}
CAPS = {"none": 0, "round": 1, "triangle_in": 2, "triangle_out": 3, "square": 4, "butt": 5}


@dataclass
class _ParentState:
    pos: np.ndarray | None = None
    color: np.ndarray | None = None
    width: float = 1.0
    antialias: float = 1.0
    alpha_floor: float = 0.0
    alpha_softness: float = 0.0
    changed_pos: bool = False
    changed_color: bool = False
    segment_lengths: np.ndarray | None = None


class _ConstantAlphaAggLineVisual(Visual):
    """Low-level renderer for anti-aliased poly-lines without endpoint fade."""

    _agg_vtype = np.dtype([
        ("a_position", np.float32, (2,)),
        ("a_tangents", np.float32, (4,)),
        ("a_segment", np.float32, (2,)),
        ("a_angles", np.float32, (2,)),
        ("a_texcoord", np.float32, (2,)),
        ("alength", np.float32),
        ("color", np.float32, (4,)),
    ])

    def __init__(self, parent_state: _ParentState) -> None:
        self._state = parent_state
        self._vbo = gloo.VertexBuffer()

        self._dash_data = DashAtlas()
        dash_index, dash_period = self._dash_data["solid"]
        self._dash_texture = gloo.Texture2D(self._dash_data._data)

        Visual.__init__(self, vcode=VERT_SHADER, fcode=FRAG_SHADER)
        self._index_buffer = gloo.IndexBuffer()
        self.set_gl_state("translucent", depth_test=False)
        self._draw_mode = "triangles"

        self._uniform_defaults = dict(
            closed=False,
            miter_limit=4.0,
            dash_phase=0.0,
            linewidth=parent_state.width,
            antialias=parent_state.antialias,
            linejoin=JOINS["round"],
            linecaps=(CAPS["round"], CAPS["round"]),
            dash_caps=(CAPS["round"], CAPS["round"]),
            dash_index=dash_index,
            dash_period=dash_period,
        )

    def _prepare_transforms(self, view) -> None:  # pragma: no cover
        data_doc = view.get_transform("visual", "document")
        doc_px = view.get_transform("document", "framebuffer")
        px_ndc = view.get_transform("framebuffer", "render")
        vert = view.view_program.vert
        vert["transform"] = data_doc
        vert["doc_px_transform"] = doc_px
        vert["px_ndc_transform"] = px_ndc

    def _prepare_draw(self, view) -> None:  # pragma: no cover
        state = self._state

        if state.changed_pos:
            if state.pos is None:
                return False
            self._pos = np.ascontiguousarray(state.pos, dtype=np.float32)
        if state.changed_color:
            if state.color is None:
                return False
            self._color = np.ascontiguousarray(state.color, dtype=np.float32)

        if state.changed_pos or state.changed_color:
            V, idxs = self._agg_bake(self._pos, self._color, state.segment_lengths)
            self._vbo.set_data(V)
            self._index_buffer.set_data(idxs)
            state.changed_pos = False
            state.changed_color = False

        self.shared_program.bind(self._vbo)

        uniforms = dict(self._uniform_defaults)
        uniforms.update(
            linewidth=state.width,
            antialias=state.antialias,
            alpha_floor=state.alpha_floor,
            alpha_softness=state.alpha_softness,
        )
        for name, value in uniforms.items():
            self.shared_program[name] = value

        self.shared_program["u_dash_atlas"] = self._dash_texture

    @classmethod
    def _agg_bake(cls, vertices: np.ndarray, color: np.ndarray, segment_lengths: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if segment_lengths is None or len(segment_lengths) <= 1:
            return cls._agg_bake_single(vertices, color)

        V_blocks: list[np.ndarray] = []
        idx_blocks: list[np.ndarray] = []
        offset = 0
        cursor = 0
        for seg_len in segment_lengths:
            seg_vertices = vertices[cursor : cursor + seg_len]
            seg_color = color[cursor : cursor + seg_len]
            cursor += seg_len
            V_seg, idx_seg = cls._agg_bake_single(seg_vertices, seg_color)
            idx_blocks.append(idx_seg + offset)
            offset += len(V_seg)
            V_blocks.append(V_seg)

        V = np.concatenate(V_blocks)
        idxs = np.concatenate(idx_blocks)
        return V, idxs.astype(np.uint32)

    @classmethod
    def _agg_bake_single(cls, vertices: np.ndarray, color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(vertices)
        if n < 2:
            raise ValueError("at least two vertices required for a polyline")

        P = np.array(vertices, dtype=float).reshape(n, 2)
        idx = np.arange(n)

        V = np.zeros(len(P), dtype=cls._agg_vtype)
        V["a_position"] = P

        T = P[1:] - P[:-1]
        V["a_tangents"][1:, :2] = T
        V["a_tangents"][0, :2] = T[0]
        V["a_tangents"][:-1, 2:] = T
        V["a_tangents"][-1, 2:] = T[-1]

        A = np.arctan2(
            V["a_tangents"][:, 0] * V["a_tangents"][:, 3] - V["a_tangents"][:, 1] * V["a_tangents"][:, 2],
            V["a_tangents"][:, 0] * V["a_tangents"][:, 2] + V["a_tangents"][:, 1] * V["a_tangents"][:, 3],
        )
        V["a_angles"][:-1, 0] = A[:-1]
        V["a_angles"][:-1, 1] = A[1:]

        N = np.sqrt((T ** 2).sum(axis=1))
        L = np.cumsum(N)
        V["a_segment"][1:, 0] = L
        V["a_segment"][:-1, 1] = L

        V = np.repeat(V, 2, axis=0)[1:-1]
        V["a_segment"][1:] = V["a_segment"][:-1]
        V["a_angles"][1:] = V["a_angles"][:-1]
        V["a_texcoord"][0::2] = -1
        V["a_texcoord"][1::2] = +1
        idx = np.repeat(idx, 2)[1:-1]

        V = np.repeat(V, 2, axis=0)
        V["a_texcoord"][0::2, 1] = -1
        V["a_texcoord"][1::2, 1] = +1
        idx = np.repeat(idx, 2)

        idxs = np.resize(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32), (n - 1) * (2 * 3))
        idxs += np.repeat(4 * np.arange(n - 1, dtype=np.uint32), 6)

        V["alength"] = L[-1] * np.ones(len(V))

        if color.ndim == 1:
            color = np.tile(color, (len(V), 1))
        elif color.ndim == 2 and len(color) == 1:
            color = np.tile(color, (len(V), 1))
        elif color.ndim == 2 and len(color) == len(vertices):
            color = color[idx]
        elif color.ndim == 2 and len(color) == len(V):
            pass
        else:
            raise ValueError("colour array shape incompatible with vertices")
        V["color"] = color

        return V, idxs


class ConstantAlphaLineVisual(CompoundVisual):
    def __init__(
        self,
        pos: Iterable[Iterable[float]] | None = None,
        *,
        color=(1.0, 1.0, 1.0, 1.0),
        width: float = 1.0,
        antialias: float = 1.0,
        alpha_floor: float = 0.0,
        alpha_softness: float = 0.0,
    ) -> None:
        self._state = _ParentState(
            width=float(width),
            antialias=float(antialias),
            alpha_floor=float(alpha_floor),
            alpha_softness=float(np.clip(alpha_softness, 0.0, 1.0)),
        )
        self._line_visual = _ConstantAlphaAggLineVisual(self._state)
        CompoundVisual.__init__(self, [self._line_visual])
        self.set_data(pos=pos, color=color, width=width)

    def set_data(
        self,
        *,
        pos: Iterable[Iterable[float]] | None = None,
        color: Iterable[float] | np.ndarray | None = None,
        width: float | None = None,
    ) -> None:
        total_points = None
        if pos is not None:
            lengths: list[int] = []
            if isinstance(pos, (list, tuple)) and not isinstance(pos, np.ndarray):
                segments = []
                for segment in pos:
                    seg_array = np.asarray(segment, dtype=np.float32)
                    if seg_array.ndim != 2 or seg_array.shape[1] != 2:
                        raise TypeError("each segment must have shape (M, 2)")
                    if seg_array.shape[0] < 2:
                        continue
                    segments.append(seg_array)
                    lengths.append(seg_array.shape[0])
                if not segments:
                    raise ValueError("no valid segments provided")
                array = np.vstack(segments)
            else:
                array = np.asarray(pos, dtype=np.float32)
                if array.ndim != 2 or array.shape[1] != 2:
                    raise TypeError("pos must be (N, 2)")
                lengths.append(array.shape[0])
            self._state.pos = array
            self._state.segment_lengths = np.array(lengths, dtype=np.int32)
            self._state.changed_pos = True
            total_points = array.shape[0]

        if color is not None:
            if isinstance(color, (list, tuple)) and not isinstance(color, np.ndarray):
                if self._state.segment_lengths is None:
                    raise ValueError("segment lengths unknown; set positions before color")
                if len(color) != len(self._state.segment_lengths):
                    raise ValueError("color list must match number of segments")
                color_blocks = []
                for col, seg_len in zip(color, self._state.segment_lengths):
                    rgba = np.asarray(col, dtype=np.float32)
                    if rgba.ndim == 1:
                        if rgba.size == 3:
                            rgba = np.concatenate((rgba, [1.0]))
                        if rgba.size != 4:
                            raise ValueError("segment color must be RGB or RGBA")
                        rgba = np.tile(rgba, (seg_len, 1))
                    elif rgba.ndim == 2:
                        if rgba.shape[0] == seg_len:
                            if rgba.shape[1] == 3:
                                rgba = np.hstack((rgba, np.ones((seg_len, 1), dtype=np.float32)))
                        elif rgba.shape[0] == 1:
                            rgba = np.tile(rgba, (seg_len, 1))
                        else:
                            raise ValueError("segment color array length mismatch")
                        if rgba.shape[1] == 3:
                            rgba = np.hstack((rgba, np.ones((rgba.shape[0], 1), dtype=np.float32)))
                        if rgba.shape[1] != 4:
                            raise ValueError("segment color must have 4 channels")
                    else:
                        raise ValueError("segment color must be 1D or 2D array")
                    color_blocks.append(rgba.astype(np.float32))
                rgba = np.vstack(color_blocks)
            else:
                rgba = ColorArray(color).rgba.astype(np.float32)
                if rgba.ndim == 1:
                    if total_points is None:
                        raise ValueError("set positions before scalar color")
                    rgba = np.tile(rgba, (total_points, 1))
                elif total_points is not None and rgba.shape[0] == 1:
                    rgba = np.tile(rgba, (total_points, 1))
                elif total_points is not None and rgba.shape[0] != total_points:
                    raise ValueError("color array length does not match vertices")
            self._state.color = rgba
            self._state.changed_color = True

        if width is not None:
            self._state.width = float(width)

        self.update()

    def set_alpha_controls(self, *, alpha_floor: float | None = None, alpha_softness: float | None = None) -> None:
        if alpha_floor is not None:
            self._state.alpha_floor = float(alpha_floor)
        if alpha_softness is not None:
            self._state.alpha_softness = float(np.clip(alpha_softness, 0.0, 1.0))
        self.update()


ConstantAlphaLine = create_visual_node(ConstantAlphaLineVisual)
