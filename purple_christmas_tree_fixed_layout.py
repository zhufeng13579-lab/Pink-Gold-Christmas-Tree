#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINK & GOLD STARDUST CHRISTMAS TREE - SOFT HORIZON REVISION | Python 3.10+

A 3D particle scene reconstructed from the supplied clear reference frame:
broad winding pearl-pink ribbons, a translucent tree filled with round jewels,
a golden FIVE-POINT star, and a gold/pink galaxy. No gifts or popups.

This rewrite removes the independent horizontal tree rings. Front/back light
levels, medium round pearls, fine dust and sparse star flares form separate
layers. The inner ground rings no longer dominate the composition.
The existing camera framing and Retina viewport handling are retained.
This revision adds clear medium-sized ground grains and more visible snow;
the approved tree geometry, five-point star and large bokeh are preserved.
The galaxy now feathers into the sky through sparse floating gold/pink dust.
The top star has a wider, slightly overexposed warm-white luminous core.

Install: python3 -m pip install numpy pygame moderngl
Run:     python3 purple_christmas_tree_luxe.py

Controls: mouse-drag = orbit, wheel = zoom, SPACE = toggle auto orbit,
          S = save screenshot, R = reset camera, ESC / Q = exit.

For a slower computer: python3 purple_christmas_tree_luxe.py --quality low
For more particles:   python3 purple_christmas_tree_luxe.py --quality high
For no glow pass:     python3 purple_christmas_tree_luxe.py --no-bloom
CPU smoke test:       python3 purple_christmas_tree_luxe.py --self-test
"""

import argparse
import math
import sys
from datetime import datetime

import numpy as np

WINDOW_W, WINDOW_H = 1280, 780
TREE_HEIGHT = 9.35
QUALITY_SCALE = {"low": 0.54, "medium": 1.0, "high": 1.65}
RIBBON_BOTTOM, RIBBON_TOP = 0.88, 9.50
RIBBON_RADIUS, RIBBON_TURNS = 3.72, 4.85

# Keep the galaxy crisp. The old bloom/exposure washed out the tree.
DEFAULT_EXPOSURE = 0.90
DEFAULT_BLOOM = 0.44
BLOOM_THRESHOLD = 0.57


def _particle_block(rng, positions, colors, sizes, kind, speed=0.0):
    """Pack xyz, RGB, point radius, phase, category, and animation speed."""
    p = np.asarray(positions, dtype="f4").reshape(-1, 3)
    n = len(p)
    c = np.broadcast_to(np.asarray(colors, dtype="f4"), (n, 3))
    s = np.broadcast_to(np.asarray(sizes, dtype="f4"), (n,))
    v = np.broadcast_to(np.asarray(speed, dtype="f4"), (n,))
    phase = rng.uniform(0.0, 2.0 * np.pi, n).astype("f4")
    block = np.empty((n, 10), dtype="f4")
    block[:, :3] = p
    block[:, 3:6] = c
    block[:, 6] = s
    block[:, 7] = phase
    block[:, 8] = float(kind)
    block[:, 9] = v
    return block


def _palette(rng, n, colors, weights=None):
    palette = np.asarray(colors, dtype="f4")
    idx = rng.choice(len(palette), size=n, p=weights)
    jitter = rng.uniform(0.85, 1.15, (n, 1)).astype("f4")
    return palette[idx] * jitter


def helix_path(t, phase=0.0):
    """One continuous tapered ribbon; no independent horizontal tree rings."""
    t = np.asarray(t)
    remaining = np.maximum(1.0-t,0.0)
    r = RIBBON_RADIUS*remaining**.90
    a = 2*np.pi*RIBBON_TURNS*t+.35+phase
    z = RIBBON_BOTTOM+(RIBBON_TOP-RIBBON_BOTTOM)*t
    return r,a,z


def ribbon_width(t):
    t = np.asarray(t)
    remaining = np.maximum(1-t,0)
    width = (.32+.095*np.sin(t*2*np.pi*2.4+.6)**2)*np.sqrt(remaining)
    # The sheet must not cross the cone axis in its final few samples.
    return np.minimum(width,.72*(RIBBON_TOP-RIBBON_BOTTOM)/.90*remaining)


def sample_helix(rng,n,start=0.0,end=1.0):
    """Sample by distance, avoiding excessive light density at the tip."""
    grid = np.linspace(start,end,4097)
    r,a,z = helix_path(grid)
    p = np.column_stack((r*np.cos(a),r*np.sin(a),z))
    cumulative = np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    return np.interp(rng.uniform(0,cumulative[-1],n),cumulative,grid)


def crown_tint(colors,t):
    """Pearl pink transitions gradually to champagne at the very top."""
    f = np.clip((np.asarray(t)-.78)/.17,0,1)
    f = f*f*(3-2*f)
    return colors*(1+f[:,None]*np.array([.18,-.035,-.62],dtype="f4"))


def build_scene(quality="medium",seed=2026):
    """Static geometry for broad glitter ribbons, a pearl-filled tree and a galaxy.

    Layout: xyz, linear HDR RGB, sprite diameter, phase, category, speed.
    The GPU animates these points; no particle geometry is rebuilt per frame.
    """
    if quality not in QUALITY_SCALE:
        raise ValueError("quality must be low, medium, or high")
    q = QUALITY_SCALE[quality]
    rng = np.random.default_rng(seed)
    blocks = []

    def add(p,c,size,kind,speed=0.0):
        blocks.append(_particle_block(rng,p,c,size,kind,speed))

    # Broad, broken orbits. Population increases toward the outer rings;
    # the inner rings are sparse and dim so there is no luminous doughnut.
    rings = np.array([2.0,2.7,3.5,4.4,5.4,6.5,7.7,9.0,10.4,11.9,13.5,15.3],dtype="f4")
    n = round(84_000*q)
    rid = rng.choice(len(rings),n,p=rings**.8/np.sum(rings**.8))
    base = rings[rid]
    a = rng.uniform(0,2*np.pi,n)
    r = base+.075*np.sin(3*a+base*.8)+.04*np.sin(7*a-base)
    r += rng.normal(0,.10+.007*base,n)
    col = _palette(rng,n,[(2.25,1.78,.93),(1.62,1.26,.63),(2.55,2.14,1.30)],
                   weights=[.42,.30,.28])
    pink = (rid%4)==2
    col[pink] = _palette(rng,pink.sum(),[(2.14,1.28,1.75),(1.72,.96,1.49)])
    broken = .30+.70*(.5+.5*np.sin(a*3+base*.76))**.65
    inner = np.clip((base-1.0)/4.3,.18,1.0)
    col *= (broken*inner*rng.uniform(.40,1.10,n))[:,None]
    add(np.column_stack((r*np.cos(a),r*np.sin(a),rng.normal(.02,.035,n))),
        col,rng.uniform(.050,.104,n),0)

    # Small isolated grains cover the whole disk, not just its ring lines.
    n = round(65_000*q)
    r = np.sqrt(rng.uniform(2.0**2,16.6**2,n))
    a = rng.uniform(0,2*np.pi,n)
    col = _palette(rng,n,[(1.48,1.17,.62),(1.90,1.49,.82),(1.65,.91,1.38)],
                   weights=[.40,.33,.27])
    col *= rng.uniform(.16,.62,(n,1))
    add(np.column_stack((r*np.cos(a),r*np.sin(a),rng.normal(.01,.05,n))),
        col,rng.uniform(.044,.096,n),0)

    n = round(30_000*q)
    arm = rng.integers(0,4,n)
    r = rng.uniform(2.5,16,n)
    a = arm*np.pi/2+r*.33+rng.normal(0,.10,n)
    col = _palette(rng,n,[(2.1,1.68,.85),(2.55,2.08,1.16)])
    pink = arm==1
    col[pink] = _palette(rng,pink.sum(),[(2.15,1.17,1.69),(1.85,1.13,1.63)])
    col *= rng.uniform(.18,.68,(n,1))
    add(np.column_stack((r*np.cos(a),r*np.sin(a),rng.normal(.04,.035,n))),
        col,rng.uniform(.050,.102,n),0)

    n = round(5_000*q)
    r = np.sqrt(rng.uniform(2.5**2,16.0**2,n))
    a = rng.uniform(0,2*np.pi,n)
    col = _palette(rng,n,[(3.65,2.95,1.65),(3.45,2.72,1.15),(3.65,2.12,2.83)],
                   weights=[.45,.29,.26])
    add(np.column_stack((r*np.cos(a),r*np.sin(a),rng.uniform(.03,.12,n))),
        col,rng.uniform(.09425,.21025,n),7)

    # A small warm fountain under the tree, not a bright ring or a gift box.
    n = round(950*q)
    r = np.abs(rng.normal(.20,.11,n))
    a = rng.uniform(0,2*np.pi,n)
    p = np.column_stack((r*np.cos(a),r*np.sin(a),rng.normal(.40,.18,n)))
    col = _palette(rng,n,[(2.95,2.06,.95),(3.2,2.5,1.3),(2.5,1.5,.62)])
    add(p,col,rng.uniform(.055,.11,n),7)
    add([[0,0,.35]],(1.12,.68,.26),1.45,6)
    add([[0,0,.39]],(2.4,1.55,.72),.52,6)

    # Thin sheet wound around a cone. Across-band displacement follows the
    # cone slope; its radial thickness is much less than its vertical width.
    n = round(59_000*q)
    t = sample_helix(rng,n)
    r,a,z = helix_path(t)
    w = ribbon_width(t)
    across = rng.uniform(-1,1,n)
    meridian = across*w
    slope = -.90*RIBBON_RADIUS/(RIBBON_TOP-RIBBON_BOTTOM)*np.maximum(1-t,1e-4)**(-.10)
    r += meridian*slope+rng.normal(0,.040*np.sqrt(1-t)+.002,n)
    z += meridian+rng.normal(0,.021*np.sqrt(1-t)+.001,n)
    col = _palette(rng,n,[(2.65,2.30,2.48),(2.40,1.88,2.17),(2.15,1.27,1.69),
                         (2.80,2.58,2.69)],weights=[.32,.30,.18,.20])
    edge_fade = .18+.82*np.maximum(1-across*across,0)**.7
    start_fade = np.clip(t/.12,0,1)
    start_fade = .40+.60*start_fade*start_fade*(3-2*start_fade)
    col *= (edge_fade*start_fade)[:,None]
    col = crown_tint(col,t)
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,rng.uniform(.055,.108,n),1)

    # Fine threads distributed across the sheet, rather than one hard core.
    n = round(21_000*q)
    t = sample_helix(rng,n)
    r,a,z = helix_path(t)
    w = ribbon_width(t)
    across = rng.choice([-.75,-.28,.33,.78],n)+rng.normal(0,.11,n)
    r -= across*w*.38
    r += rng.normal(0,.022*np.sqrt(1-t)+.001,n)
    z += across*w+rng.normal(0,.017*np.sqrt(1-t)+.001,n)
    col = _palette(rng,n,[(3.05,2.86,2.96),(2.80,2.46,2.69),(3.2,2.8,2.96)])
    start_fade = np.clip(t/.12,0,1)
    start_fade = .40+.60*start_fade*start_fade*(3-2*start_fade)
    col *= start_fade[:,None]
    col = crown_tint(col,t)
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,rng.uniform(.048,.092,n),2)

    # A loose train of pink dust below the sheet gives it a feathered edge.
    n = round(24_000*q)
    t = sample_helix(rng,n,0,.94)
    r,a,z = helix_path(t)
    along = rng.beta(1.2,2.0,n)
    drop = (.28+.42*(1-t))*along
    r += drop*.20+rng.normal(0,.12*np.sqrt(1-t),n)
    a += rng.normal(0,.023,n)
    z -= drop+rng.normal(0,.10*np.sqrt(1-t),n)
    col = _palette(rng,n,[(1.65,.76,1.29),(1.80,.98,1.44),(1.25,.77,1.4)])
    col *= rng.uniform(.24,.68,(n,1))
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,rng.uniform(.042,.095,n),8)

    # A short, matching strand makes the final crown join the five-point star.
    n = round(2_100*q)
    t = sample_helix(rng,n,.81,1)
    r,a,z = helix_path(t,np.pi)
    w = ribbon_width(t)
    z += rng.uniform(-.50,.50,n)*w
    r += rng.normal(0,.014*np.sqrt(1-t),n)
    fade = np.clip((t-.81)/.075,0,1)
    col = _palette(rng,n,[(3.35,2.73,1.25),(3.55,2.86,1.42)])
    col *= (.72*fade)[:,None]
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,rng.uniform(.048,.087,n),2)

    # Transparent rose tree volume. These points fill the cone without
    # forming extra outline circles or obscuring the large individual pearls.
    n = round(31_000*q)
    t = rng.beta(1.2,1.65,n)*.96
    maxr,a,z = helix_path(t)
    a = rng.uniform(0,2*np.pi,n)
    r = maxr*np.sqrt(rng.uniform(.035,1,n))
    col = _palette(rng,n,[(1.6,.85,1.38),(1.9,1.04,1.55),(1.12,.78,1.32)])
    col *= rng.uniform(.24,.70,(n,1))
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,rng.uniform(.040,.096,n),3)

    # The medium-sized round pearls are a distinct scale of light, not tiny
    # dust enlarged by bloom. Most have a 2-4 pixel bright core at launch.
    n = round(650*q)
    t = rng.beta(1.20,2.10,n)*.93
    maxr,a,z = helix_path(t)
    a = rng.uniform(0,2*np.pi,n)
    r = maxr*np.sqrt(rng.uniform(.04,.96,n))
    col = _palette(rng,n,[(3.1,2.72,2.97),(3.45,3.14,3.27),(3.2,2.13,2.77),
                         (3.2,2.63,1.6)],weights=[.33,.31,.28,.08])
    size = rng.uniform(.16,.33,n)
    large = rng.random(n)<.055
    size[large] = rng.uniform(.34,.44,large.sum())
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,size,11,rng.uniform(.15,.8,n))

    n = round(150*q)
    t = rng.beta(1.2,1.8,n)*.92
    r,a,z = helix_path(t)
    r *= rng.uniform(.45,1.03,n)
    a = rng.uniform(0,2*np.pi,n)
    col = _palette(rng,n,[(3.7,3.2,3.5),(3.9,2.92,1.6),(3.8,2.3,3.0)])
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),col,rng.uniform(.33,.60,n),10,
        rng.uniform(.2,1.3,n))

    n = round(130*q)
    r = rng.uniform(3.5,15,n)
    a = rng.uniform(0,2*np.pi,n)
    col = _palette(rng,n,[(3.2,2.61,1.48),(3.5,2.78,1.4),(3.5,1.92,2.8)],
                   weights=[.45,.30,.25])
    add(np.column_stack((r*np.cos(a),r*np.sin(a),rng.uniform(.05,.18,n))),
        col,rng.uniform(.24,.48,n),10,rng.uniform(.2,1.2,n))

    # One billboard star; its five-point silhouette and halo are shader-made.
    add([[0,0,9.93]],(1,1,1),1.62,9)

    # Distant grains, visible falling snow, and very few large soft flakes.
    n = round(12_500*q)
    p = np.column_stack((rng.uniform(-45,45,n),rng.uniform(-45,45,n),rng.uniform(-7,33,n)))
    col = _palette(rng,n,[(1.9,1.52,.85),(2.30,1.97,1.28),(2.2,1.25,1.85)],
                   weights=[.43,.32,.25])
    col *= rng.uniform(.40,1.20,(n,1))
    add(p,col,rng.uniform(.090,.216,n),4)

    n = round(2_600*q)
    p = np.column_stack((rng.uniform(-18,18,n),rng.uniform(-18,18,n),rng.uniform(-4,27,n)))
    col = _palette(rng,n,[(2.65,2.20,1.3),(2.95,2.66,1.95),(2.80,1.75,2.40)],
                   weights=[.42,.32,.26])
    col *= rng.uniform(.62,1.28,(n,1))
    add(p,col,rng.uniform(.1125,.275,n),5,rng.uniform(.45,1.15,n))

    n = round(65*q)
    p = np.column_stack((rng.uniform(-16,16,n),rng.uniform(-10,12,n),rng.uniform(-3,20,n)))
    col = _palette(rng,n,[(.50,.40,.19),(.44,.34,.20),(.47,.23,.37)],weights=[.42,.32,.26])
    add(p,col,rng.uniform(.50,1.12,n),12,rng.uniform(.35,.85,n))

    # Additional layers use their own random stream. Ground/background edits
    # therefore do not shuffle the approved ribbon, pearls or star geometry.
    extra = np.random.default_rng(np.random.SeedSequence([seed,820]))

    # A clear intermediate size between fine floor dust and large flares.
    # These grains cover the spaces between rings, starting outside the base.
    n = round(11_000*q)
    r = np.sqrt(extra.uniform(3.8**2,16.5**2,n))
    a = extra.uniform(0,2*np.pi,n)
    col = _palette(extra,n,[(3.40,2.65,1.45),(3.55,3.00,1.85),(3.40,1.80,2.70)],
                   weights=[.44,.30,.26])
    col *= extra.uniform(.65,1.10,(n,1))
    p = np.column_stack((r*np.cos(a),r*np.sin(a),extra.uniform(.025,.10,n)))
    add(p,col,extra.uniform(.105,.220,n),13)

    # Lightly fill the outer disk without brightening its inner orbit.
    n = round(15_000*q)
    r = np.sqrt(extra.uniform(3.2**2,16.6**2,n))
    a = extra.uniform(0,2*np.pi,n)
    col = _palette(extra,n,[(1.85,1.46,.78),(2.0,1.64,1.0),(1.95,1.08,1.62)],
                   weights=[.44,.30,.26])
    col *= extra.uniform(.20,.58,(n,1))
    p = np.column_stack((r*np.cos(a),r*np.sin(a),extra.normal(.025,.04,n)))
    add(p,col,extra.uniform(.055,.110,n),0)

    # Forty percent more distant stars and mid-distance snow. Their bright
    # cores are larger; the separate large, defocused bokeh layer is not.
    n = round(5_000*q)
    p = np.column_stack((extra.uniform(-45,45,n),extra.uniform(-45,45,n),extra.uniform(-7,33,n)))
    col = _palette(extra,n,[(2.20,1.80,1.14),(2.60,2.30,1.62),(2.40,1.50,2.10)],
                   weights=[.43,.32,.25])
    col *= extra.uniform(.45,1.20,(n,1))
    add(p,col,extra.uniform(.090,.216,n),4)

    n = round(1_040*q)
    p = np.column_stack((extra.uniform(-18,18,n),extra.uniform(-18,18,n),extra.uniform(-4,27,n)))
    col = _palette(extra,n,[(2.80,2.35,1.45),(3.05,2.75,2.05),(2.95,1.90,2.55)],
                   weights=[.42,.32,.26])
    col *= extra.uniform(.62,1.28,(n,1))
    add(p,col,extra.uniform(.1125,.275,n),5,extra.uniform(.45,1.15,n))

    # A loose three-dimensional transition connects the disk to the sky.
    # Exponential height gives many low grains and a few higher drifting ones.
    n = round(5_000*q)
    r = np.sqrt(extra.uniform(7.0**2,22.0**2,n))
    a = extra.uniform(0,2*np.pi,n)
    z = extra.exponential(1.0,n)+.08
    edge = np.clip((r-10.0)/13.0,0.0,1.0)
    fade = (1.0-edge*edge*(3.0-2.0*edge))*np.exp(-z*.13)
    col = _palette(extra,n,[(2.80,2.20,1.28),(2.85,2.45,1.65),(2.65,1.50,2.20)],
                   weights=[.45,.30,.25])
    col *= (fade*extra.uniform(.30,.72,n))[:,None]
    add(np.column_stack((r*np.cos(a),r*np.sin(a),z)),
        col,extra.uniform(.085,.205,n),14,extra.uniform(.20,.65,n))

    scene = np.ascontiguousarray(np.concatenate(blocks),dtype="f4")
    kind = scene[:,8].astype(int)
    ground = (kind==0)|(kind==13)|(((kind==7)|(kind==10))&(scene[:,2]<.25))
    radius = np.linalg.norm(scene[ground,:2],axis=1)
    edge = np.clip((radius-10.5)/6.5,0.0,1.0)
    fade = 1.0-edge*edge*(3.0-2.0*edge)
    scene[ground,3:6] *= fade[:,None]
    # Spread the last orbits instead of terminating every layer at one radius.
    spread = .10*np.maximum(radius-10.0,0.0)**2
    scene[ground,:2] *= (1.0+spread/np.maximum(radius,1e-4))[:,None]
    return scene


def build_silk_ribbon(samples=1800):
    """A faint continuous underlay for the broad particle sheet."""
    t = np.linspace(0,1,samples,dtype="f4")
    meshes,indices = [],[]
    offset = 0
    for track in range(2):
        r,a,z = helix_path(t,.018*track*(1-t))
        width = ribbon_width(t)*(1.0 if track==0 else .65)
        slope = -.90*RIBBON_RADIUS/(RIBBON_TOP-RIBBON_BOTTOM)*np.maximum(1-t,1e-4)**(-.10)
        color = np.tile(np.array((1.05,.55,.79) if track==0 else (.65,.47,.66),dtype="f4"),(samples,1))
        color = crown_tint(color,t)
        vertices = np.empty((2*samples,8),dtype="f4")
        for side in (0,1):
            sign = -1 if side==0 else 1
            rr = r+sign*width*slope
            vertices[side::2,:3] = np.column_stack((rr*np.cos(a),rr*np.sin(a),z+sign*width))
            vertices[side::2,3:6] = color
            vertices[side::2,6] = sign
            vertices[side::2,7] = t
        k = np.arange(samples-1,dtype="i4")*2
        faces = np.stack((k,k+1,k+2,k+2,k+1,k+3),axis=1).ravel()
        meshes.append(vertices)
        indices.append(faces+offset)
        offset += len(vertices)
    return np.concatenate(meshes),np.concatenate(indices)


def camera_matrices(aspect, distance, yaw, pitch, target_z=4.1):
    """OpenGL perspective/view matrices in column-vector convention (Z up)."""
    target = np.array([0.0, 0.0, target_z], dtype="f8")
    eye = target + distance * np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ], dtype="f8")
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4, dtype="f4")
    view[:3, :3] = np.stack((right, up, -forward)).astype("f4")
    view[:3, 3] = [-right @ eye, -up @ eye, forward @ eye]
    fov, near, far = math.radians(47), 0.1, 180.0
    scale = 1.0 / math.tan(fov * .5)
    projection = np.zeros((4, 4), dtype="f4")
    projection[0, 0] = scale / aspect
    projection[1, 1] = scale
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2 * near * far / (near - far)
    projection[3, 2] = -1
    return projection @ view, view


PARTICLE_VS = """#version 330 core
uniform mat4 u_mvp;
uniform float u_time;
uniform float u_pixel_scale;
in vec3 in_pos;
in vec3 in_color;
in float in_size;
in float in_phase;
in float in_kind;
in float in_speed;
out vec3 v_color;
out float v_phase;
out float v_speed;
flat out int v_kind;
void main() {
    int kind = int(in_kind + 0.5);
    vec3 p = in_pos;
    if (kind == 5 || kind == 12) {
        p.z = mod(in_pos.z - u_time * in_speed + 38.0, 29.0) - 4.0;
        p.x += sin(u_time * 0.39 + in_phase) * 0.85;
        p.y += cos(u_time * 0.28 + in_phase * 0.81) * 0.48;
    } else if (kind == 1 || kind == 2 || kind == 3 || kind == 7 || kind == 8 || kind == 10 || kind == 11) {
        float a = u_time * ((p.z < .40) ? .014 : .065);
        p.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * p.xy;
        float taper = clamp((9.50 - p.z) / 2.5, 0.0, 1.0);
        p.xy += 0.012 * taper * vec2(sin(u_time * 1.2 + in_phase),
                                     cos(u_time * 1.1 + in_phase));
    } else if (kind == 0 || kind == 13 || kind == 14) {
        float a = u_time * 0.014;
        p.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * p.xy;
        if (kind == 14) {
            p.z += .13*sin(u_time*in_speed+in_phase);
            p.xy += .18*vec2(sin(u_time*.19+in_phase),cos(u_time*.17+in_phase));
        }
    }
    vec4 clip = u_mvp * vec4(p, 1.0);
    gl_Position = clip;
    float sz = in_size * u_pixel_scale / max(clip.w, 0.1);
    if (kind == 6) sz *= 1.0 + 0.11 * sin(u_time * 3.4 + in_phase);
    if (kind == 0) sz *= 1.36;
    if (kind == 13) sz *= 1.15;
    if (kind == 1) sz *= 1.12;
    if (kind == 5) sz *= 1.25;
    // Reserve halo space around the star; its physical silhouette stays fixed.
    if (kind == 9) sz *= 1.6;
    float max_size = (kind == 9) ? 153.6 : ((kind == 6) ? 96.0 : ((kind == 12) ? 52.0 : ((kind == 5) ? 16.0 : ((kind == 13) ? 10.0 : ((kind == 11 || kind == 10) ? 24.0 : ((kind == 4 || kind == 7) ? 9.0 : 12.0))))));
    gl_PointSize = clamp(sz, 1.5, max_size);
    v_color = in_color;
    // Compare radial depth with the cone axis using the existing MVP.
    // The front glitters strongly; the far side remains softly visible.
    float axis_w = (u_mvp * vec4(0.0,0.0,p.z,1.0)).w;
    float facing = clamp((axis_w-clip.w)/max(length(p.xy),.05),-1.0,1.0);
    if (kind == 1 || kind == 2 || kind == 8) {
        v_color *= .48+.52*smoothstep(-.65,.65,facing);
    } else if (kind == 3 || kind == 11) {
        v_color *= .78+.22*smoothstep(-.65,.65,facing);
    }
    v_phase = in_phase;
    v_speed = in_speed;
    v_kind = kind;
}
"""

PARTICLE_FS = """#version 330 core
uniform float u_time;
uniform float u_density_normalizer;
in vec3 v_color;
in float v_phase;
in float v_speed;
flat in int v_kind;
out vec4 frag;
// Unit five-point star: outer radius 0.83, inner/outer ratio 0.46.
// p uses an upward-positive Y coordinate. There is one point at the top.
float star5_sdf(vec2 p) {
    const vec2 v[10] = vec2[10](
        vec2(0.0, 1.0),
        vec2(0.27038122, 0.37214782),
        vec2(0.95105652, 0.30901699),
        vec2(0.43748599, -0.14214782),
        vec2(0.58778525, -0.80901699),
        vec2(0.0, -0.46),
        vec2(-0.58778525, -0.80901699),
        vec2(-0.43748599, -0.14214782),
        vec2(-0.95105652, 0.30901699),
        vec2(-0.27038122, 0.37214782)
    );
    p /= 0.83;
    float d2 = 100.0;
    bool inside = false;
    for (int i = 0; i < 10; ++i) {
        vec2 a = v[i];
        vec2 b = v[(i+1)%10];
        vec2 e = b-a;
        vec2 w = p-a;
        vec2 nearest = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
        d2 = min(d2,dot(nearest,nearest));
        if ((a.y > p.y) != (b.y > p.y)) {
            float crossing_x = a.x+(p.y-a.y)*(b.x-a.x)/(b.y-a.y);
            if (p.x < crossing_x) inside = !inside;
        }
    }
    return 0.83*sqrt(d2)*(inside ? -1.0 : 1.0);
}

vec3 star5_emission(vec2 point_coord, float time) {
    // OpenGL's default point-coordinate origin is upper-left.
    vec2 p = vec2(point_coord.x,-point_coord.y)*1.6;
    float sd = star5_sdf(p);
    // Round the very sharp polygon corners by 0.006 sprite units.
    sd -= 0.006;
    float aa = max(fwidth(sd),0.006);
    float coverage = 1.0-smoothstep(-aa,aa,sd);
    float center = exp(-dot(p,p)*6.0);
    vec3 champagne = vec3(3.55,2.26,0.75);
    vec3 warm_white = vec3(9.5,8.6,6.8);
    vec3 body = mix(champagne,warm_white,center)*coverage;
    // Smooth outline haze, no random cloud and no extra four-ray cross.
    float halo = exp(-max(sd,0.0)*7.0)*(1.0-coverage)*0.32;
    vec3 glow = vec3(1.25,0.88,0.42)*halo;
    glow += vec3(1.0,.87,.55)*.12*exp(-dot(p,p)*3.0)*(1.0-coverage);
    glow *= 1.0-smoothstep(1.30,1.60,length(p));
    float pulse = 0.96+0.04*sin(time*1.7);
    return (body+glow)*pulse;
}

void main() {
    vec2 p = gl_PointCoord*2.0-1.0;
    float rr = dot(p,p);
    // Compute star derivatives over the full sprite before any circular cut.
    if (v_kind == 9) {
        frag = vec4(star5_emission(p,u_time),1.0);
        return;
    }
    if (rr > 1.0) discard;
    float glow = 0.0;
    if (v_kind == 0) {
        glow = .68*exp(-rr*15.0)+.012*exp(-rr*3.0);
    } else if (v_kind == 1) {
        glow = .90*exp(-rr*14.0)+.035*exp(-rr*3.0);
    } else if (v_kind == 2) {
        glow = 1.10*exp(-rr*18.0)+.020*exp(-rr*3.0);
    } else if (v_kind == 3) {
        glow = .85*exp(-rr*13.0)+.020*exp(-rr*2.5);
    } else if (v_kind == 4) {
        glow = .90*exp(-rr*12.0)+.018*exp(-rr*3.0);
    } else if (v_kind == 5) {
        float a = v_phase*1.9;
        vec2 q = mat2(cos(a),-sin(a),sin(a),cos(a))*p;
        float diamond = 1.0-smoothstep(.32,.63,abs(q.x)+.80*abs(q.y));
        float shape = fract(v_phase*3.34)>.82 ? diamond : exp(-rr*9.0);
        glow = .60*exp(-rr*10.0)+.18*shape;
    } else if (v_kind == 6) {
        glow = .95*exp(-rr*7.0)+.025*exp(-rr*2.0);
    } else if (v_kind == 7) {
        float cross = exp(-abs(p.x)*24.0)*exp(-abs(p.y)*2.8)
                    + exp(-abs(p.y)*24.0)*exp(-abs(p.x)*2.8);
        glow = 1.15*exp(-rr*25.0)+.075*cross;
    } else if (v_kind == 8) {
        glow = .70*exp(-rr*12.0)+.045*exp(-rr*2.8);
    } else if (v_kind == 10) {
        float rays = exp(-abs(p.x)*30.0)*exp(-abs(p.y)*2.4)
                   + exp(-abs(p.y)*30.0)*exp(-abs(p.x)*2.4);
        glow = .90*exp(-rr*35.0)+.20*rays;
    } else if (v_kind == 11) {
        glow = .95*exp(-rr*9.0)+.07*exp(-rr*2.7);
    } else if (v_kind == 12) {
        glow = .25*(1.0-smoothstep(.55,1.0,sqrt(rr)))+.18*exp(-rr*3.0);
    } else if (v_kind == 13) {
        glow = .95*exp(-rr*12.0)+.012*exp(-rr*2.5);
    } else if (v_kind == 14) {
        glow = .90*exp(-rr*12.0)+.015*exp(-rr*3.0);
    }
    float pulse = .84+.16*sin(u_time*(1.5+v_speed)+v_phase*7.2);
    if (v_kind == 4) {
        pulse = .70+.30*pow(.5+.5*sin(u_time*2.3+v_phase*6.8),3.0);
    } else if (v_kind == 10) {
        pulse = .35+.65*pow(.5+.5*sin(u_time*(1.4+v_speed)+v_phase),5.0);
    } else if (v_kind == 11) {
        pulse = .80+.20*sin(u_time*(1.1+v_speed)+v_phase);
    } else if (v_kind == 6 || v_kind == 12) {
        pulse = .95+.05*sin(u_time*1.3+v_phase);
    }
    float gain = .46;
    if      (v_kind == 1) gain = .31;
    else if (v_kind == 2) gain = .36;
    else if (v_kind == 3) gain = .40;
    else if (v_kind == 4) gain = 1.05;
    else if (v_kind == 5) gain = .58;
    else if (v_kind == 6) gain = .75;
    else if (v_kind == 7) gain = .85;
    else if (v_kind == 8) gain = .30;
    else if (v_kind == 10) gain = 1.10;
    else if (v_kind == 11) gain = .78;
    else if (v_kind == 12) gain = .45;
    else if (v_kind == 13) gain = .70;
    else if (v_kind == 14) gain = .75;
    vec3 emission = v_color*glow;
    if (v_kind == 11) {
        emission += vec3(1.0,.94,.95)*.50*exp(-rr*40.0);
    }
    float normalization = v_kind == 6 ? 1.0 : u_density_normalizer;
    frag = vec4(emission*pulse*gain*normalization,1.0);
}
"""


RIBBON_VS = """#version 330 core
uniform mat4 u_mvp;
uniform float u_time;
in vec3 in_pos;
in vec3 in_color;
in float in_edge;
in float in_progress;
out vec3 v_color;
out float v_edge, v_progress;
void main() {
    vec3 p = in_pos;
    float a = u_time * 0.065;
    p.xy = mat2(cos(a), -sin(a), sin(a), cos(a)) * p.xy;
    vec4 clip = u_mvp * vec4(p, 1.0);
    gl_Position = clip;
    v_edge = in_edge;
    float axis_w = (u_mvp * vec4(0.0,0.0,p.z,1.0)).w;
    float facing = clamp((axis_w-clip.w)/max(length(p.xy),.05),-1.0,1.0);
    v_color = in_color*(.48+.52*smoothstep(-.65,.65,facing));
    v_progress = in_progress;
}
"""

RIBBON_FS = """#version 330 core
uniform float u_time;
in vec3 v_color;
in float v_edge, v_progress;
out vec4 frag;
void main() {
    float center = pow(max(0.0, 1.0 - abs(v_edge)), .80);
    float shimmer = .84 + .16 * sin(v_progress * 200.0 - u_time * 1.6);
    float envelope = smoothstep(0.0, .018, v_progress) *
                     (1.0 - smoothstep(.97, 1.0, v_progress));
    float intensity = .045 * center * shimmer * envelope;
    frag = vec4(v_color * intensity, intensity);
}
"""

QUAD_VS = """#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

EXTRACT_FS = """#version 330 core
uniform sampler2D u_scene;
uniform float u_threshold;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    frag = vec4(max(c - vec3(u_threshold), vec3(0.0)), 1.0);
}
"""

BLUR_FS = """#version 330 core
uniform sampler2D u_tex;
uniform vec2 u_step;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 c = texture(u_tex, v_uv).rgb * 0.2270270;
    c += texture(u_tex, v_uv + u_step * 1.3846154).rgb * 0.3162162;
    c += texture(u_tex, v_uv - u_step * 1.3846154).rgb * 0.3162162;
    c += texture(u_tex, v_uv + u_step * 3.2307692).rgb * 0.0702703;
    c += texture(u_tex, v_uv - u_step * 3.2307692).rgb * 0.0702703;
    frag = vec4(c, 1.0);
}
"""

COMPOSITE_FS = """#version 330 core
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_strength;
uniform float u_exposure;
uniform float u_aspect;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb
             + texture(u_bloom, v_uv).rgb * u_bloom_strength;
    vec3 color = vec3(1.0) - exp(-max(hdr, vec3(0.0)) * u_exposure);
    color = pow(color, vec3(1.05));
    vec2 uv = (v_uv - 0.5) * vec2(u_aspect, 1.0);
    vec3 sky = vec3(0.006, 0.004, 0.006)
             + vec3(0.013, 0.007, 0.004) * exp(-dot(uv, uv) * 1.3);
    vec2 floor_cloud = uv - vec2(0.0, -0.34);
    float haze = exp(-((floor_cloud.x*floor_cloud.x)/.21 +
                       (floor_cloud.y*floor_cloud.y)/.045));
    sky += vec3(.028,.015,.007) * haze;
    float vig = 1.0 - 0.25 * smoothstep(0.28, 1.32, length(uv));
    frag = vec4((color + sky) * vig, 1.0);
}
"""

TEXT_VS = """#version 330 core
in vec2 in_screen;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_screen, 0.0, 1.0);
}
"""

TEXT_FS = """#version 330 core
uniform sampler2D u_text;
uniform float u_time;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec2 px = 1.0 / vec2(textureSize(u_text, 0));
    float a = texture(u_text, v_uv).a;
    float halo = 0.0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            halo = max(halo, texture(u_text, v_uv + vec2(dx,dy)*px*2.0).a);
        }
    }
    float shimmer = 0.97 + 0.03 * sin(u_time * .9 + v_uv.x * 7.0);
    vec3 pastel = mix(vec3(.98, .92, .99), vec3(1.0,.76,.91), v_uv.x);
    float alpha = clamp(a + 0.16 * halo * (1.0 - a), 0.0, 1.0);
    frag = vec4(pastel * shimmer, alpha);
}
"""


def _make_text_surface(pg):
    # Prefer a real calligraphic face already installed on the computer.
    # This keeps the single .py file self-contained, without a font download.
    for family in ("Snell Roundhand", "Apple Chancery", "Brush Script MT",
                   "Segoe Script", "URW Chancery L", "Z003"):
        path = pg.font.match_font(family)
        if path:
            font = pg.font.Font(path, 54)
            break
    else:
        path = pg.font.match_font("Georgia,Times New Roman,serif")
        font = pg.font.Font(path, 48) if path else pg.font.Font(None,52)
        font.set_italic(True)
    return font.render("Merry Christmas", True, (255, 255, 255)).convert_alpha()


def _targets(ctx, gl, w, h, bloom_on=True):
    """Set up HDR scene and two reduced-resolution bloom buffers."""
    results = {}
    try:
        scene = ctx.texture((w, h), 4, dtype="f2")
        scene.filter = (gl.LINEAR, gl.LINEAR)
        scene.repeat_x = scene.repeat_y = False
        scene_fbo = ctx.framebuffer(color_attachments=[scene])
    except Exception:
        # Half-float render targets are standard on macOS GL 3.3+, but there
        # is a simple compatible 8-bit fallback for older implementations.
        scene = ctx.texture((w, h), 4, dtype="f1")
        scene.filter = (gl.LINEAR, gl.LINEAR)
        scene.repeat_x = scene.repeat_y = False
        scene_fbo = ctx.framebuffer(color_attachments=[scene])
        bloom_on = False
    results.update(scene=scene, scene_fbo=scene_fbo, bloom_on=bloom_on)
    if bloom_on:
        try:
            bw, bh = max(1, w // 4), max(1, h // 4)
            a = ctx.texture((bw, bh), 4, dtype="f2")
            b = ctx.texture((bw, bh), 4, dtype="f2")
            for tex in (a, b):
                tex.filter = (gl.LINEAR, gl.LINEAR)
                tex.repeat_x = tex.repeat_y = False
            af = ctx.framebuffer(color_attachments=[a])
            bf = ctx.framebuffer(color_attachments=[b])
            results.update(a=a, b=b, af=af, bf=bf, bw=bw, bh=bh)
        except Exception:
            results["bloom_on"] = False
    return results


def _release_targets(targets):
    for key in ("scene_fbo", "af", "bf", "scene", "a", "b"):
        resource = targets.get(key)
        if resource is not None:
            resource.release()


def bind_target(ctx, framebuffer, size):
    """Bind each pass first, then restore its own pixel viewport.

    Bloom buffers are 1/4 the width/height of the main scene. Setting
    ctx.viewport *before* framebuffer.use() changes the previously bound
    framebuffer's cached viewport. That caused the main tree to render in
    the lower-left quarter of the screen on the next frame.
    """
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("Render target dimensions must be positive")
    framebuffer.use()
    ctx.viewport = (0, 0, int(w), int(h))


def screen_drawable_scale(ctx, window_size):
    """Detect logical-window vs physical-drawable DPI scaling on launch.

    On macOS Retina the default OpenGL framebuffer is often 2x the SDL
    window size. Querying its initial size gives us the physical viewport.
    """
    w, h = window_size
    fw, fh = ctx.screen.size
    rx, ry = fw / max(w, 1), fh / max(h, 1)
    if not (0.9 <= rx <= 4.1 and 0.9 <= ry <= 4.1):
        return (1.0, 1.0)
    return (rx, ry)


def run(quality="medium", width=WINDOW_W, height=WINDOW_H, bloom=True):
    try:
        import pygame as pg
        import moderngl as gl
    except ImportError as exc:
        raise SystemExit(
            "缺少图形依赖：请运行 python3 -m pip install numpy pygame moderngl\n"
            f"具体错误：{exc}"
        ) from exc

    pg.init()
    ctx = None
    try:
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)
        # Apple requires a forward-compatible context for OpenGL core profile.
        if hasattr(pg, "GL_CONTEXT_FLAGS"):
            pg.display.gl_set_attribute(
                pg.GL_CONTEXT_FLAGS,
                getattr(pg, "GL_CONTEXT_FORWARD_COMPATIBLE_FLAG", 1)
            )
        flags = pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE
        pg.display.set_mode((width, height), flags)
        pg.display.set_caption("Pink & Gold Galaxy | Merry Christmas")
        ctx = gl.create_context(require=330)
        ctx.enable(gl.PROGRAM_POINT_SIZE | gl.BLEND)
        ctx.disable(gl.DEPTH_TEST)
        ctx.blend_func = (gl.ONE, gl.ONE)
        try:
            pg.display.gl_set_swap_interval(1)
        except (AttributeError, pg.error):
            pass

        scene_data = build_scene(quality)
        print(f"粉金星河圣诞树 SOFT HORIZON · {len(scene_data):,} 个粒子 · 画质 {quality}")
        particle_program = ctx.program(vertex_shader=PARTICLE_VS, fragment_shader=PARTICLE_FS)
        particle_vbo = ctx.buffer(scene_data.tobytes())
        particles = ctx.vertex_array(particle_program, [(
            particle_vbo, "3f 3f 1f 1f 1f 1f",
            "in_pos", "in_color", "in_size", "in_phase", "in_kind", "in_speed",
        )])

        # Soft continuous translucent silk is a subtle underlay; the sparkling
        # points remain the foreground. Same MVP/orbit as the glitter.
        ribbon_data, ribbon_indices = build_silk_ribbon()
        ribbon_program = ctx.program(vertex_shader=RIBBON_VS, fragment_shader=RIBBON_FS)
        ribbon_vao = ctx.vertex_array(ribbon_program, [(
            ctx.buffer(ribbon_data.tobytes()), "3f 3f 1f 1f",
            "in_pos", "in_color", "in_edge", "in_progress",
        )], ctx.buffer(ribbon_indices.tobytes()))

        quad_vbo = ctx.buffer(np.array([-1., -1., 1., -1., -1., 1., 1., 1.], dtype="f4").tobytes())
        extract_program = ctx.program(vertex_shader=QUAD_VS, fragment_shader=EXTRACT_FS)
        blur_program = ctx.program(vertex_shader=QUAD_VS, fragment_shader=BLUR_FS)
        compose_program = ctx.program(vertex_shader=QUAD_VS, fragment_shader=COMPOSITE_FS)
        extract_quad = ctx.vertex_array(extract_program, [(quad_vbo, "2f", "in_position")])
        blur_quad = ctx.vertex_array(blur_program, [(quad_vbo, "2f", "in_position")])
        compose_quad = ctx.vertex_array(compose_program, [(quad_vbo, "2f", "in_position")])
        extract_program["u_scene"].value = 0
        blur_program["u_tex"].value = 0
        compose_program["u_scene"].value = 0
        compose_program["u_bloom"].value = 1
        compose_program["u_exposure"].value = DEFAULT_EXPOSURE
        extract_program["u_threshold"].value = BLOOM_THRESHOLD
        particle_program["u_density_normalizer"].value = 1.0 / QUALITY_SCALE[quality]

        surface = _make_text_surface(pg)
        text_w, text_h = surface.get_size()
        text_tex = ctx.texture((text_w, text_h), 4,
                               pg.image.tostring(surface, "RGBA", True))
        text_tex.filter = (gl.LINEAR, gl.LINEAR)
        text_tex.repeat_x = text_tex.repeat_y = False
        text_program = ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        text_program["u_text"].value = 0
        text_vbo = ctx.buffer(reserve=16 * 4)
        text_quad = ctx.vertex_array(text_program,
                                     [(text_vbo, "2f 2f", "in_screen", "in_uv")])

        width, height = pg.display.get_window_size()
        drawable_scale = screen_drawable_scale(ctx, (width, height))
        print(f"显示器渲染比例：{drawable_scale[0]:.2f}x / {drawable_scale[1]:.2f}x")
        targets = _targets(ctx, gl, width, height, bloom)
        yaw, pitch, distance = -math.pi / 2, 0.43, 23.0
        auto = True
        dragging = False
        clock = pg.time.Clock()
        elapsed = 0.0
        running = True
        snapshot_requested = False
        exposure = DEFAULT_EXPOSURE
        bloom_strength = DEFAULT_BLOOM

        def draw_caption(w, h):
            # Upper left, same placement as the reference video.
            scale = min(1.0, w * 0.26 / max(text_w, 1))
            pw, ph = text_w * scale, text_h * scale
            x = w * .071
            y = h * .172
            x0, x1 = -1 + 2 * x / w, -1 + 2 * (x + pw) / w
            y0, y1 = 1 - 2 * (y + ph) / h, 1 - 2 * y / h
            quad = np.array([
                x0, y0, 0, 0,
                x1, y0, 1, 0,
                x0, y1, 0, 1,
                x1, y1, 1, 1,
            ], dtype="f4")
            text_vbo.write(quad.tobytes())
            ctx.enable(gl.BLEND)
            ctx.blend_func = (gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
            text_tex.use(location=0)
            text_program["u_time"].value = elapsed
            text_quad.render(gl.TRIANGLE_STRIP)
            ctx.blend_func = (gl.ONE, gl.ONE)

        print("操作：拖动旋转 / 滚轮缩放 / 空格自动旋转 / -和+调整亮度 / [和]调整辉光 / S 截图 / R 重置 / Esc 退出")
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.08)
            elapsed += dt
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_ESCAPE, pg.K_q):
                        running = False
                    elif event.key == pg.K_SPACE:
                        auto = not auto
                    elif event.key == pg.K_r:
                        yaw, pitch, distance = -math.pi / 2, .43, 23.0
                    elif event.key == pg.K_s:
                        # Save after this frame is rendered below.
                        snapshot_requested = True
                    elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        exposure = max(.2, exposure - .08)
                        print(f"曝光: {exposure:.2f}")
                    elif event.key in (pg.K_EQUALS, pg.K_KP_PLUS):
                        exposure = min(1.5, exposure + .08)
                        print(f"曝光: {exposure:.2f}")
                    elif event.key == pg.K_LEFTBRACKET:
                        bloom_strength = max(0.0, bloom_strength - .06)
                        print(f"辉光: {bloom_strength:.2f}")
                    elif event.key == pg.K_RIGHTBRACKET:
                        bloom_strength = min(1.0, bloom_strength + .06)
                        print(f"辉光: {bloom_strength:.2f}")
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    dragging = True
                    pg.mouse.get_rel()
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                elif event.type == pg.MOUSEWHEEL:
                    distance = float(np.clip(distance - event.y * 1.15, 13.0, 36.0))

            if not running:
                break
            if dragging and pg.mouse.get_pressed()[0]:
                dx, dy = pg.mouse.get_rel()
                yaw -= dx * .006
                pitch = float(np.clip(pitch + dy * .0045, -.08, 1.12))
            else:
                pg.mouse.get_rel()
                if auto:
                    yaw += dt * .065

            w, h = pg.display.get_window_size()
            if w < 16 or h < 16:
                continue
            if (w, h) != (width, height):
                _release_targets(targets)
                width, height = w, h
                targets = _targets(ctx, gl, w, h, bloom)

            mvp, _ = camera_matrices(w / h, distance, yaw, pitch)
            pixel_scale = h * .5 / math.tan(math.radians(47) / 2)
            ctx.enable(gl.BLEND)
            ctx.blend_func = (gl.ONE, gl.ONE)
            bind_target(ctx, targets["scene_fbo"], (w, h))
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            particle_program["u_mvp"].write(mvp.T.astype("f4").tobytes())
            particle_program["u_time"].value = elapsed
            particle_program["u_pixel_scale"].value = pixel_scale
            particles.render(gl.POINTS)
            ribbon_program["u_mvp"].write(mvp.T.astype("f4").tobytes())
            ribbon_program["u_time"].value = elapsed
            ribbon_vao.render(gl.TRIANGLES)

            ctx.disable(gl.BLEND)
            if targets["bloom_on"]:
                bind_target(ctx, targets["af"], (targets["bw"], targets["bh"]))
                ctx.clear(0, 0, 0, 1)
                targets["scene"].use(location=0)
                extract_quad.render(gl.TRIANGLE_STRIP)
                # Two Gaussian passes; repeat once for a more velvety glow.
                for _ in range(1):
                    bind_target(ctx, targets["bf"], (targets["bw"], targets["bh"]))
                    targets["a"].use(location=0)
                    blur_program["u_step"].value = (1 / targets["bw"], 0.0)
                    blur_quad.render(gl.TRIANGLE_STRIP)
                    bind_target(ctx, targets["af"], (targets["bw"], targets["bh"]))
                    targets["b"].use(location=0)
                    blur_program["u_step"].value = (0.0, 1 / targets["bh"])
                    blur_quad.render(gl.TRIANGLE_STRIP)

            # Draw to the *physical* display framebuffer (Retina aware).
            screen_size = (max(1, round(w * drawable_scale[0])),
                           max(1, round(h * drawable_scale[1])))
            bind_target(ctx, ctx.screen, screen_size)
            ctx.clear(0, 0, 0, 1)
            targets["scene"].use(location=0)
            (targets["a"] if targets["bloom_on"] else targets["scene"]).use(location=1)
            compose_program["u_bloom_strength"].value = bloom_strength if targets["bloom_on"] else 0.0
            compose_program["u_exposure"].value = exposure
            compose_program["u_aspect"].value = w / h
            compose_quad.render(gl.TRIANGLE_STRIP)
            draw_caption(w, h)

            if snapshot_requested:
                try:
                    filename = "ChristmasTree_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
                    raw = ctx.screen.read(viewport=(0, 0, *screen_size),
                                          components=3, alignment=1)
                    shot = pg.image.fromstring(raw, screen_size, "RGB", True)
                    pg.image.save(shot, filename)
                    print("截图已保存：", filename)
                except Exception as exc:
                    print("保存截图失败：", exc)
                snapshot_requested = False
            pg.display.flip()

        _release_targets(targets)
    finally:
        if ctx is not None:
            ctx.release()
        pg.quit()


def self_test():
    """CPU-only sanity check; works without any OpenGL or pygame installation."""
    for quality in ("low", "medium"):
        scene = build_scene(quality, seed=100)
        assert scene.shape[1] == 10
        assert len(scene) > 50_000
        assert np.isfinite(scene).all()
        assert (scene[:,6] > 0).all()
        assert set(range(15)).issubset(set(scene[:, 8].astype(int)))
        print(f"{quality:6s}: {len(scene):,} particles, all finite")
    ribbon, indices = build_silk_ribbon()
    assert np.isfinite(ribbon).all()
    assert indices.min() >= 0 and indices.max() < len(ribbon)
    half = len(ribbon)//2
    assert np.allclose(ribbon[half-2:half,:3], [0.,0.,RIBBON_TOP])
    assert np.allclose(ribbon[-2:,:3], [0.,0.,RIBBON_TOP])
    print("Silk ribbons: valid mesh, both ends meet at the tapered tip")
    for yaw in (-3, 0, 2.1):
        mvp, view = camera_matrices(WINDOW_W/WINDOW_H, 22.5, yaw, .3)
        assert np.isfinite(mvp).all() and abs(np.linalg.det(view)) > 0.95
    print("CPU 检查通过。GPU 图形渲染还需要在本机运行检查。")


def main():
    parser = argparse.ArgumentParser(description="粉金星河圣诞树 (ModernGL)")
    parser.add_argument("--quality", choices=tuple(QUALITY_SCALE), default="medium")
    parser.add_argument("--width", type=int, default=WINDOW_W)
    parser.add_argument("--height", type=int, default=WINDOW_H)
    parser.add_argument("--no-bloom", action="store_true", help="禁用辉光，提高帧率")
    parser.add_argument("--self-test", action="store_true", help="只检查粒子数据，不打开图形窗口")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run(args.quality, max(args.width, 480), max(args.height, 320), not args.no_bloom)


if __name__ == "__main__":
    main()
