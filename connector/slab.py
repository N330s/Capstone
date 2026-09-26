"""Slab decomposition of a socket face with round/slot openings into convex pieces.

MuJoCo has no concave collision primitives, so the socket block (face rectangle minus the plug
openings, extruded along +X) is cut into horizontal Z-strips at every opening vertex and every
crossing between opening edges. Inside a strip no boundary crosses, so each opening section and
each solid gap between openings is a trapezoid; extruded along X it is a box (vertical edges) or an
8-vertex convex hull. The 1 mm lead-in uses the same strips with the openings dilated at the mouth,
giving one tapered hull per solid piece exactly like the hand-written legacy ``*_lead`` hulls.
Everything here is 2-D numpy geometry with no MuJoCo dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from connector.spec import OpeningSpec, Part

EPS = 1e-9


# ----------------------------------------------------------------- polygons

def regular_polygon(center, apothem, sides=16):
    """CCW regular polygon whose *inscribed* circle has radius ``apothem`` (flats on the axes)."""
    if sides % 4 or sides < 8:
        raise ValueError("sides must be a multiple of 4 and >= 8")
    circumradius = apothem / math.cos(math.pi / sides)
    angles = (np.arange(sides) + 0.5) * 2 * math.pi / sides
    return np.column_stack((center[0] + circumradius * np.cos(angles),
                            center[1] + circumradius * np.sin(angles)))


def rectangle(center, half):
    cy, cz = center
    hy, hz = half
    return np.array([(cy - hy, cz - hz), (cy + hy, cz - hz), (cy + hy, cz + hz), (cy - hy, cz + hz)])


def part_polygon(part: Part, sides=16, dilate=0.0):
    if part.kind == "hole":
        return regular_polygon(part.center_yz, part.radius_m + dilate, sides)
    return rectangle(part.center_yz, (part.half_yz[0] + dilate, part.half_yz[1] + dilate))


def opening_polygons(opening: OpeningSpec, sides=16, dilate=0.0):
    return [part_polygon(p, sides, dilate) for p in opening.parts]


def polygon_area(poly):
    y, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1))))


def point_in_polygon(point, poly, tol=0.0):
    """Inside test for a convex CCW polygon; ``tol`` > 0 shrinks the polygon by that margin."""
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)   # inward for CCW
        if float(np.dot(normal, point - a)) < tol:
            return False
    return True


def interval_at(poly, z):
    """Y-interval of a convex polygon on the horizontal line at ``z`` or None."""
    ys = []
    n = len(poly)
    for i in range(n):
        (ya, za), (yb, zb) = poly[i], poly[(i + 1) % n]
        if abs(za - z) <= EPS:
            ys.append(float(ya))
        if abs(zb - z) <= EPS:
            ys.append(float(yb))
        if (za < z < zb) or (zb < z < za):
            ys.append(float(ya + (yb - ya) * (z - za) / (zb - za)))
    if not ys:
        return None
    return (min(ys), max(ys))


def z_range(poly):
    return float(poly[:, 1].min()), float(poly[:, 1].max())


def _segment_crossing_z(p, q, r, s):
    """Z of the proper intersection of segments p-q and r-s, or None."""
    d1, d2 = q - p, s - r
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < 1e-15:
        return None
    w = r - p
    t = (w[0] * d2[1] - w[1] * d2[0]) / denominator
    u = (w[0] * d1[1] - w[1] * d1[0]) / denominator
    if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS:
        return float(p[1] + t * d1[1])
    return None


def edge_crossings(polys):
    """Z coordinates where edges of *different* polygons intersect."""
    zs = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a, b = polys[i], polys[j]
            for k in range(len(a)):
                for l in range(len(b)):
                    z = _segment_crossing_z(a[k], a[(k + 1) % len(a)], b[l], b[(l + 1) % len(b)])
                    if z is not None:
                        zs.append(z)
    return zs


def breakpoints(z_lo, z_hi, poly_sets, snap=1e-6):
    """Sorted strip boundaries: face bounds, polygon vertices and inter-polygon edge crossings."""
    values = [z_lo, z_hi]
    for polys in poly_sets:
        for poly in polys:
            values.extend(float(z) for z in poly[:, 1])
        values.extend(edge_crossings(polys))
    values = sorted(v for v in values if z_lo - EPS <= v <= z_hi + EPS)
    out = [values[0]]
    for v in values[1:]:
        if v - out[-1] > snap:
            out.append(v)
    out[0], out[-1] = z_lo, z_hi
    return out


def union_intervals(polys, za, zb):
    """Merged opening Y-intervals over one strip as [(lo_a, hi_a, lo_b, hi_b)], sorted by Y."""
    zm = 0.5 * (za + zb)
    sections = []
    for poly in polys:
        lo, hi = z_range(poly)
        if lo < zm < hi:
            ia, ib, im = interval_at(poly, za), interval_at(poly, zb), interval_at(poly, zm)
            sections.append((im, ia, ib))
    sections.sort(key=lambda s: s[0][0])
    merged = []
    for im, ia, ib in sections:
        if merged and im[0] <= merged[-1][0][1] + EPS:
            pm, pa, pb = merged[-1]
            merged[-1] = ((pm[0], max(pm[1], im[1])), (min(pa[0], ia[0]), max(pa[1], ia[1])),
                          (min(pb[0], ib[0]), max(pb[1], ib[1])))
        else:
            merged.append((im, ia, ib))
    return [(ia[0], ia[1], ib[0], ib[1]) for _, ia, ib in merged]


def strip_solids(y_lo, y_hi, polys, za, zb):
    """Solid gaps between openings in one strip as [(lo_a, hi_a, lo_b, hi_b)]."""
    gaps = []
    left_a = left_b = y_lo
    for lo_a, hi_a, lo_b, hi_b in union_intervals(polys, za, zb):
        gaps.append((left_a, lo_a, left_b, lo_b))
        left_a, left_b = hi_a, hi_b
    gaps.append((left_a, y_hi, left_b, y_hi))
    return [g for g in gaps if max(g[1] - g[0], g[3] - g[2]) > 1e-7]


# ----------------------------------------------------------------- pieces

@dataclass
class Piece:
    """One convex solid: trapezoid ``back`` (and ``front`` at the mouth for a lead-in piece)."""
    name: str
    za: float
    zb: float
    back: tuple            # (lo_a, hi_a, lo_b, hi_b) at za / zb
    front: tuple | None = None
    x0: float = 0.0
    x1: float = 0.0

    @property
    def is_box(self) -> bool:
        lo_a, hi_a, lo_b, hi_b = self.back
        vertical = abs(lo_a - lo_b) < EPS and abs(hi_a - hi_b) < EPS
        return vertical and (self.front is None or all(abs(f - b) < EPS for f, b in zip(self.front, self.back)))

    def box(self):
        """(pos, half_size) for a box piece."""
        lo, hi = self.back[0], self.back[1]
        return ((0.5 * (self.x0 + self.x1), 0.5 * (lo + hi), 0.5 * (self.za + self.zb)),
                (0.5 * (self.x1 - self.x0), 0.5 * (hi - lo), 0.5 * (self.zb - self.za)))

    def vertices(self):
        """8 hull vertices: front trapezoid at x0 (dilated if lead-in) and back trapezoid at x1."""
        front = self.front if self.front is not None else self.back
        points = []
        for x, (lo_a, hi_a, lo_b, hi_b) in ((self.x0, front), (self.x1, self.back)):
            points.extend([(x, lo_a, self.za), (x, hi_a, self.za), (x, lo_b, self.zb), (x, hi_b, self.zb)])
        return points

    def section_area(self):
        lo_a, hi_a, lo_b, hi_b = self.back
        return 0.5 * ((hi_a - lo_a) + (hi_b - lo_b)) * (self.zb - self.za)

    def contains(self, y, z, tol=0.0):
        """Point inside the back trapezoid (shrunk by ``tol``)."""
        if not (self.za + tol <= z <= self.zb - tol):
            return False
        t = (z - self.za) / (self.zb - self.za) if self.zb > self.za else 0.0
        lo = self.back[0] + t * (self.back[2] - self.back[0])
        hi = self.back[1] + t * (self.back[3] - self.back[1])
        return lo + tol <= y <= hi - tol


def _collinear(prev: Piece, nxt: Piece, attr):
    """Left and right edges of two vertically adjacent trapezoids continue in a straight line."""
    a, b = getattr(prev, attr), getattr(nxt, attr)
    if a is None or b is None:
        return a is None and b is None
    if abs(a[2] - b[0]) > EPS or abs(a[3] - b[1]) > EPS:   # shared boundary at prev.zb == nxt.za
        return False
    ha, hb = prev.zb - prev.za, nxt.zb - nxt.za
    for i_lo, i_hi in ((0, 2), (1, 3)):   # left edge (lo_a -> lo_b), right edge (hi_a -> hi_b)
        if abs((a[i_hi] - a[i_lo]) / ha - (b[i_hi] - b[i_lo]) / hb) > 1e-7:
            return False
    return True


def merge_strips(rows):
    """Merge vertically adjacent pieces whose edges are collinear (``rows`` = pieces per strip)."""
    done, current = [], (list(rows[0]) if rows else [])
    for row in rows[1:]:
        pairs = list(zip(current, row))
        if len(row) == len(current) and all(
                abs(p.zb - n.za) < EPS and _collinear(p, n, "back") and _collinear(p, n, "front")
                for p, n in pairs):
            current = [Piece(p.name, p.za, n.zb, (p.back[0], p.back[1], n.back[2], n.back[3]),
                             None if p.front is None else (p.front[0], p.front[1], n.front[2], n.front[3]),
                             p.x0, p.x1) for p, n in pairs]
        else:
            done.extend(current)
            current = list(row)
    done.extend(current)
    return done


def pair_gaps(back, front):
    """Match mouth (dilated) solid gaps to throat gaps within one strip.

    Dilation only enlarges openings, so every front gap lies inside exactly one back gap. A back gap
    that contains several front gaps (the mouth has an opening where the throat is solid) is split
    where that notch closes, at the notch midpoint; a back gap inside a dilated opening gets a
    degenerate zero-width front so the hull tapers to a line at the mouth.
    Returns [(back_gap, front_gap)].
    """
    pairs = []
    for lo_a, hi_a, lo_b, hi_b in back:
        ma, mb = 0.5 * (lo_a + hi_a), 0.5 * (lo_b + hi_b)
        inside = [f for f in front if lo_a - EPS <= 0.5 * (f[0] + f[1]) <= hi_a + EPS
                  and lo_b - EPS <= 0.5 * (f[2] + f[3]) <= hi_b + EPS]
        if not inside:
            pairs.append(((lo_a, hi_a, lo_b, hi_b), (ma, ma, mb, mb)))
            continue
        cuts_a = [lo_a] + [0.5 * (inside[k][1] + inside[k + 1][0]) for k in range(len(inside) - 1)] + [hi_a]
        cuts_b = [lo_b] + [0.5 * (inside[k][3] + inside[k + 1][2]) for k in range(len(inside) - 1)] + [hi_b]
        for k, f in enumerate(inside):
            pairs.append(((cuts_a[k], cuts_a[k + 1], cuts_b[k], cuts_b[k + 1]), f))
    return pairs


def slab_decompose(face, openings, sides=16, x_lo=0.0, x_hi=0.0, dilate=0.0, prefix="socket_w",
                   snap=1e-6):
    """Convex pieces of ``face`` (y_lo, y_hi, z_lo, z_hi) minus the openings, from x_lo to x_hi.

    With ``dilate`` > 0 the pieces are lead-in frusta: front section (x_lo) uses openings dilated by
    that amount, back section (x_hi) the nominal openings. Returns a flat list of ``Piece``.
    """
    y_lo, y_hi, z_lo, z_hi = face
    nominal = [poly for o in openings for poly in opening_polygons(o, sides)]
    sets = [nominal]
    front = None
    if dilate > 0:
        front = [poly for o in openings for poly in opening_polygons(o, sides, dilate)]
        sets.append(front)
    levels = breakpoints(z_lo, z_hi, sets, snap)
    rows = []
    for index, (za, zb) in enumerate(zip(levels[:-1], levels[1:])):
        back = strip_solids(y_lo, y_hi, nominal, za, zb)
        if front is None:
            pairs = [(gap, None) for gap in back]
        else:
            pairs = pair_gaps(back, strip_solids(y_lo, y_hi, front, za, zb))
        rows.append([Piece(f"{prefix}{index:02d}_{k}", za, zb, gap, mouth, x_lo, x_hi)
                     for k, (gap, mouth) in enumerate(pairs)])
    pieces = merge_strips(rows)
    for i, piece in enumerate(pieces):
        piece.name = f"{prefix}{i:02d}"
    return pieces


def socket_pieces(socket_spec, leadin=True, margin=0.002):
    """All collision pieces of a socket: outer frame boxes + decomposed opening zone (+ lead-in).

    Returns ``(frame, throat, lead)`` lists of ``Piece``. ``frame`` boxes run the full depth; the
    zone pieces run from ``leadin_m`` (or 0 without lead-in) to ``depth_m``; ``lead`` frusta from
    0 to ``leadin_m``.
    """
    s = socket_spec
    lead_length = s.leadin_m if leadin else 0.0
    ylo, yhi, zlo, zhi = zip(*[o.bounds() for o in s.openings])
    zone = (max(s.face_y_m[0], min(ylo) - margin), min(s.face_y_m[1], max(yhi) + margin),
            max(s.face_z_m[0], min(zlo) - margin), min(s.face_z_m[1], max(zhi) + margin))
    frame = []
    fy0, fy1, fz0, fz1 = s.face_y_m[0], s.face_y_m[1], s.face_z_m[0], s.face_z_m[1]
    zy0, zy1, zz0, zz1 = zone
    # The frame boxes overlap the zone by half the margin and sit 0.3 mm behind the face, so the
    # opening zone reads as a raised centre panel and no face is coplanar with the zone pieces
    # (avoids rendering seams); the pins never reach the frame.
    lap, recess = margin / 2, 3e-4
    for name, box in (("bottom", (fy0, fy1, fz0, zz0 + lap)), ("top", (fy0, fy1, zz1 - lap, fz1)),
                      ("left", (fy0, zy0 + lap, fz0, fz1)), ("right", (zy1 - lap, fy1, fz0, fz1))):
        y0, y1, z0, z1 = box
        if y1 - y0 > 1e-7 and z1 - z0 > 1e-7:
            frame.append(Piece(f"socket_frame_{name}", z0, z1, (y0, y1, y0, y1), None, recess, s.depth_m))
    throat = slab_decompose(zone, s.openings, s.polygon_sides, lead_length, s.depth_m, prefix="socket_w")
    lead = []
    if lead_length > 0:
        lead = slab_decompose(zone, s.openings, s.polygon_sides, 0.0, lead_length,
                              dilate=s.leadin_dilation_m, prefix="socket_lead")
    return frame, throat, lead
