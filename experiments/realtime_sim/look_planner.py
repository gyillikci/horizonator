#!/usr/bin/env python3
"""Where should the camera look? From where the boat (roughly) is, COP30 says which way the
land stands that the skyline solver can use: for camera yaws around the boat (a pan head),
the skyline's relief over the field of view, refused where it is flat (under 8 mrad) or
bearing-only (ridges all at one range: 1/r spread under 0.05 per km; margins over skyfix's
own 5 mrad and 0.03 gates, on whose edge the first looks' wrong accepts sat), mostly a shore
nearer than skyfix's 1 km --dmin, or without the true sea horizon across half the frame (the
levelling's). Two ways to choose among the rest (best_yaw's mode):

  explore   relief x land over the ridges' range (a fix errs about in proportion to it;
            under 3 km COP30 errs more, dem_range.py, so nearer counts as 3 km): the most
            position information, for a filter that has not found the boat yet
  exploit   relief x land on ridges beyond 8 km, else dead ahead: near ridges pin a single
            frame but bias a converged filter (COP30's near-range error, and a roll error
            trading against position where the parallax is)

ties going to the yaw nearest dead ahead. The look does not need the fix's precision: a few
hundred metres do not move a mountain range.

    python3 look_planner.py routes/NAME.json [...]     adds "look": [[lat, lon, yaw_deg], ...]
    python3 look_planner.py --views out/aegean out/views_   poses ahead / explore / exploit for every
                                                            frame of a recording (cesium_render)
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import skyline as S  # noqa: E402

DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')
YAWS = np.arange(-180, 180, 15)


def marcher(lat, lon):
    return S.CMarcher(DEM, (lat - .6, lat + .6), (lon - .8, lon + .8), d_min=50.0, d_max=40000.0, refraction_k=0.0)


def scores(cm, lat, lon, z, heading, fov, yaws=YAWS):
    """(yaw, relief mrad, parallax per km, land fraction, median ridge range m, levelable span)
    per camera yaw. Open sea comes back as the sea horizon itself (el = -dip at its distance)."""
    dip = math.sqrt(2 * max(z, 0.5) / S.REARTH)
    d_hor = math.sqrt(2 * max(z, 0.5) * S.REARTH)
    out = []
    for y in yaws:
        az = heading + y + np.arange(-fov / 2, fov / 2 + 1e-9, 0.5)
        el, r = cm.skyline(lat, lon, z, az)
        # skyfix models terrain from 1 km out (--dmin): a nearer shore in the way spoils the column
        land = np.isfinite(r) & (el > -dip + 0.3e-3) & (r >= 1000.0)
        # the levelling needs the true sea horizon across a third of the frame: open sea, or
        # land beyond the horizon's distance (its waterline is hidden behind the horizon)
        lev = np.flatnonzero((el <= -dip + 0.3e-3) | (r > d_hor))
        span = float((lev[-1] - lev[0] + 1) / az.size) if lev.size else 0.0
        if land.sum() < 10:
            out.append((float(y), 0.0, 0.0, 0.0, np.inf, span)); continue
        inv = 1e3 / r[land]
        out.append((float(y), float(np.std(el[land]) * 1e3), float(np.percentile(inv, 90) - np.percentile(inv, 10)),
                    float(land.mean()), float(np.median(r[land])), span))
    return out


def best_yaw(cm, lat, lon, z, heading, fov, mode='explore', min_relief=8.0, min_parallax=0.05, far_m=8000.0):
    """The scores of the yaw to look at. explore: the most position information (relief over
    range), for a filter that has not found the boat yet; exploit: the most relief on ridges
    beyond far_m, where COP30 errs least and a roll error trades least against position, for
    one that has (dead ahead when none qualifies). Both keep a margin over skyfix's own gates
    (5 mrad, 0.03 per km): looks chosen on the gates' edge were most of the wrong accepts."""
    sc = scores(cm, lat, lon, z, heading, fov)
    ok = [s for s in sc if s[1] >= min_relief and s[2] >= min_parallax and s[3] > 0.3 and s[5] >= 0.5]
    turn = lambda s: 1 - 0.05 * abs(s[0]) / 90          # dead ahead wins near-ties (5 % per 90 deg of turn)
    if mode == 'exploit':
        far = [s for s in ok if s[4] >= far_m]
        return max(far, key=lambda s: s[1] * s[3] * turn(s)) if far else next(s for s in sc if s[0] == 0.0)
    return max(ok or sc, key=lambda s: score(s) * turn(s))


def score(s):
    return s[1] * s[3] * 5000.0 / max(s[4], 3000.0)


def annotate(path, step=150.0, fov=70.0, z=2.5):
    r = json.load(open(path))
    cm = marcher(r['lat'], r['lon'])
    pts = [(r['lat'], r['lon'])] + [tuple(w) for w in r['waypoints']]
    look, gain = [], []
    for (la0, lo0), (la1, lo1) in zip(pts[:-1], pts[1:]):
        mlat, mlon = S.meters_per_degree(la0)
        dn, de = (la1 - la0) * mlat, (lo1 - lo0) * mlon
        L, hdg = math.hypot(dn, de), math.degrees(math.atan2(de, dn))
        for s in np.arange(0, L, step):
            la, lo = la0 + dn * s / L / mlat, lo0 + de * s / L / mlon
            b = best_yaw(cm, la, lo, z, hdg, fov)
            look.append([la, lo, b[0]])
            gain.append((score(scores(cm, la, lo, z, hdg, fov, yaws=[0.0])[0]), score(b)))
    r['look'] = look
    json.dump(r, open(path, 'w'), indent=1)
    yaws, g = np.array([x[2] for x in look]), np.array(gain)
    print('%-9s %3d look points: %3d %% off dead ahead (|yaw| > 20), score dead ahead p50 %.1f -> looked %.1f' % (
        r['name'], len(look), 100 * np.mean(np.abs(yaws) > 20), np.median(g[:, 0]), np.median(g[:, 1])))


def view_poses(rec, prefix):
    """cesium_render poses for every frame of a dead-ahead recording, the camera turned three
    ways at the same instant: ahead, explore, exploit; the hull's pitch and roll mixed into
    the camera's as cruise.html mixes them (its camera pitch is -1 deg plus the hull's)."""
    import glob
    out, cms = {'ahead': [], 'explore': [], 'exploit': []}, {}
    for f in sorted(glob.glob(os.path.join(rec, 'cruise_*_[0-9][0-9][0-9].json'))):
        p = json.load(open(f))
        route = p['tag'].split('_')[1]
        cm = cms.get(route) or cms.setdefault(route, marcher(p['lat'], p['lon']))
        wp, wr = p['pitch_deg'] + 1.0, (p['roll_deg'] + 180) % 360 - 180
        for view, poses in out.items():
            y = 0.0 if view == 'ahead' else best_yaw(cm, p['lat'], p['lon'], p['z_msl'], p['heading_deg'], p['hfov_deg'], view)[0]
            cy, sy = math.cos(math.radians(y)), math.sin(math.radians(y))
            poses.append(dict({k: p[k] for k in ('tag', 'lat', 'lon', 'z_msl', 'geoid_n', 'h_ellipsoid', 'hfov_deg', 'width', 'height')},
                              heading_deg=(p['heading_deg'] + y) % 360, pitch_deg=-1.0 + wp * cy + wr * sy, roll_deg=wr * cy - wp * sy,
                              yaw_deg=y, hull_heading_deg=p['heading_deg'], sse=16, timeout_s=300))
    for view, poses in out.items():
        json.dump(poses, open(prefix + view + '.json', 'w'), indent=0)
        y = np.array([q['yaw_deg'] for q in poses])
        print('%-8s %d poses, %d %% turned off the bow, |yaw| median %.0f deg' % (view, len(poses), 100 * np.mean(y != 0), np.median(np.abs(y))))


if __name__ == '__main__':
    if sys.argv[1] == '--views':          # look_planner.py --views RECORDING_DIR POSES_PREFIX
        view_poses(sys.argv[2], sys.argv[3])
    else:
        for f in sys.argv[1:]:
            annotate(f)
