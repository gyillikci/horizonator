"""Where is sea level in the sim's NED frame? Over open water the horizon's
dip below level is sqrt(2 (h - h0) / Reff): fit h0 from images at several heights."""
import sys, time, io, math
import numpy as np
from PIL import Image
import cosysairsim as airsim

NORTH = float(sys.argv[1])
c = airsim.VehicleClient(); c.confirmConnection()
REFF = 6371000 / (1 - 0.0)            # straight rays: the engine draws no refraction
f = 640.0
hs, dips = [], []
for h in (100, 200, 400, 800, 1600):
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(NORTH, 0, -h), airsim.Quaternionr(0, 0, 0, 1)), True, 'Boat')
    time.sleep(12)
    r = c.simGetImages([airsim.ImageRequest('front', airsim.ImageType.Scene, False, True)], 'Boat')[0]
    im = np.asarray(Image.open(io.BytesIO(r.image_data_uint8)).convert('RGB')).astype(float)
    Image.fromarray(im.astype(np.uint8)).save('out/sea_h%d.png' % h)
    # sky is bright-blue-grey, sea dark blue: row of the largest vertical brightness step at the middle columns
    g = im.mean(axis=2)[:, 400:880].mean(axis=1)
    d = np.abs(np.diff(g)); row = int(np.argmax(d[100:]) + 100) + 0.5
    dip = math.atan((row - 359.5) / f)
    hs.append(h); dips.append(dip)
    print('h %5d  horizon row %.1f  dip %.2f mrad' % (h, row, dip * 1e3), flush=True)
hs, dips = np.array(hs), np.array(dips)
# dip^2 * REFF / 2 = h - h0 (linear in h, slope 1 for straight rays)
y = dips ** 2 * REFF / 2
A = np.vstack([hs, np.ones_like(hs)]).T
slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
print('slope %.2f (expect 1), sea level at NED height h0 = %.1f m (fit with slope fixed: %.1f m)' % (slope, -icpt / slope, float(np.mean(hs - y))))
