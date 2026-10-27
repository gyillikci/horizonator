"""Place a camera-only vehicle (AirSim ComputerVision mode: no body, no physics)
at a set of heights and save what it sees. NED: z is down, so height = -z."""
import sys, time
import cosysairsim as airsim

c = airsim.VehicleClient()
for _ in range(120):
    try:
        c.confirmConnection(); break
    except Exception:
        time.sleep(2)
print('vehicles', c.listVehicles())
for h in [float(x) for x in sys.argv[2:]]:
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, -h), airsim.Quaternionr(0, 0, 0, 1)), True, 'Boat')
    time.sleep(float(sys.argv[1]))
    r = c.simGetImages([airsim.ImageRequest('front', airsim.ImageType.Scene, False, True)], 'Boat')[0]
    open('out/boat_h%g.png' % h, 'wb').write(r.image_data_uint8)
    p = c.simGetVehiclePose('Boat').position
    print('h %5.1f -> pose z %.2f  image %dx%d' % (h, p.z_val, r.width, r.height), flush=True)
