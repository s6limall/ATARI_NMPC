import pyvicon_datastream as pv
import time

client = pv.PyViconDatastream()
ret = client.connect("131.220.7.195:801")
print(f"Connect: {ret}")
client.enable_segment_data()
client.set_stream_mode(pv.StreamMode.ServerPush)

for i in range(20):
    frame = client.get_frame()
    print(f"get_frame: {frame}")
    if frame == pv.Result.Success:
        n = client.get_subject_count()
        print(f"  subjects: {n}")
        for j in range(n):
            name = client.get_subject_name(j)
            print(f"    [{j}] '{name}'")
            seg_count = client.get_segment_count(name)
            for k in range(seg_count):
                seg = client.get_segment_name(name, k)
                trans = client.get_segment_global_translation(name, seg)
                quat = client.get_segment_global_quaternion(name, seg)
                print(f"      seg '{seg}': pos={trans}, quat={quat}")

    time.sleep(0.1)
