import numpy as np
import sys
from PIL import Image 

baseDir = "test_run_m133/"
f = open(f"{baseDir}actions.txt", "r")
actions = f.read()
actions = actions.split("\n")
actions = [int(a.split("\t")[2]) for a in actions if len(a.split("\t"))>=3]
#print(actions)

#sys.exit()

steps = [29, 45, 89, 136, 259, 418, 479, 494, 516, 576, 599, 640, 854, 889, 920, 1003, 1055]

x = 46
dx = 10
ds = 25
for step in steps:
    print(step)
    start_step = step - 3
    final_arr = np.zeros((210,dx*ds*2,4),dtype="uint8")
    for I in range(ds):
        im = Image.open(f"{baseDir}{start_step+I}_0_real.png")
        arr = np.array(im)
        slice = arr[:,x-dx:x+dx,:]
        slice[:,-1:,3] = 255
        slice[200:,:,3] = 255
        slice[200:,:,actions[start_step+I]] = 255
        final_arr[:,I*dx*2:(I+1)*dx*2,:] = slice
    im = Image.fromarray(final_arr)
    im.save(f"test_{step}.png")