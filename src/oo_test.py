from env_model.model import EnvModelFreewayManual
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

env_model = EnvModelFreewayManual(None)
a = env_model.get_samples(10)
print(a[0].shape)
print(a[0])
print(a[1].shape)
print(a[1])
print(a[2].shape)
print(a[2])
print(a[3].shape)
print(a[3])


#from utils.memory import ReplayBuffer, NStepBuffer
#
#mem = NStepBuffer(4, 8)
#s_off = 0
#for I in range(30):
#	mem.on_step('s' + str(s_off+I), 100+I, 's' +  str(s_off+I+1), 0, False if I%12>0 else True)
#	print(I)
#	if (mem.has_sample()):
#		if I%12==0:
#			s_off += 100
#		print(mem.get_sample())
#
#exit()