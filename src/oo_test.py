from utils.memory import ReplayBuffer, NStepBuffer

mem = NStepBuffer(4, 8)
s_off = 0
for I in range(30):
	mem.on_step('s' + str(s_off+I), 100+I, 's' +  str(s_off+I+1), 0, False if I%12>0 else True)
	print(I)
	if (mem.has_sample()):
		if I%12==0:
			s_off += 100
		print(mem.get_sample())

exit()