import threading

class StoppableThread(threading.Thread):
	"""Thread class with a stop() method. The thread itself has to check
	regularly for the stopped() condition."""

	def __init__(self):
		super(StoppableThread, self).__init__()
		self._stop_event = threading.Event()

	def stop(self):
		print("STOP AGENT**************************************************")
		self._stop_event.set()

	def stopped(self):
		if self._stop_event.is_set():
			print("AGENT IS STOPPED: ", self._stop_event.is_set())
		return self._stop_event.is_set()