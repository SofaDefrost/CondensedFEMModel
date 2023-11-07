import SofaController as Controller
import time as time

Embedded_Object = Controller.EmbeddedController()

while 1 :
	Embedded_Object.EachTimeStep()
	time.sleep(0.1)


