import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst,GLib

Gst.init(None)

pipeline = Gst.parse_launch('autoaudiosrc ! audioconvert !' +
                            ' audioresample ! pocketsphinx name=asr !' +
                            ' fakesink')

hmm = "/home/shauvik/catkin_ws/src/pocketsphinx/model/en-us/en-us"
lm = "/home/shauvik/catkin_ws/src/pocketsphinx/model/en-us/7156.lm.bin"
dict_path = "/home/shauvik/catkin_ws/src/pocketsphinx/model/en-us/7156.dic"

asr = pipeline.get_by_name('asr')
asr.set_property('hmm', hmm)
asr.set_property('lm', lm)
asr.set_property('dict', dict_path)

bus = pipeline.get_bus()
bus.add_signal_watch()

pipeline.set_state(Gst.State.PLAYING)

def on_message(bus,message):
    if message.type == Gst.MessageType.ELEMENT:
        structure = message.get_structure()
        if structure.get_name() == 'pocketsphinx':
            final = structure.get_value('final')
            hypothesis = structure.get_value('hypothesis')
            if final:
                print("Recognized:", hypothesis)
    if message.type == Gst.MessageType.EOS:
        pass

            
bus.connect("message",on_message)

loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
   pipeline.set_state(Gst.State.NULL)
   