#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst,GLib
import rospy
from std_msgs.msg import String
import signal

Gst.init(None)

pub = rospy.Publisher('speech', String, queue_size=10)
rospy.init_node('speech_node', anonymous=True)

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

def on_message(bus,message):
    if message.type == Gst.MessageType.ELEMENT:
        structure = message.get_structure()
        if structure.get_name() == 'pocketsphinx':
            final = structure.get_value('final')
            hypothesis = structure.get_value('hypothesis')
            if final:
                print("Recognized:", hypothesis)
                pub.publish(hypothesis)
    if message.type == Gst.MessageType.EOS:
        pass

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message",on_message)

def signal_handler(signum, frame):
    rospy.loginfo(f"Signal {signum} received, stopping...")
    loop.quit()

signal.signal(signal.SIGINT, signal_handler)

pipeline.set_state(Gst.State.PLAYING)

loop = GLib.MainLoop()

if __name__ == '__main__':
    try:
        loop.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        rospy.loginfo("Pipeline terminated.")
   