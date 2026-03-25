"""
@file patch_bridge.py
@brief Patches the CARLA ROS bridge source after cloning.

Applied during docker build (see ros2/Dockerfile). Catches RuntimeError when
the bridge's _update_thread tries to stop/destroy a sensor the env already
destroyed, which otherwise crashes Thread-1 and hangs world.tick() in sync mode.
"""
import pathlib

BASE = pathlib.Path("/workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge")

# sensor.py: catch RuntimeError on stop()/destroy() of an already-gone actor
p = BASE / "sensor.py"
t = p.read_text()
t = t.replace(
    "        self._callback_active.acquire()\n"
    "        if self.carla_actor.is_listening:\n"
    "            self.carla_actor.stop()\n"
    "        super(Sensor, self).destroy()",
    "        self._callback_active.acquire()\n"
    "        try:\n"
    "            if self.carla_actor.is_listening:\n"
    "                self.carla_actor.stop()\n"
    "        except Exception:\n"
    "            pass\n"
    "        try:\n"
    "            super(Sensor, self).destroy()\n"
    "        except Exception:\n"
    "            pass",
)
p.write_text(t)

# actor_factory.py: catch RuntimeError on destroy() of an already-gone actor
p = BASE / "actor_factory.py"
t = p.read_text()
t = t.replace(
    '        actor.destroy()\n'
    '        if carla_actor and delete_actor:\n'
    '            carla_actor.destroy()\n'
    '        self.node.loginfo("Removed {}(id={})".format(actor.__class__.__name__, actor.uid))',
    '        try:\n'
    '            actor.destroy()\n'
    '        except Exception:\n'
    '            pass\n'
    '        if carla_actor and delete_actor:\n'
    '            try:\n'
    '                carla_actor.destroy()\n'
    '            except Exception:\n'
    '                pass\n'
    '        self.node.loginfo("Removed {}(id={})".format(actor.__class__.__name__, actor.uid))',
)
p.write_text(t)
