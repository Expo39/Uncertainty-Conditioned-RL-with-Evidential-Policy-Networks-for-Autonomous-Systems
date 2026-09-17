"""
@file patch_bridge.py
@brief Patches the CARLA ROS bridge source after cloning.

Applied during docker build (see ros2/Dockerfile). Catches RuntimeError when
the bridge's _update_thread tries to stop/destroy a sensor the env already
destroyed, which otherwise crashes Thread-1 and hangs world.tick() in sync mode.
"""

import pathlib

BASE = pathlib.Path("/workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge")

p = BASE / "sensor.py"
original = p.read_text()
_SENSOR_TARGET = (
    "        self._callback_active.acquire()\n"
    "        if self.carla_actor.is_listening:\n"
    "            self.carla_actor.stop()\n"
    "        super(Sensor, self).destroy()"
)
_SENSOR_PATCH = (
    "        self._callback_active.acquire()\n"
    "        try:\n"
    "            if self.carla_actor.is_listening:\n"
    "                self.carla_actor.stop()\n"
    "        except Exception:\n"
    "            pass\n"
    "        try:\n"
    "            super(Sensor, self).destroy()\n"
    "        except Exception:\n"
    "            pass"
)
assert _SENSOR_TARGET in original, (
    f"patch_bridge.py: patch target not found in {p}. "
    "ros-bridge may have been updated - review and update the patch."
)
p.write_text(original.replace(_SENSOR_TARGET, _SENSOR_PATCH))

p = BASE / "actor_factory.py"
original = p.read_text()
_FACTORY_TARGET = (
    "        actor.destroy()\n"
    "        if carla_actor and delete_actor:\n"
    "            carla_actor.destroy()\n"
    '        self.node.loginfo("Removed {}(id={})".format(actor.__class__.__name__, actor.uid))'
)
_FACTORY_PATCH = (
    "        try:\n"
    "            actor.destroy()\n"
    "        except Exception:\n"
    "            pass\n"
    "        if carla_actor and delete_actor:\n"
    "            try:\n"
    "                carla_actor.destroy()\n"
    "            except Exception:\n"
    "                pass\n"
    '        self.node.loginfo("Removed {}(id={})".format(actor.__class__.__name__, actor.uid))'
)
assert _FACTORY_TARGET in original, (
    f"patch_bridge.py: patch target not found in {p}. "
    "ros-bridge may have been updated - review and update the patch."
)
p.write_text(original.replace(_FACTORY_TARGET, _FACTORY_PATCH))
