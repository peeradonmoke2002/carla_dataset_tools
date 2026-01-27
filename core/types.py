#!/usr/bin/python3

from dataclasses import dataclass
from core.geometry import *


@dataclass
class ObjectLabel(object):
    frame: int
    timestamp: float
    label_type: str
    carla_id: str
    transform: Transform
    bounding_box: BoundingBox
    velocity: Vector3d = None  # Velocity in m/s (vx, vy, vz)

    def __str__(self):
        return "ObjectLabel(frame={}, timestamp={}, label_type={}, carla_id={}, transform={}, bounding_box={}, velocity={}".format(
            self.frame,
            self.timestamp,
            self.label_type,
            self.carla_id,
            self.transform,
            self.bounding_box,
            self.velocity
        )
