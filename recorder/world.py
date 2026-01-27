#!/usr/bin/python3
import os
import pickle
import logging
import carla
from dataclasses import dataclass

from recorder.actor import PseudoActor
from core.types import *
from core.transform import carla_bbox_to_bbox, carla_transform_to_transform

# Get logger instance
logger = logging.getLogger(__name__)


class WorldActor(PseudoActor):
    def __init__(self, uid, carla_world: carla.World, base_save_dir: str):
        super().__init__(uid, self.get_type_id(), None)
        self.save_dir = "{}/{}_{}".format(base_save_dir, self.get_type_id(), uid)
        self.carla_world = carla_world

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save world objects to disk

        Args:
            frame_id: Absolute CARLA frame ID (for file naming)
            timestamp: Timestamp
            debug: Debug flag

        Returns:
            dict: World objects information including counts by type
        """
        # Note: Currently saves object labels (vehicles, pedestrians, static objects)
        # with transform and bounding box information. The implementation covers:
        # - Dynamic actors (vehicles, walkers) via get_actors()
        # - Static environment objects (cars, trucks, buses, etc.) via get_environment_objects()
        # 
        # Future enhancement: Could add more detailed bbox information or
        # additional object metadata if needed for specific use cases.
        # 
        # Current format: ObjectLabel with frame, timestamp, label_type, carla_id,
        # transform, and bounding_box (location + extent)
        object_labels = []

        # Get environment objects for different vehicle types (CARLA 0.9.16 API)
        # In CARLA 0.9.16, CityObjectLabel.Vehicles was removed, use specific types instead
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Car)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Truck)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Bus)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Motorcycle)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Bicycle)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Pedestrians)

        carla_actors = self.carla_world.get_actors()
        for carla_actor in carla_actors:
            if carla_actor.type_id.startswith('vehicle') \
                    or carla_actor.type_id.startswith('walker'):
                transform = carla_transform_to_transform(carla_actor.get_transform())
                bbox = carla_bbox_to_bbox(carla_actor.bounding_box)
                # Get velocity from CARLA actor
                carla_vel = carla_actor.get_velocity()
                velocity = Vector3d(carla_vel.x, carla_vel.y, carla_vel.z)
                if carla_actor.type_id.startswith('walker'):
                    label_type = 'Pedestrian'
                else:
                    # Classify vehicles - keep only Cars for training
                    vehicle_id = carla_actor.type_id.lower()
                    if any(x in vehicle_id for x in ['truck', 'bus', 'motorcycle', 'bicycle', 'motorbike', 'bike']):
                        continue  # Skip non-car vehicles
                    label_type = 'Car'
                object_labels.append(ObjectLabel(frame=frame_id,
                                                 timestamp=timestamp,
                                                 label_type=label_type,
                                                 carla_id=carla_actor.id,
                                                 transform=transform,
                                                 bounding_box=bbox,
                                                 velocity=velocity))

        if len(object_labels) == 0:
            return {
                'type': 'world',
                'name': self.name,
                'objects_count': 0,
                'vehicle_count': 0,
                'pedestrian_count': 0,
                'static_count': 0,
                'file': None
            }

        # Count by type
        vehicle_count = sum(1 for obj in object_labels if obj.label_type == 'Car')
        pedestrian_count = sum(1 for obj in object_labels if obj.label_type == 'Pedestrian')
        static_count = len(object_labels) - vehicle_count - pedestrian_count

        os.makedirs(self.save_dir, exist_ok=True)
        filename = '{:0>10d}.pkl'.format(frame_id)
        filepath = '{}/{}'.format(self.save_dir, filename)

        with open(filepath, 'wb') as pkl_file:
            pickle.dump(obj=object_labels, file=pkl_file)

        if debug:
            logger.debug(f"WorldObjectsLabel: Frame: {frame_id} Total counts: {len(object_labels)}")

        return {
            'type': 'world',
            'name': self.name,
            'objects_count': len(object_labels),
            'vehicle_count': vehicle_count,
            'pedestrian_count': pedestrian_count,
            'static_count': static_count,
            'file': filename
        }

    def get_type_id(self):
        return 'others.world'

    def get_save_dir(self):
        return self.save_dir

    def get_carla_transform(self) -> carla.Transform:
        return carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))

    def get_env_objects_labels(self, frame, timestamp, object_type: carla.CityObjectLabel) -> list:
        object_labels = []
        # Map CARLA 0.9.16 CityObjectLabel types to match Autoware classes
        # Keep CAR and PEDESTRIAN separate for training
        if object_type == carla.CityObjectLabel.Car:
            label_type = 'Car'
        elif object_type == carla.CityObjectLabel.Pedestrians:
            label_type = 'Pedestrian'
        elif object_type in (carla.CityObjectLabel.Truck, carla.CityObjectLabel.Bus,
                           carla.CityObjectLabel.Motorcycle, carla.CityObjectLabel.Bicycle):
            # Skip other vehicle types (we only train on Cars)
            return object_labels
        else:
            label_type = 'DontCare'
        env_objects = self.carla_world.get_environment_objects(object_type=object_type)
        for env_object in env_objects:
            transform = carla_transform_to_transform(env_object.transform)
            bbox_extent = Vector3d(env_object.bounding_box.extent.x,
                                   env_object.bounding_box.extent.y,
                                   env_object.bounding_box.extent.z)
            object_labels.append(ObjectLabel(frame=frame,
                                             timestamp=timestamp,
                                             label_type=label_type,
                                             carla_id=env_object.id,
                                             transform=transform,
                                             bounding_box=BoundingBox(Location(0, 0, 0), bbox_extent),
                                             velocity=Vector3d(0, 0, 0)))
        return object_labels
