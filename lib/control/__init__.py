"""
Control module for motor commands and flight controller communication.
"""

from .motor_controller import QuadcopterMotorController, SimulatedMotorController

__all__ = ['QuadcopterMotorController', 'SimulatedMotorController']
