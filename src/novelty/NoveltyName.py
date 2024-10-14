from enum import Enum

class NoveltyName (Enum):
    ORIGINAL = "original"
    ATMOSPHERE = "atmosphere"
    THRUSTER = "thruster"
    GRAVITY = "gravity"
    TURBULENCE = "turbulence"
    SENSOR = "sensor"

    # Novelties from other groups
    ASTEROID = "asteroid"
    BLACKHOLE = "blackhole"
    OVERHANG = "overhang"
    TURRET = "turret"
    WIND = "wind"

def noveltyList():
    novelties = list()
    for novelty in NoveltyName:
        novelties.append(novelty.value)
    return novelties
