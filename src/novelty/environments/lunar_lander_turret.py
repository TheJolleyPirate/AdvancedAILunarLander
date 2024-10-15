"""
Edited by Samuel Price, u7482502

Novelty Changes
-> Added a turret that spawns either the left or the right.
    -> Turret fires small bullets at the lander that collide with it.
    -> Turret bullets break on impact with the moon surface.
    -> Turret aims predictively at the lander.
    -> Turret aim starts at the horizontal.
    -> Turret shows a red aiming laser for where it is currently pointing
    -> Turret has limited ammo, and will stop after it runs out
-> Added additional exit conditions for the simulation, the turret now wins if the lander legs are on the ground for more than 1s, even if still being hit by bullets
-> Added a 60s timeout for the simulation, it seems to get stuck around 1/5000 times.

Solution Approach:

"""
__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from math import pi
from random import randrange

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle, colorize
from gymnasium.utils.step_api_compatibility import step_api_compatibility


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400

DT = 1.0 / FPS

TUR_ROT_SPEED = 1.2 * DT#0.01
LANDER_SIM_OFFSET = (0,3/SCALE)
LANDER_SIM_RAD = 18/SCALE

TUR_MAX_COOLDOWN = int(30*50 * DT)
TUR_AT_TARGET_MIN = int(5*50 * DT)

BULLET_WIDTH = 0.1 / SCALE
BULLET_HEIGHT = 0.1 / SCALE
B_HW = BULLET_WIDTH/2
B_HH = BULLET_HEIGHT/2

TUR_BUL_POLY = [ (-B_HW, B_HH), (B_HW, B_HH), (B_HW, -B_HH), (-B_HW, -B_HH) ]
TUR_BUL_DENSITY = 7

BULLET_MASS = BULLET_WIDTH*BULLET_HEIGHT*TUR_BUL_DENSITY
TUR_BUL_COLOUR = (200,192,192)
TUR_BUL_SPEED = 1 / DT
TUR_BUL_FORCE_MAG =  TUR_BUL_SPEED/ BULLET_MASS #technically force mag not speed

SMART_AIM = True
SMART_AIM_THRESHOLD = 100 / SCALE
SMART_AIM_OFFSET = [0,7/SCALE]

START_HORIZONTAL_AIM = True

GROUND_TIMER_DURATION = 1*FPS

FULL_TIMEOUT = 60*FPS #timeout after a minute

MAX_AMMO = 5

#helpers
def dA(d_1, d_2): return [d_1[0]+d_2[0], d_1[1]+d_2[1]] #duple add
def dS(d1,d2): return [d1[0]-d2[0],d1[1]-d2[1]] #duple subtract
def dSM(s, d): return [s*d[0], s*d[1]] #duple scalar multiply

def coDistance(co1,co2): return math.sqrt(sum([(co1[n]-co2[n])**2 for n in range(len(co1))]))

def ciS(m,arg): return [m*math.cos(arg) , m*math.sin(arg)]
def twoCoAngle(co_1,co_2):
    vec = dS(co_2,co_1)
    if vec[0] == 0: return pi/2 if vec[1] >= 0 else 3*pi/2
    else:
        pre_rads = math.atan(vec[1]/vec[0])
        if vec[0] < 0: pre_rads += pi
        return (pre_rads) % (2*pi)
def vecAngle(vec):
    if vec[0] == 0: return pi if vec[1] >= 0 else 0
    else:
        pre_rads = math.atan(vec[1]/vec[0])
        if vec[0] < 0: pre_rads += pi
        return (pre_rads) % (2*pi)   
def vecMag(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2)

def cMult(a,b,c,d): return a*c - b*d, a*d + b*c
def vecRot(co,d=1): return cMult(*ciS(1,d)+co)# 2*pi is a full rotation
def vecSub(co,d=1):
    co = list(co)
    if co == [0,0]: return [0,0]
    mag = d/math.sqrt(co[0]**2+co[1]**2)
    return [co[0]*mag,co[1]*mag]
def vecChange(co,mD,rD): return vecRot(vecSub(co,mD),rD)
def vecRotLeft(co): return [-co[1], co[0]]
def vecRotRight(co): return [co[1], -co[0]]

def inLine(x,sCo,eCo):#assuming on gradient, checking range
    if sCo[0] != eCo[0]: return min(sCo[0],eCo[0]) < x < max(sCo[0],eCo[0])
    else: return
def circleLine(co1,co2,center,radius):
    out = False
    if co1[0] != co2[0]:
        m = (co1[1]-co2[1])/(co1[0]-co2[0]) 
        b = co1[1] - m*co1[0]
        denom = m**2 + 1
        disc = (b*m - m*center[1] - center[0])**2 - denom*(center[0]**2 + center[1]**2 + b**2 - 2*b*center[1] - radius**2)
        if disc >= 0:
            fNum = center[0] + m*center[1] - b*m
            disc = math.sqrt(disc)
            if inLine((fNum + disc)/denom,co1,co2) or inLine((fNum - disc)/denom,co1,co2): out = True
    else:
        n = co1[0]
        disc = radius**2 - (n-center[0])**2
        if disc >= 0:
            disc = math.sqrt(disc)
            miP = min(co1[1],co2[1]);  maP = max(co1[1],co2[1])
            if miP < center[1] + disc < maP or miP < center[1] -  disc < maP:
                out = True
    return out

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (contact.fixtureA.body in self.env.bullets or contact.fixtureB.body in self.env.bullets):
            self.env.reportBulletCol(contact)
        else:
            if ( (self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body) ):
                self.env.game_over = True
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        if not ((contact.fixtureA.body in self.env.bullets or contact.fixtureB.body in self.env.bullets)):
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = False

class LunarLanderTurret(gym.Env, EzPickle):
    """
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```
    python gymnasium/envs/box2d/lunar_lander.py
    ```
    <!-- To play yourself, run: -->
    <!-- python examples/agents/keyboard_agent.py LunarLander-v2 -->

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments
    To use to the _continuous_ environment, you need to specify the
    `continuous=True` argument like below:
    ```python
    import gymnasium as gym
    env = gym.make(
        "LunarLander-v2",
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    )
    ```
    If `continuous=True` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the
    action space will be `Box(-1, +1, (2,), dtype=np.float32)`.
    The first coordinate of an action determines the throttle of the main engine, while the second
    coordinate specifies the throttle of the lateral boosters.
    Given an action `np.array([main, lateral])`, the main engine will be turned off completely if
    `main < 0` and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (in particular, the
    main engine doesn't work  with less than 50% power).
    Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
    booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
    from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12.

    If `enable_wind=True` is passed, there will be wind effects applied to the lander.
    The wind is generated using the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))`.
    `k` is set to 0.01.
    `C` is sampled randomly between -9999 and 9999.

    `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for `wind_power` is between 0.0 and 20.0.
    `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft. The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground
        give +10 reward points, and -10 if then lose contact; reward
        renormalized to 200; harder initial random push.
    - v0: Initial version


    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thursters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation depentant torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10
    'y': 6.666
    'vx': 5
    'vy': 7.5
    'angle': 1
    'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units)
    'y': (units)
    'vx': (units/second)
    'vy': (units/second)
    'angle': (radians)
    'angular velocity': (radians/second)


    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        gravity: float = -10,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            warnings.warn(
                colorize(
                    f"WARN: wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})",
                    "yellow",
                ),
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            warnings.warn(
                colorize(
                    f"WARN: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})",
                    "yellow",
                ),
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind
        self.wind_idx = np.random.randint(-9999, 9999)
        self.torque_idx = np.random.randint(-9999, 9999)

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Optional[Box2D.b2Body] = None
        self.particles = []

        self.ground_timer = 0

        self.prev_reward = None

        self.continuous = continuous

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self.W, self.H = W, H

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        self.draw_moon_poly = [(0,0)] + [dSM(SCALE, (chunk_x[n], smooth_y[n])) for n in range(CHUNKS)] + [(VIEWPORT_W,0)]

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x101,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x101,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        #set up turret drawing
        t_w = W/150
        t_h = H/10
        self.t_l = t_h*1
        base_chunk = -2 if randrange(0,2) else 2
        if START_HORIZONTAL_AIM: self.t_ori = pi if base_chunk == -2 else 0
        else: self.t_ori = twoCoAngle(self.t_s_pos, self.lander.position)

        self.inactive_t_ori = pi + pi/4 if base_chunk == -2 else 2*pi-pi/4
        

        self.t_b_pos = (chunk_x[base_chunk], smooth_y[base_chunk]) #turret base pos
        self.t_s_pos = (self.t_b_pos[0], self.t_b_pos[1]+t_h) #turret stand pos
        self.d_t_s_pos = dSM(SCALE, self.t_s_pos)
        t_l_shift = [-t_w/2, 0]
        t_r_shift = [t_w/2, 0]
        self.t_d_poly = [dSM(SCALE, co) for co in [dA(self.t_b_pos, t_l_shift), dA(self.t_b_pos, t_r_shift), dA(self.t_s_pos, t_r_shift), dA(self.t_s_pos, t_l_shift)]] #turret draw poly

        self.max_laser_len = max(coDistance(self.t_b_pos, (0,H)), coDistance(self.t_b_pos, (W,H)))
                                 
        #set up turret logic
        self.t_at_target = TUR_AT_TARGET_MIN
        self.t_cooldown = TUR_MAX_COOLDOWN//4
        self.bullets = []
        self.bul_destroy_q = []

        self.timeout_ticks = 0
        self.ammo = MAX_AMMO
        self.tur_active = True

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.lander is not None

        self.updateTurret() #update the turret!

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is a the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation depentant torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        #check if lander legs on ground
        if self.legs[0].ground_contact and self.legs[1].ground_contact: self.ground_timer += 1
        else: self.ground_timer = 0

        self.timeout_ticks += 1

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0 or self.timeout_ticks > FULL_TIMEOUT:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100
        elif self.ground_timer > GROUND_TIMER_DURATION: #the lander might not fall asleep due to bullet hits, but it should still be allowed to end the simulation
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        #draw turret
        if self.tur_active:
            try_end_point = dA(self.t_s_pos, ciS(self.max_laser_len, self.t_ori))
            if not circleLine(self.t_s_pos, try_end_point, dA(self.lander.position, LANDER_SIM_OFFSET), LANDER_SIM_RAD): end_point = try_end_point
            else: end_point = dA(self.t_s_pos, ciS(coDistance(self.t_s_pos, self.lander.position)*1.02, self.t_ori)) #approximation, not exact point
            pygame.draw.line(self.surf, (240,30,30), self.d_t_s_pos, dSM(SCALE, end_point))


        t_barrel_base = ciS(self.t_l/2,self.t_ori)
        t_barrel_rot = dSM(1/3, vecRotRight(t_barrel_base))
        base_draw_pos = dA(dA(self.t_s_pos,dSM(0.2, t_barrel_base)), dSM(-0.5, t_barrel_rot))
        tur_barrel_poly_draw = [dSM(SCALE, dA(co, base_draw_pos)) for co in [t_barrel_base, dA(t_barrel_base, t_barrel_rot), dS(t_barrel_rot, t_barrel_base), dSM(-1, t_barrel_base)]]
        pygame.draw.polygon(self.surf, [50,50,50], tur_barrel_poly_draw)
        pygame.draw.polygon(self.surf, [200,200,200], tur_barrel_poly_draw, 2)

        pygame.draw.polygon(self.surf, [50,50,50], self.t_d_poly)
    

        for obj in self.bullets + self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        #redraw the moon to cover the laser
        pygame.draw.polygon(self.surf, (255,255,255), self.draw_moon_poly)

        self.surf = pygame.transform.flip(self.surf, False, True)


        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
    
    def updateTurret(self):
        #update bullets
        for b in self.bullets:
            if b.position[0] < 0 or b.position[0] > self.W: 
                self.bul_destroy_q.append(b)
        
        for b in self.bul_destroy_q:
            if b in self.bullets:
                self.bullets.remove(b)
                self.world.DestroyBody(b)
                del b
        self.bul_destroy_q = []



        # update orientation
        if self.tur_active:
            if SMART_AIM:
                dist = coDistance(self.t_s_pos, self.lander.position)
                if dist > SMART_AIM_THRESHOLD:
                    travel_steps = dist / (TUR_BUL_SPEED)
                    vel_add = dSM(travel_steps, self.lander.linearVelocity)
                    t = twoCoAngle(self.t_s_pos, dA(SMART_AIM_OFFSET,dA(self.lander.position, vel_add)))
                else:
                    t = twoCoAngle(self.t_s_pos, self.lander.position)
            else:
                t = twoCoAngle(self.t_s_pos, self.lander.position)
        else: t = self.inactive_t_ori

        p = self.t_ori
        right_rot = (t-p)%(2*pi)
        left_rot = (p-t)%(2*pi)

        if right_rot > left_rot: self.t_ori = self.t_ori-min(TUR_ROT_SPEED, left_rot)
        else: self.t_ori = self.t_ori + min(TUR_ROT_SPEED, right_rot)

        if self.tur_active:
        #update cooldown
            if abs(self.t_ori - t) < 0.01: self.t_at_target += 1
            else: self.t_at_target = 0
            self.t_cooldown = max(self.t_cooldown-1,0)

            #update shot
            if self.ammo > 0 and self.t_at_target > TUR_AT_TARGET_MIN and self.t_cooldown == 0:
                self.t_cooldown = TUR_MAX_COOLDOWN
                self.ammo -= 1
                #shooting logic
                new_bullet = self.world.CreateDynamicBody(
                    position=self.t_s_pos[:],
                    angle=0.0,
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=[ dSM(SCALE, co) for co in TUR_BUL_POLY]
                        ),
                        density=TUR_BUL_DENSITY,
                        friction=0.01,
                        categoryBits=0x0100,
                        maskBits=0x0111,  # collide with the ground and the lander
                        restitution=0.0,
                    )
                )
                # new_bullet.ApplyForceToCenter( ciS(TUR_BUL_FORCE_MAG, self.t_ori), True)
                new_vel = ciS(TUR_BUL_SPEED, self.t_ori)
                new_bullet.bullet = True
                new_bullet.linearVelocity[0], new_bullet.linearVelocity[1] = new_vel[0], new_vel[1]
                new_bullet.color1 = TUR_BUL_COLOUR
                new_bullet.color2 = TUR_BUL_COLOUR
                self.bullets.append(new_bullet)

                if self.ammo <= 0: self.tur_active = False

    def reportBulletCol(self, contact):
        if (contact.fixtureA.body in self.bullets):
            bul = contact.fixtureA.body
            other = contact.fixtureB.body
        else:
            bul = contact.fixtureB.body
            other = contact.fixtureA.body
        # print("reported!", bul.active)
        if not other in [self.lander] + self.legs:
            if not bul in self.bul_destroy_q:
                self.bul_destroy_q.append(bul)


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)
    while True:
        a = heuristic(env, s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if steps % 20 == 0 or terminated or truncated:
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward


if __name__ == "__main__":
    env = LunarLanderTurret(render_mode="human")
    demo_heuristic_lander(env, render=True)
