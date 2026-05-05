"""
visualizer.py — Visualisation interactive du carrefour à feux
Gestion Intelligente du Trafic Urbain — Projet IAD & SMA 2025-2026

Contrôles :
  [1] Mode Manuel       — M = maintenir, C = changer
  [2] Mode Baseline     — feu à durée fixe (période 5 pas)
  [3] Mode Q-Learning   — agent entraîné (outputs/agent_*.pkl)
  [C] Mode Comparaison  — Baseline vs Q-Learning côte à côte
  [4] Trafic équilibré  (λ = 0.4 partout)
  [5] Trafic asymétrique (λ_NS = 0.7, λ_EW = 0.2)
  [SPACE]  Pause / Reprendre (fige l'écran pour expliquer)
  [+/-]    Vitesse de simulation
  [R]      Réinitialiser
  [ESC]    Quitter
"""

import sys, os, pickle, math, random
import pygame
from simulation import IntersectionEnv, PHASE_NS, PHASE_EW, PHASE_ORANGE
from agent import BaselineAgent

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = ( 28,  28,  33)
ROAD_COL    = ( 52,  52,  58)
ROAD_EDGE   = ( 38,  38,  44)
GRASS       = ( 50,  80,  46)
LANE_MARK   = (190, 190,  85)

RED_LIGHT   = (225,  42,  42)
AMBER_LIGHT = (232, 158,  18)
GREEN_LIGHT = ( 42, 198,  72)
LIGHT_OFF   = ( 45,  45,  45)
LIGHT_BODY  = ( 20,  20,  20)

WHITE    = (255, 255, 255)
BLACK    = (  0,   0,   0)
GRAY     = (145, 145, 145)
YELLOW   = (255, 218,  48)
PANEL_BG = ( 34,  34,  42)

CAR_COLORS = [
    (212,  52,  52), ( 52, 112, 212), ( 52, 192,  72),
    (212, 172,  38), (172,  52, 212), ( 38, 192, 192),
    (212, 112,  38), (192, 192, 192), (255, 138,  78),
    (255, 200,  80), ( 80, 200, 255), (200,  80, 255),
]

# ── Layout ────────────────────────────────────────────────────────────────────
W, H      = 940, 740
CX, CY    = 460, 355
ROAD_W    = 88          # half-width of road (pixels)
LANE_W    = ROAD_W // 2 # one lane width
CAR_W     = 18
CAR_LEN   = 32
CAR_GAP   = 8
PANEL_X   = 680
STOP_OFF  = 6           # stop line inset from intersection edge
SPAWN_D   = 360         # how far back cars spawn from stop line

# ── Comparison mode ───────────────────────────────────────────────────────────
COMPARISON_STEPS = 200  # Number of steps to run comparison

# ── Timing ────────────────────────────────────────────────────────────────────
FPS     = 60
STEP_MS = [2500, 1500, 800, 400, 180]   # ms per sim step (speed levels 0-4)

LAMBDAS_BAL  = (0.4, 0.4, 0.4, 0.4)
LAMBDAS_ASYM = (0.7, 0.7, 0.2, 0.2)

TURN_PROBS = [0.55, 0.225, 0.225]   # straight / left / right


# ─────────────────────────────────────────────────────────────────────────────
# Road geometry  (right-hand traffic, French convention)
# ─────────────────────────────────────────────────────────────────────────────
#
# The road is split into two lanes of width LANE_W = ROAD_W//2.
# Each direction uses the RIGHT lane (when facing the direction of travel).
#
#   North cars travel SOUTH  → right lane = west half  → x = CX - LH
#   South cars travel NORTH  → right lane = east half  → x = CX + LH
#   East  cars travel WEST   → right lane = north half → y = CY - LH
#   West  cars travel EAST   → right lane = south half → y = CY + LH
#
# LH = LANE_W // 2  (offset of lane centre from road centre)

LH = LANE_W // 2

# ── Entry / exit points ───────────────────────────────────────────────────────
# Entry = where the path starts, at the stop line
# Exit  = where the path ends, far off screen so cars drive all the way out
#
# RIGHT-HAND TRAFFIC RULES:
# - Cars stay on the RIGHT side of the road (when facing direction of travel)
# - North cars (going south): use WEST half (x = CX - LH)
# - South cars (going north): use EAST half (x = CX + LH)  
# - East cars (going west): use NORTH half (y = CY - LH)
# - West cars (going east): use SOUTH half (y = CY + LH)
#
# When exiting after a turn, cars must be in the CORRECT lane for that direction:
# - Exiting south: WEST half (x = CX - LH)
# - Exiting north: EAST half (x = CX + LH)
# - Exiting west: NORTH half (y = CY - LH)
# - Exiting east: SOUTH half (y = CY + LH)

ENTRY = {
    0: (CX - LH,     CY - ROAD_W),   # North → south (west half)
    1: (CX + LH,     CY + ROAD_W),   # South → north (east half)
    2: (CX + ROAD_W, CY - LH),       # East  → west (north half)
    3: (CX - ROAD_W, CY + LH),       # West  → east (south half)
}

# Exit points - far off screen (400px past intersection)
# MUST use correct lane for exit direction!
EXIT_DIST = 400

EXIT = {
    # North (dir 0) - enters from north going south
    (0, 0): (CX - LH,     CY + ROAD_W + EXIT_DIST),   # Straight → south (WEST half) ✓
    (0, 1): (CX + ROAD_W + EXIT_DIST, CY + LH),       # Left → east (SOUTH half) ✓
    (0, 2): (CX - ROAD_W - EXIT_DIST, CY - LH),       # Right → west (NORTH half) ✓

    # South (dir 1) - enters from south going north  
    (1, 0): (CX + LH,     CY - ROAD_W - EXIT_DIST),   # Straight → north (EAST half) ✓
    (1, 1): (CX - ROAD_W - EXIT_DIST, CY - LH),       # Left → west (NORTH half) ✓
    (1, 2): (CX + ROAD_W + EXIT_DIST, CY + LH),       # Right → east (SOUTH half) ✓

    # East (dir 2) - enters from east going west
    (2, 0): (CX - ROAD_W - EXIT_DIST, CY - LH),       # Straight → west (NORTH half) ✓
    (2, 1): (CX - LH,     CY + ROAD_W + EXIT_DIST),   # Left → south (WEST half) ✓
    (2, 2): (CX + LH,     CY - ROAD_W - EXIT_DIST),   # Right → north (EAST half) ✓

    # West (dir 3) - enters from west going east
    (3, 0): (CX + ROAD_W + EXIT_DIST, CY + LH),       # Straight → east (SOUTH half) ✓
    (3, 1): (CX + LH,     CY - ROAD_W - EXIT_DIST),   # Left → north (EAST half) ✓
    (3, 2): (CX - LH,     CY + ROAD_W + EXIT_DIST),   # Right → south (WEST half) ✓
}

# ── Right-turn near corners (the corner the car sweeps around) ────────────────
# Each right turn stays in the near quadrant of the intersection.
# The corner is at the intersection of the entry road edge and the exit road edge.
RIGHT_CORNER = {
    0: (CX - ROAD_W, CY - ROAD_W),   # N→W  NW corner
    1: (CX + ROAD_W, CY + ROAD_W),   # S→E  SE corner
    2: (CX + ROAD_W, CY - ROAD_W),   # E→N  NE corner
    3: (CX - ROAD_W, CY + ROAD_W),   # W→S  SW corner
}

# ── Left-turn pivot point (car advances to near-centre before turning) ────────
# The car drives straight to a point just past the centre of the intersection,
# then arcs to the exit lane.  We place the pivot slightly offset toward the
# entry side so the car doesn't cross the centre line of the exit road.
LEFT_PIVOT = {
    0: (CX - LH//2, CY - LH//2),    # N left pivot (slightly NW of centre)
    1: (CX + LH//2, CY + LH//2),    # S left pivot (slightly SE of centre)
    2: (CX + LH//2, CY - LH//2),    # E left pivot (slightly NE of centre)
    3: (CX - LH//2, CY + LH//2),    # W left pivot (slightly SW of centre)
}


def _qbez(p0, p1, p2, t):
    """Quadratic Bezier point."""
    u = 1 - t
    return (u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0],
            u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1])


def _sample_qbez(p0, p1, p2, n):
    return [_qbez(p0, p1, p2, i / (n - 1)) for i in range(n)]


def _lerp_seg(p0, p1, n):
    return [(p0[0] + (p1[0]-p0[0])*i/(n-1),
             p0[1] + (p1[1]-p0[1])*i/(n-1)) for i in range(n)]


def _build_paths(n_turn=60, n_straight=80):
    """
    Build simple, natural driving paths for all 12 (direction, turn) combinations.
    
    Straight: Just drive straight through to the far exit
    Right turn: Smooth arc around the corner
    Left turn: Drive to center, then arc to exit lane
    """
    paths = {}

    for d in range(4):
        entry = ENTRY[d]
        
        # ── STRAIGHT ──────────────────────────────────────────────────────────
        # Simple straight line all the way through
        exit_straight = EXIT[(d, 0)]
        paths[(d, 0)] = _lerp_seg(entry, exit_straight, n_straight)

        # ── RIGHT TURN ────────────────────────────────────────────────────────
        # Smooth arc from entry to exit using corner as control point
        exit_right = EXIT[(d, 2)]
        corner = RIGHT_CORNER[d]
        # Control point slightly inside the corner for smooth arc
        ctrl_x = corner[0] + (CX - corner[0]) * 0.15
        ctrl_y = corner[1] + (CY - corner[1]) * 0.15
        ctrl = (ctrl_x, ctrl_y)
        paths[(d, 2)] = _sample_qbez(entry, ctrl, exit_right, n_turn)

        # ── LEFT TURN ─────────────────────────────────────────────────────────
        # Two segments: straight to center, then arc to exit
        exit_left = EXIT[(d, 1)]
        pivot = LEFT_PIVOT[d]
        
        # Segment 1: Straight to pivot (30% of points)
        n1 = n_turn // 3
        seg1 = _lerp_seg(entry, pivot, n1)
        
        # Segment 2: Arc from pivot to exit (70% of points)
        n2 = n_turn - n1
        # Control point for the arc - midpoint between pivot and exit, offset toward center
        ctrl_x = (pivot[0] + exit_left[0]) / 2
        ctrl_y = (pivot[1] + exit_left[1]) / 2
        seg2 = _sample_qbez(pivot, (ctrl_x, ctrl_y), exit_left, n2)
        
        paths[(d, 1)] = seg1 + seg2

    return paths


PATHS = _build_paths()


# ─────────────────────────────────────────────────────────────────────────────
# Car
# ─────────────────────────────────────────────────────────────────────────────
class Car:
    """
    States:
      approach  — driving from spawn toward stop line, decelerating
      queued    — stopped at stop line, waiting for green
      crossing  — following Bezier path through intersection
      done      — off screen, ready to remove
    """
    MAX_SPEED  = 2.6
    ACCEL      = 0.09
    DECEL      = 0.20
    BRAKE_DIST = 90

    # Per-direction approach config.
    # stop = centre of car when fully stopped BEHIND the stop line.
    # The car front is at (stop + sign * CAR_LEN//2), which must be <= stop_line.
    # stop_line for N = CY - ROAD_W.  We want front at CY-ROAD_W-STOP_OFF.
    # So centre = CY - ROAD_W - STOP_OFF - CAR_LEN//2.
    _S = STOP_OFF + CAR_LEN // 2   # total setback from road edge to car centre
    APPROACH = {
        0: dict(lx=CX - LH, ly=CY - ROAD_W - SPAWN_D,
                axis='y', sign=+1, stop=CY - ROAD_W - _S),
        1: dict(lx=CX + LH, ly=CY + ROAD_W + SPAWN_D,
                axis='y', sign=-1, stop=CY + ROAD_W + _S),
        2: dict(lx=CX + ROAD_W + SPAWN_D, ly=CY - LH,
                axis='x', sign=-1, stop=CX + ROAD_W + _S),
        3: dict(lx=CX - ROAD_W - SPAWN_D, ly=CY + LH,
                axis='x', sign=+1, stop=CX - ROAD_W - _S),
    }

    def __init__(self, direction, color):
        self.direction = direction
        self.color     = color
        self.turn      = random.choices([0, 1, 2], weights=TURN_PROBS)[0]

        cfg        = self.APPROACH[direction]
        self.axis  = cfg['axis']
        self.sign  = cfg['sign']
        self.stop  = float(cfg['stop'])
        self.x     = float(cfg['lx'])
        self.y     = float(cfg['ly'])
        self.speed = 0.0
        self.angle = {0: 0.0, 1: 180.0, 2: 90.0, 3: 270.0}[direction]

        self.state    = 'approach'
        self.path     = None
        self.path_idx = 0
        self.path_spd = 0.0
        self.leader   = None   # Car directly ahead in queue
        self.wait_frames = 0   # Frames to wait before crossing

    # ── position helpers ─────────────────────────────────────────────────────
    @property
    def pos(self):
        return self.y if self.axis == 'y' else self.x

    @pos.setter
    def pos(self, v):
        if self.axis == 'y': self.y = v
        else:                 self.x = v

    def dist_to_stop(self):
        return self.sign * (self.stop - self.pos)

    def dist_to_leader(self):
        if self.leader is None or self.leader.state not in ('approach', 'queued'):
            return float('inf')
        return abs(self.pos - self.leader.pos) - CAR_LEN

    # ── update ───────────────────────────────────────────────────────────────
    def update(self, green: bool):
        if self.state == 'done':
            return
        if self.state == 'crossing':
            self._cross()
            return

        # Count down wait frames
        if self.wait_frames > 0:
            self.wait_frames -= 1
            return  # Don't move while waiting

        d_stop   = self.dist_to_stop()
        d_leader = self.dist_to_leader()
        eff_dist = max(0.0, min(d_stop, d_leader - CAR_GAP))

        # Front car at stop line on green → start crossing
        if green and self.leader is None and d_stop <= 1.5:
            self._start_crossing()
            return

        # Desired speed
        if eff_dist < self.BRAKE_DIST:
            desired = self.MAX_SPEED * max(0.06, (eff_dist / self.BRAKE_DIST) ** 0.6)
        else:
            desired = self.MAX_SPEED

        if self.speed < desired:
            self.speed = min(self.speed + self.ACCEL, desired)
        else:
            self.speed = max(self.speed - self.DECEL, desired)
        self.speed = max(0.0, self.speed)

        self.pos += self.sign * min(self.speed, eff_dist)
        self.state = 'queued' if d_stop <= 1.0 else 'approach'

    def _start_crossing(self):
        self.state    = 'crossing'
        self.path     = PATHS[(self.direction, self.turn)]
        self.path_idx = 0
        self.path_spd = max(self.speed, 0.5)
        self.leader   = None
        
        # Snap position to path start for clean entry
        if self.path:
            self.x, self.y = self.path[0]
        
        # Snap angle to the first path segment so there's no spin-up
        if len(self.path) >= 2:
            dx = self.path[1][0] - self.path[0][0]
            dy = self.path[1][1] - self.path[0][1]
            if abs(dx) + abs(dy) > 0.01:
                self.angle = -math.degrees(math.atan2(dy, dx)) + 90

    def _cross(self):
        """Follow the path through intersection - simple and smooth."""
        if self.path_idx >= len(self.path) - 1:
            self.state = 'done'
            return

        # Accelerate to max speed
        self.path_spd = min(self.path_spd + self.ACCEL, self.MAX_SPEED)
        remaining = self.path_spd

        # Follow the path point by point
        while remaining > 0 and self.path_idx < len(self.path) - 1:
            nx, ny = self.path[self.path_idx + 1]
            dx, dy = nx - self.x, ny - self.y
            dist = math.hypot(dx, dy)
            
            if dist < 0.1:
                self.path_idx += 1
                continue
                
            if dist <= remaining:
                # Move to next point
                self.x, self.y = nx, ny
                self.path_idx += 1
                remaining -= dist
            else:
                # Move partway to next point
                self.x += dx * (remaining / dist)
                self.y += dy * (remaining / dist)
                remaining = 0

        # Update angle to point toward next waypoint
        if self.path_idx < len(self.path) - 1:
            # Look ahead a bit for smoother angle
            look_idx = min(self.path_idx + 3, len(self.path) - 1)
            nx, ny = self.path[look_idx]
            dx, dy = nx - self.x, ny - self.y
            
            if abs(dx) + abs(dy) > 0.5:
                target_angle = -math.degrees(math.atan2(dy, dx)) + 90
                # Smooth angle transition
                angle_diff = (target_angle - self.angle + 180) % 360 - 180
                self.angle += angle_diff * 0.3  # Smooth but responsive

        # Check if done
        if self.path_idx >= len(self.path) - 1:
            self.state = 'done'

    # ── draw ─────────────────────────────────────────────────────────────────
    def draw(self, surf):
        if self.state == 'done':
            return
        body = pygame.Surface((CAR_W, CAR_LEN), pygame.SRCALPHA)
        c = self.color
        pygame.draw.rect(body, c, (0, 3, CAR_W, CAR_LEN-6), border_radius=5)
        pygame.draw.rect(body, (165, 210, 255, 195), (3, 5, CAR_W-6, 8), border_radius=2)
        pygame.draw.rect(body, (165, 210, 255, 145), (3, CAR_LEN-13, CAR_W-6, 6), border_radius=2)
        lighter = tuple(min(v+45, 255) for v in c)
        pygame.draw.rect(body, (*lighter, 110), (5, 14, CAR_W-10, CAR_LEN-28), border_radius=3)
        for wx, wy in [(0,5),(CAR_W-4,5),(0,CAR_LEN-13),(CAR_W-4,CAR_LEN-13)]:
            pygame.draw.rect(body, (18,18,18), (wx, wy, 4, 8), border_radius=1)
        rot  = pygame.transform.rotate(body, self.angle)
        rect = rot.get_rect(center=(int(self.x), int(self.y)))
        surf.blit(rot, rect)


# ─────────────────────────────────────────────────────────────────────────────
# Car Manager
# ─────────────────────────────────────────────────────────────────────────────
class CarManager:
    def __init__(self):
        self.cars: list[Car] = []
        self._cidx = 0

    def reset(self):
        self.cars = []
        self._cidx = 0

    def _color(self):
        c = CAR_COLORS[self._cidx % len(CAR_COLORS)]
        self._cidx += 1
        return c

    def _queue(self, d):
        """Front-first list of queued/approaching cars for direction d."""
        return sorted(
            [c for c in self.cars if c.direction == d
             and c.state in ('approach', 'queued')],
            key=lambda c: -c.sign * c.pos
        )

    def sync(self, queues):
        for d in range(4):
            q      = self._queue(d)
            target = int(queues[d])

            # Spawn new cars at the back
            while len(q) < target:
                car = Car(d, self._color())
                if q:
                    last = q[-1]
                    if car.axis == 'y':
                        car.y = last.y - car.sign * (CAR_LEN + CAR_GAP + 4)
                    else:
                        car.x = last.x - car.sign * (CAR_LEN + CAR_GAP + 4)
                self.cars.append(car)
                q = self._queue(d)

            # When queue shrinks, cars are released but not forced to cross
            # They will start crossing naturally when they reach the stop line
            # (handled by _assign_leaders setting leader=None for released cars)

    def _assign_leaders(self, target_queues):
        """Assign leaders based on simulation queue sizes."""
        for d in range(4):
            q = self._queue(d)
            target = int(target_queues[d])
            for i, car in enumerate(q):
                # Cars beyond the target queue size have no leader (they're released)
                if i < target:
                    car.leader = q[i-1] if i > 0 else None
                else:
                    car.leader = None
                    # Add wait time so cars don't all rush into intersection at once
                    # Each car waits a bit for the one ahead to clear
                    if i == target and i > 0:
                        # First released car waits for previous car to get into intersection
                        car.wait_frames = 15  # ~0.25 seconds at 60fps

    def update(self, phase, queues):
        # Determine which directions have green light
        if phase == PHASE_NS:
            green = [0, 1]  # North and South
        elif phase == PHASE_EW:
            green = [2, 3]  # East and West
        else:
            green = []  # Orange phase - no one should cross
        
        self._assign_leaders(queues)
        
        for car in self.cars:
            car.update(green=car.direction in green)
        self.cars = [c for c in self.cars if c.state != 'done']

    def draw(self, surf):
        for c in self.cars:
            if c.state != 'crossing': c.draw(surf)
        for c in self.cars:
            if c.state == 'crossing': c.draw(surf)


# ─────────────────────────────────────────────────────────────────────────────
# Traffic light
# ─────────────────────────────────────────────────────────────────────────────
def draw_traffic_light(surf, x, y, phase, controls_ns, countdown, font):
    bw, bh = 24, 66
    pygame.draw.rect(surf, (70,70,70), (x-2, y, 4, 72))
    pygame.draw.rect(surf, LIGHT_BODY, (x-bw//2, y-bh, bw, bh), border_radius=5)
    pygame.draw.rect(surf, (52,52,52), (x-bw//2, y-bh, bw, bh), border_radius=5, width=1)

    if phase == PHASE_ORANGE:
        cols = [LIGHT_OFF, AMBER_LIGHT, LIGHT_OFF]
    elif (phase == PHASE_NS and controls_ns) or (phase == PHASE_EW and not controls_ns):
        cols = [LIGHT_OFF, LIGHT_OFF, GREEN_LIGHT]
    else:
        cols = [RED_LIGHT, LIGHT_OFF, LIGHT_OFF]

    bys = [y-bh+11, y-bh+33, y-bh+55]
    for col, by in zip(cols, bys):
        if col != LIGHT_OFF:
            g = pygame.Surface((30,30), pygame.SRCALPHA)
            pygame.draw.circle(g, (*col, 50), (15,15), 15)
            surf.blit(g, (x-15, by-15))
        pygame.draw.circle(surf, col,   (x, by), 8)
        pygame.draw.circle(surf, BLACK, (x, by), 8, 1)

    if countdown and countdown > 0:
        if phase == PHASE_ORANGE:
            by2, bc = bys[1], AMBER_LIGHT
        elif (phase == PHASE_NS and controls_ns) or (phase == PHASE_EW and not controls_ns):
            by2, bc = bys[2], GREEN_LIGHT
        else:
            by2, bc = bys[0], RED_LIGHT
        bx2 = x + 14
        by2 -= 9
        pygame.draw.circle(surf, (16,16,16), (bx2, by2), 9)
        pygame.draw.circle(surf, bc,         (bx2, by2), 9, 2)
        t = font.render(str(countdown), True, WHITE)
        surf.blit(t, t.get_rect(center=(bx2, by2)))


# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────
def draw_scene(surf, phase, countdown, font_cnt, paused):
    surf.fill(BG)

    # Grass corners
    for rx, ry, rw, rh in [
        (0,           0,           CX-ROAD_W, CY-ROAD_W),
        (CX+ROAD_W,   0,           W-CX-ROAD_W, CY-ROAD_W),
        (0,           CY+ROAD_W,   CX-ROAD_W, H-CY-ROAD_W),
        (CX+ROAD_W,   CY+ROAD_W,   W-CX-ROAD_W, H-CY-ROAD_W),
    ]:
        pygame.draw.rect(surf, GRASS, (rx, ry, rw, rh))

    # Roads
    pygame.draw.rect(surf, ROAD_COL, (CX-ROAD_W, 0,       ROAD_W*2, H))
    pygame.draw.rect(surf, ROAD_COL, (0,          CY-ROAD_W, W, ROAD_W*2))

    # Road edges
    for rx, ry, rw, rh in [
        (CX-ROAD_W-2, 0,           2, CY-ROAD_W),
        (CX+ROAD_W,   0,           2, CY-ROAD_W),
        (CX-ROAD_W-2, CY+ROAD_W,   2, H-CY-ROAD_W),
        (CX+ROAD_W,   CY+ROAD_W,   2, H-CY-ROAD_W),
        (0,           CY-ROAD_W-2, CX-ROAD_W, 2),
        (0,           CY+ROAD_W,   CX-ROAD_W, 2),
        (CX+ROAD_W,   CY-ROAD_W-2, W-CX-ROAD_W, 2),
        (CX+ROAD_W,   CY+ROAD_W,   W-CX-ROAD_W, 2),
    ]:
        pygame.draw.rect(surf, ROAD_EDGE, (rx, ry, rw, rh))

    # Dashed centre lines
    dash, gap = 18, 12
    y = 0
    while y < H:
        if not (CY-ROAD_W < y < CY+ROAD_W):
            pygame.draw.rect(surf, LANE_MARK, (CX-1, y, 2, dash))
        y += dash + gap
    x = 0
    while x < W:
        if not (CX-ROAD_W < x < CX+ROAD_W):
            pygame.draw.rect(surf, LANE_MARK, (x, CY-1, dash, 2))
        x += dash + gap

    # Stop lines
    pygame.draw.rect(surf, WHITE, (CX-ROAD_W, CY-ROAD_W-4, ROAD_W, 4))
    pygame.draw.rect(surf, WHITE, (CX,        CY+ROAD_W,   ROAD_W, 4))
    pygame.draw.rect(surf, WHITE, (CX+ROAD_W, CY-ROAD_W,   4, ROAD_W))
    pygame.draw.rect(surf, WHITE, (CX-ROAD_W-4, CY,        4, ROAD_W))

    # Crosswalk stripes
    for i in range(4):
        sx = CX - ROAD_W + i*(ROAD_W//4) + 4
        pygame.draw.rect(surf, (75,75,80), (sx, CY-ROAD_W-14, 8, 10))
        pygame.draw.rect(surf, (75,75,80), (sx, CY+ROAD_W+4,  8, 10))
    for i in range(4):
        sy = CY - ROAD_W + i*(ROAD_W//4) + 4
        pygame.draw.rect(surf, (75,75,80), (CX-ROAD_W-14, sy, 10, 8))
        pygame.draw.rect(surf, (75,75,80), (CX+ROAD_W+4,  sy, 10, 8))

    # Traffic lights (4 corners - outside the road)
    # Place them on the grass corners, not in the street
    light_offset = 35  # Distance from intersection corner
    draw_traffic_light(surf, CX-ROAD_W-light_offset, CY-ROAD_W-light_offset, phase, True,  countdown, font_cnt)
    draw_traffic_light(surf, CX+ROAD_W+light_offset, CY+ROAD_W+light_offset, phase, True,  countdown, font_cnt)
    draw_traffic_light(surf, CX+ROAD_W+light_offset, CY-ROAD_W-light_offset, phase, False, countdown, font_cnt)
    draw_traffic_light(surf, CX-ROAD_W-light_offset, CY+ROAD_W+light_offset, phase, False, countdown, font_cnt)

    # Simple pause overlay - just darken the screen
    if paused:
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 120))
        surf.blit(ov, (0, 0))
        
        # Show "PAUSED" text
        font_lg = pygame.font.SysFont("monospace", 32, bold=True)
        text = font_lg.render("PAUSED", True, (255, 255, 255))
        text_rect = text.get_rect(center=(PANEL_X // 2, H // 2))
        surf.blit(text, text_rect)


# No slides - just simple pause/unpause


# ─────────────────────────────────────────────────────────────────────────────
# Info panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_comparison_results(surf, baseline_stats, ql_stats, traffic_name, font_sm, font_md, font_lg):
    """Affiche les résultats de comparaison entre Baseline et Q-Learning."""
    # Fond semi-transparent
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surf.blit(overlay, (0, 0))
    
    # Panneau de résultats
    panel_w, panel_h = 700, 500
    panel_x, panel_y = (W - panel_w) // 2, (H - panel_h) // 2
    pygame.draw.rect(surf, PANEL_BG, (panel_x, panel_y, panel_w, panel_h), border_radius=10)
    pygame.draw.rect(surf, (100, 100, 120), (panel_x, panel_y, panel_w, panel_h), 3, border_radius=10)
    
    x0 = panel_x + 30
    y = panel_y + 20
    
    def lbl(text, color=WHITE, font=font_sm, center=False):
        nonlocal y
        t = font.render(text, True, color)
        if center:
            surf.blit(t, (panel_x + (panel_w - t.get_width()) // 2, y))
        else:
            surf.blit(t, (x0, y))
        y += t.get_height() + 5
    
    def sp(n=10): nonlocal y; y += n
    
    # Titre
    lbl("COMPARAISON BASELINE vs Q-LEARNING", (170, 196, 255), font_lg, center=True)
    sp(5)
    lbl(f"Scénario : {traffic_name}", (170, 170, 255), font_md, center=True)
    sp(15)
    
    # Métriques
    b_avg_wait = baseline_stats['avg_waiting']
    q_avg_wait = ql_stats['avg_waiting']
    b_reward = baseline_stats['total_reward']
    q_reward = ql_stats['total_reward']
    
    # Calcul du gain
    wait_gain = ((b_avg_wait - q_avg_wait) / b_avg_wait * 100) if b_avg_wait > 0 else 0
    reward_gain = ((q_reward - b_reward) / abs(b_reward) * 100) if b_reward != 0 else 0
    
    # Tableau de comparaison
    col1_x = x0
    col2_x = x0 + 220
    col3_x = x0 + 400
    
    # En-têtes
    lbl("Métrique", (200, 200, 200), font_md)
    y_header = y - font_md.get_height() - 5
    t = font_md.render("Baseline", True, (255, 120, 70))
    surf.blit(t, (col2_x, y_header))
    t = font_md.render("Q-Learning", True, (70, 216, 110))
    surf.blit(t, (col3_x, y_header))
    sp(10)
    
    # Ligne de séparation
    pygame.draw.line(surf, (100, 100, 120), (x0, y), (x0 + 640, y), 2)
    sp(15)
    
    # Attente moyenne
    lbl("Attente moyenne", WHITE, font_md)
    y_line = y - font_md.get_height() - 5
    t = font_md.render(f"{b_avg_wait:.2f} veh/pas", True, (255, 120, 70))
    surf.blit(t, (col2_x, y_line))
    t = font_md.render(f"{q_avg_wait:.2f} veh/pas", True, (70, 216, 110))
    surf.blit(t, (col3_x, y_line))
    sp(10)
    
    # Récompense totale
    lbl("Récompense totale", WHITE, font_md)
    y_line = y - font_md.get_height() - 5
    t = font_md.render(f"{b_reward:.0f}", True, (255, 120, 70))
    surf.blit(t, (col2_x, y_line))
    t = font_md.render(f"{q_reward:.0f}", True, (70, 216, 110))
    surf.blit(t, (col3_x, y_line))
    sp(10)
    
    # Nombre d'étapes
    lbl("Étapes simulées", WHITE, font_md)
    y_line = y - font_md.get_height() - 5
    t = font_md.render(f"{baseline_stats['steps']}", True, GRAY)
    surf.blit(t, (col2_x, y_line))
    t = font_md.render(f"{ql_stats['steps']}", True, GRAY)
    surf.blit(t, (col3_x, y_line))
    sp(20)
    
    # Ligne de séparation
    pygame.draw.line(surf, (100, 100, 120), (x0, y), (x0 + 640, y), 2)
    sp(15)
    
    # Gains
    lbl("PERFORMANCE Q-LEARNING", (255, 218, 48), font_lg, center=True)
    sp(10)
    
    # Gain sur l'attente
    gain_color = (70, 216, 110) if wait_gain > 0 else (225, 42, 42)
    gain_text = f"{'Réduction' if wait_gain > 0 else 'Augmentation'} de l'attente : {abs(wait_gain):.1f}%"
    lbl(gain_text, gain_color, font_md, center=True)
    sp(5)
    
    # Gain sur la récompense
    gain_color = (70, 216, 110) if reward_gain > 0 else (225, 42, 42)
    gain_text = f"{'Amélioration' if reward_gain > 0 else 'Dégradation'} du reward : {abs(reward_gain):.1f}%"
    lbl(gain_text, gain_color, font_md, center=True)
    sp(20)
    
    # Interprétation
    if wait_gain > 15:
        interpretation = "Gain SIGNIFICATIF - Q-Learning optimise bien le trafic asymétrique"
        interp_color = (70, 216, 110)
    elif wait_gain > 5:
        interpretation = "Gain MODÉRÉ - Q-Learning améliore légèrement les performances"
        interp_color = (255, 218, 48)
    elif wait_gain > -5:
        interpretation = "Performances SIMILAIRES - Trafic équilibré, alternance régulière optimale"
        interp_color = (170, 170, 255)
    else:
        interpretation = "Baseline MEILLEUR - Vérifier les paramètres Q-Learning"
        interp_color = (225, 42, 42)
    
    lbl(interpretation, interp_color, font_sm, center=True)
    sp(30)
    
    # Instructions
    lbl("Appuyez sur [ESPACE] pour continuer", (200, 200, 200), font_sm, center=True)


def draw_panel(surf, phase, queues, step, reward_total,
               mode_name, traffic_name, speed_idx, countdown,
               last_action, manual_action, font_sm, font_md, font_lg):
    from simulation import MAX_QUEUE

    pw = W - PANEL_X
    pygame.draw.rect(surf, PANEL_BG, (PANEL_X, 0, pw, H))
    pygame.draw.line(surf, (68, 68, 95), (PANEL_X, 0), (PANEL_X, H), 2)

    x0 = PANEL_X + 14
    y  = 14

    def lbl(text, color=WHITE, font=font_sm):
        nonlocal y
        t = font.render(text, True, color)
        surf.blit(t, (x0, y))
        y += t.get_height() + 5

    def sp(n=10): nonlocal y; y += n

    lbl("CARREFOUR IA", (170, 196, 255), font_lg)
    sp(2)

    mc = {"Manuel": (255, 196, 70), "Baseline": (255, 120, 70),
          "Q-Learning": (70, 216, 110), "Comparaison": (255, 218, 48)}.get(mode_name, WHITE)
    lbl(f"Mode    : {mode_name}", mc, font_md)
    lbl(f"Trafic  : {traffic_name}", (170, 170, 255), font_md)
    sp(4)

    ps = {PHASE_NS: "N/S  VERT", PHASE_EW: "E/O  VERT",
          PHASE_ORANGE: "ORANGE"}[phase]
    pc = {PHASE_NS: GREEN_LIGHT, PHASE_EW: GREEN_LIGHT,
          PHASE_ORANGE: AMBER_LIGHT}[phase]
    lbl(f"Phase   : {ps}", pc, font_md)

    # Countdown bar
    bar_w = pw - 28
    if countdown is not None and countdown > 0:
        frac = min(1.0, countdown / 8)
        pygame.draw.rect(surf, (50,50,60), (x0, y, bar_w, 10), border_radius=4)
        pygame.draw.rect(surf, pc, (x0, y, int(bar_w*frac), 10), border_radius=4)
    y += 14
    sp(4)

    # Last agent action
    act_str = {0: "Maintenir ▬", 1: "Changer  ↺"}.get(last_action, "—")
    act_col = (255, 200, 80) if last_action == 1 else (150, 200, 150)
    lbl(f"Action  : {act_str}", act_col)
    sp(2)

    lbl(f"Étape   : {step}",               GRAY)
    lbl(f"Attente : {int(sum(queues))} veh", GRAY)
    lbl(f"Reward  : {reward_total:.0f}",    GRAY)
    sp(4)

    speeds = ["▸○○○○", "▸▸○○○", "▸▸▸○○", "▸▸▸▸○", "▸▸▸▸▸"]
    lbl(f"Vitesse : {speeds[speed_idx]}", (196, 196, 196))
    sp(14)

    # Queue bars
    lbl("Files d'attente :", WHITE, font_md)
    sp(4)
    dirs    = ["Nord ↓", "Sud  ↑", "Est  ←", "Ouest→"]
    dcolors = [(210,70,70),(70,170,210),(70,210,110),(210,170,50)]
    for i, (d, dc) in enumerate(zip(dirs, dcolors)):
        q  = int(queues[i])
        bw = int(bar_w * q / MAX_QUEUE)
        pygame.draw.rect(surf, (50,50,60), (x0, y, bar_w, 18), border_radius=3)
        if bw > 0:
            pygame.draw.rect(surf, dc, (x0, y, bw, 18), border_radius=3)
        t = font_sm.render(f"{d}  {q}/{MAX_QUEUE}", True, WHITE)
        surf.blit(t, (x0+4, y+2))
        y += 24

    sp(14)
    lbl("── Contrôles ──", (125, 125, 150), font_sm)
    lbl("[1/2/3]  Mode agent",  GRAY)
    lbl("[C]      Comparaison", (255, 218, 48))
    lbl("[4/5]    Trafic",      GRAY)
    lbl("[+/-]    Vitesse",     GRAY)
    lbl("[SPACE]  Pause",       (255, 218, 48))
    lbl("[R]      Reset",       GRAY)
    lbl("[ESC]    Quitter",     GRAY)

    if mode_name == "Manuel":
        sp(8)
        lbl("[M]  Maintenir phase", (255, 210, 70))
        lbl("[C]  Changer phase",   (255, 210, 70))
        if manual_action is not None:
            lbl("→ " + ("Changer" if manual_action == 1 else "Maintenir"),
                (255, 255, 105), font_md)

    sp(10)
    lbl("Virages :", (125, 125, 150), font_sm)
    lbl(f"  Tout droit {int(TURN_PROBS[0]*100)}%", GRAY)
    lbl(f"  Gauche     {int(TURN_PROBS[1]*100)}%", GRAY)
    lbl(f"  Droite     {int(TURN_PROBS[2]*100)}%", GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_ql_agent(scenario="balanced"):
    path = f"outputs/agent_{scenario}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            agent = pickle.load(f)
        agent.epsilon = 0.0
        return agent
    return None


def make_env(traffic_idx):
    lam = LAMBDAS_BAL if traffic_idx == 0 else LAMBDAS_ASYM
    return IntersectionEnv(lambdas=lam, orange_duration=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Carrefour IA — Visualisation Temps Réel")
    clock  = pygame.time.Clock()

    font_sm  = pygame.font.SysFont("monospace", 13)
    font_md  = pygame.font.SysFont("monospace", 15, bold=True)
    font_lg  = pygame.font.SysFont("monospace", 20, bold=True)
    font_cnt = pygame.font.SysFont("monospace", 11, bold=True)

    MODE_MANUAL, MODE_BASELINE, MODE_QL, MODE_COMPARISON = 0, 1, 2, 3
    mode_names    = ["Manuel", "Baseline", "Q-Learning", "Comparaison"]
    traffic_names = ["Équilibré (λ=0.4)", "Asymétrique (λ_NS=0.7)"]

    mode          = MODE_BASELINE
    traffic_idx   = 0
    speed_idx     = 1
    paused        = False
    manual_action = None
    last_action   = 0
    
    # Comparison mode state
    comparison_running = False
    comparison_step = 0
    baseline_env = None
    ql_env = None
    baseline_agent = None
    ql_agent = None
    baseline_stats = {'total_reward': 0, 'total_waiting': 0, 'steps': 0}
    ql_stats = {'total_reward': 0, 'total_waiting': 0, 'steps': 0}
    show_comparison_results = False

    ql_agents = {
        "balanced":   load_ql_agent("balanced"),
        "asymmetric": load_ql_agent("asymmetric"),
    }

    env = baseline = car_mgr = None
    reward_total = step_count = 0
    phase_countdown = 8
    last_step_ms    = 0
    state           = None

    def reset_sim():
        nonlocal env, state, baseline, car_mgr
        nonlocal reward_total, step_count, phase_countdown, last_step_ms, last_action
        nonlocal comparison_running, comparison_step, show_comparison_results
        env             = make_env(traffic_idx)
        state           = env.reset()
        baseline        = BaselineAgent(period=5)
        car_mgr         = CarManager()
        reward_total    = 0
        step_count      = 0
        phase_countdown = 8
        last_action     = 0
        last_step_ms    = pygame.time.get_ticks()
        comparison_running = False
        comparison_step = 0
        show_comparison_results = False
    
    def start_comparison():
        nonlocal baseline_env, ql_env, baseline_agent, ql_agent
        nonlocal baseline_stats, ql_stats, comparison_running, comparison_step
        nonlocal show_comparison_results
        
        # Initialize both environments with same seed for fair comparison
        baseline_env = make_env(traffic_idx)
        ql_env = make_env(traffic_idx)
        
        # Set same random seed for reproducibility
        import numpy as np
        seed = int(pygame.time.get_ticks()) % 10000
        np.random.seed(seed)
        baseline_env.reset()
        np.random.seed(seed)
        ql_env.reset()
        
        # Initialize agents
        baseline_agent = BaselineAgent(period=5)
        key = "balanced" if traffic_idx == 0 else "asymmetric"
        ql_agent = ql_agents.get(key)
        if ql_agent is None:
            ql_agent = BaselineAgent(period=5)  # Fallback
        
        # Reset stats
        baseline_stats = {'total_reward': 0, 'total_waiting': 0, 'steps': 0}
        ql_stats = {'total_reward': 0, 'total_waiting': 0, 'steps': 0}
        comparison_running = True
        comparison_step = 0
        show_comparison_results = False

    reset_sim()

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if   k == pygame.K_ESCAPE: running = False
                elif k == pygame.K_SPACE:
                    if show_comparison_results:
                        # Close comparison results and return to normal mode
                        show_comparison_results = False
                        paused = False
                        mode = MODE_BASELINE
                        reset_sim()
                    else:
                        paused = not paused
                elif k == pygame.K_1: mode = MODE_MANUAL; show_comparison_results = False
                elif k == pygame.K_2: mode = MODE_BASELINE; baseline.reset(); show_comparison_results = False
                elif k == pygame.K_3: mode = MODE_QL; show_comparison_results = False
                elif k == pygame.K_c and not comparison_running:
                    mode = MODE_COMPARISON
                    start_comparison()
                elif k == pygame.K_4: traffic_idx = 0; reset_sim()
                elif k == pygame.K_5: traffic_idx = 1; reset_sim()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_idx = min(speed_idx + 1, len(STEP_MS) - 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_idx = max(speed_idx - 1, 0)
                elif k == pygame.K_r: reset_sim(); manual_action = None
                elif k == pygame.K_m and mode == MODE_MANUAL: manual_action = 0
                elif k == pygame.K_c and mode == MODE_MANUAL: manual_action = 1

        # ── Simulation step ───────────────────────────────────────────────────
        if not paused and (now - last_step_ms) >= STEP_MS[speed_idx]:
            last_step_ms = now

            if mode == MODE_COMPARISON and comparison_running:
                # Run both simulations in parallel
                if comparison_step < COMPARISON_STEPS:
                    # Baseline step
                    b_state = (baseline_env.queues[0], baseline_env.queues[1],
                              baseline_env.queues[2], baseline_env.queues[3],
                              baseline_env.phase)
                    b_action = baseline_agent.select_action(b_state)
                    _, b_reward, _ = baseline_env.step(b_action)
                    baseline_stats['total_reward'] += b_reward
                    baseline_stats['total_waiting'] += sum(baseline_env.queues)
                    baseline_stats['steps'] += 1
                    
                    # Q-Learning step
                    q_state = (ql_env.queues[0], ql_env.queues[1],
                              ql_env.queues[2], ql_env.queues[3],
                              ql_env.phase)
                    q_action = ql_agent.select_action(q_state)
                    _, q_reward, _ = ql_env.step(q_action)
                    ql_stats['total_reward'] += q_reward
                    ql_stats['total_waiting'] += sum(ql_env.queues)
                    ql_stats['steps'] += 1
                    
                    comparison_step += 1
                    
                    # Use Q-Learning env for visualization
                    state = q_state
                    env = ql_env
                    last_action = q_action
                    reward_total = ql_stats['total_reward']
                    step_count = comparison_step
                    car_mgr.sync(env.queues)
                    
                    # Update phase countdown
                    phase_countdown = min(comparison_step % 10, 99)
                else:
                    # Comparison finished
                    comparison_running = False
                    baseline_stats['avg_waiting'] = baseline_stats['total_waiting'] / baseline_stats['steps']
                    ql_stats['avg_waiting'] = ql_stats['total_waiting'] / ql_stats['steps']
                    show_comparison_results = True
                    paused = True
            
            elif mode != MODE_COMPARISON:
                if mode == MODE_MANUAL:
                    action = manual_action if manual_action is not None else 0
                    manual_action = None
                elif mode == MODE_BASELINE:
                    action = baseline.select_action(state)
                else:
                    key   = "balanced" if traffic_idx == 0 else "asymmetric"
                    agent = ql_agents.get(key)
                    action = agent.select_action(state) if agent else baseline.select_action(state)

                last_action    = action
                prev_phase     = env.phase
                state, reward, _ = env.step(action)
                reward_total  += reward
                step_count    += 1

                # Countdown: counts how many steps the current phase has been active
                if env.phase != prev_phase:
                    phase_countdown = 1
                else:
                    phase_countdown = min(phase_countdown + 1, 99)

                car_mgr.sync(env.queues)

        # ── Animation ─────────────────────────────────────────────────────────
        if not paused and not show_comparison_results:
            car_mgr.update(env.phase, env.queues)

        # ── Draw ──────────────────────────────────────────────────────────────
        # Only show countdown for Baseline mode (not Q-Learning or Comparison)
        show_countdown = phase_countdown if mode == MODE_BASELINE else None
        draw_scene(screen, env.phase, show_countdown, font_cnt, paused and not show_comparison_results)
        car_mgr.draw(screen)
        draw_panel(
            screen, env.phase, env.queues, step_count, reward_total,
            mode_names[mode], traffic_names[traffic_idx],
            speed_idx, show_countdown, last_action, manual_action,
            font_sm, font_md, font_lg
        )
        
        # Show comparison results overlay
        if show_comparison_results:
            draw_comparison_results(screen, baseline_stats, ql_stats,
                                   traffic_names[traffic_idx],
                                   font_sm, font_md, font_lg)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
