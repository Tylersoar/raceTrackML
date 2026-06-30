import math
import random
import pygame
from pygame.locals import *
import numpy as np

# IMPORT THE SHARED BRAIN
from QLearningAgent import QLearningAgent

pygame.init()

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Race Track ML - Swarm Training (10 Agents)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- HYPERPARAMETERS ---
N_AGENTS = 100  # Run 100 cars at once
SWARM_DECAY = 0.995  # Decays per checkpoint hit: ~300 CPs (100 laps) → ε≈0.22


# --- ASSETS ---
def scale_to_fit(img: pygame.Surface, max_w: int, max_h: int) -> pygame.Surface:
    iw, ih = img.get_size()
    scale = min(max_w / iw, max_h / ih)
    new_size = (int(iw * scale), int(ih * scale))
    return pygame.transform.smoothscale(img, new_size)


try:
    finish_line = pygame.image.load('res/finish.png').convert_alpha()
    rotated_finish_line = pygame.transform.rotate(finish_line, 90)
    track = pygame.image.load('res/track.png')
    track_scaled = scale_to_fit(track, WIDTH, HEIGHT)
    track_border = pygame.image.load('res/track-border.png').convert_alpha()
    track_border_scaled = scale_to_fit(track_border, WIDTH, HEIGHT)

    car_img = pygame.image.load('res/green-car.png')
    car_scaled = pygame.transform.scale_by(car_img, 0.3)

except FileNotFoundError:
    print("ERROR: Assets not found! Ensure 'res/' folder exists.")
    exit()

# Geometry
track_rect = track_scaled.get_rect(center=(WIDTH // 2, HEIGHT // 2))
track_border_rect = track_border_scaled.get_rect(center=(WIDTH // 2, HEIGHT // 2))
border_mask = pygame.mask.from_surface(track_border_scaled)
finish_rect = rotated_finish_line.get_rect(topleft=(570, 20))

# Each checkpoint: collision_rect + 2 parallel line segments for drawing
checkpoints = [
    {  # CP 0 — right side horizontal section
        "rect": pygame.Rect(790, 125, 85, 13),
        "lines": [((796, 125), (865, 125)), ((796, 135), (865, 135))]
    },

    {  # CP 1 — right side vertical section
        "rect": pygame.Rect(585, 220, 15, 80),
        "lines": [((585, 223), (585, 292)), ((595, 223), (595, 292))]
    },
    {  # CP 2 — right side vertical section
        "rect": pygame.Rect(585, 320, 15, 80),
        "lines": [((585, 325), (585, 395)), ((595, 325), (595, 395))]
    },
    {  # CP 3 — bottom horizontal section
        "rect": pygame.Rect(460, 575, 80, 15),
        "lines": [((468, 575), (535, 575)), ((468, 585), (535, 585))]
    },
    {  # CP 4 — bottom horizontal section
        "rect": pygame.Rect(655, 575, 80, 15),
        "lines": [((658, 575), (728, 575)), ((658, 585), (728, 585))]
    },
    {  # CP 5 — left side horizontal section
        "rect": pygame.Rect(120, 225, 80, 10),
        "lines": [((125, 225), (195, 225)), ((125, 235), (195, 235))]
    },
    {  # CP 6 — left side horizontal section
        "rect": pygame.Rect(240, 225, 80, 10),
        "lines": [((240, 225), (310, 225)), ((240, 235), (310, 235))]
    },
]

START_X, START_Y = 550, 80
START_ANGLE = 0.0

ACTIONS = {
    0: (0.0, 0.0), 1: (0.0, 1.0), 2: (0.0, -1.0),
    3: (-1.0, 1.0), 4: (1.0, 1.0), 5: (-1.0, 0.0), 6: (1.0, 0.0)
}
N_ACTIONS = len(ACTIONS)

# Physics
TURN_RATE = 4
MAX_SPEED = 250
ACCEL = 250
FRICTION = 120
RAY_OFFSETS = [-1.2, -1.0, -0.8, 0.0, 0.8, 1.0, 1.2]


# --- HELPER FUNCTIONS ---

def create_new_state(index=0):
    """Generates a fresh car state with random jitter to increase swarm diversity."""
    return {
        "id": index,
        "x": START_X + random.uniform(-15, 15),
        "y": START_Y + random.uniform(-10, 10),
        "speed": 0.0,
        "angle": START_ANGLE + random.uniform(-0.3, 0.3),
        "timer_running": False,
        "start_time": 0,
        "current_time": 0,
        "next_cp": 0,
        "was_on_finish": False,
        "info": None  # Will store visual data
    }


def get_observation(x, y, angle, speed, border_mask, border_rect, offsets):
    sx = x + math.cos(angle) * 15
    sy = y + math.sin(angle) * 15
    w, h = border_mask.get_size()
    ray_distances = []
    ray_hits = []

    for off in offsets:
        ray_angle = angle + off
        dx, dy = math.cos(ray_angle), math.sin(ray_angle)
        hit_dist = 150

        for d in range(0, 151, 4):
            px = int(sx + dx * d - border_rect.left)
            py = int(sy + dy * d - border_rect.top)
            if px < 0 or py < 0 or px >= w or py >= h or border_mask.get_at((px, py)):
                hit_dist = d
                break

        ray_distances.append(hit_dist)
        ray_hits.append((int(sx + dx * hit_dist), int(sy + dy * hit_dist)))

    rays_norm = [d / 150 for d in ray_distances]
    speed_norm = max(-1.0, min(1.0, speed / MAX_SPEED))
    obs = rays_norm + [speed_norm]
    return obs, ray_distances, ray_hits, (sx, sy)


def discretize_state(obs):
    buckets = [0.15, 0.35, 0.6]
    s_far_left = np.digitize(obs[0], buckets)
    s_left = np.digitize(obs[1], buckets)
    s_fwd_left = np.digitize(obs[2], buckets)
    s_center = np.digitize(obs[3], buckets)
    s_fwd_right = np.digitize(obs[4], buckets)
    s_right = np.digitize(obs[5], buckets)
    s_far_right = np.digitize(obs[6], buckets)
    s_speed = np.digitize(obs[7], [-0.5, 0.1, 0.5, 0.8])
    return s_far_left, s_left, s_fwd_left, s_center, s_fwd_right, s_right, s_far_right, s_speed


def compute_reward(speed_norm, rays, collided, cp_advanced, finished_lap):
    reward = -0.1  # Living penalty
    if collided: return -100
    if finished_lap: return 100
    if cp_advanced: return 20

    reward += speed_norm * 0.2
    min_ray = min(rays) / 150.0
    if min_ray < 0.2: reward -= 0.5
    return reward


def step(state, action, dt):
    # Physics
    steer, throttle = ACTIONS[action]
    if throttle != 0.0:
        state["speed"] += throttle * ACCEL * dt
    else:
        if state["speed"] > 0:
            state["speed"] = max(0, state["speed"] - FRICTION * dt)
        elif state["speed"] < 0:
            state["speed"] = min(0, state["speed"] + FRICTION * dt)
    state["speed"] = max(-MAX_SPEED, min(MAX_SPEED, state["speed"]))

    speed_ratio = abs(state["speed"]) / MAX_SPEED
    steer_strength = max(0.35, min(1.0, speed_ratio))
    state["angle"] += steer * TURN_RATE * steer_strength * dt
    state["x"] += math.cos(state["angle"]) * state["speed"] * dt
    state["y"] += math.sin(state["angle"]) * state["speed"] * dt

    if state["timer_running"]:
        state["current_time"] = pygame.time.get_ticks() - state["start_time"]

    # Collision & Logic
    car_rot = pygame.transform.rotate(car_scaled, -math.degrees(state["angle"]) - 90)
    car_rect = car_rot.get_rect(center=(state["x"], state["y"]))
    car_mask = pygame.mask.from_surface(car_rot)

    # Wall Collision
    offset = (int(car_rect.left - track_border_rect.left), int(car_rect.top - track_border_rect.top))
    collided = border_mask.overlap(car_mask, offset) is not None

    # Checkpoints
    prev_cp = state["next_cp"]
    if state["next_cp"] < len(checkpoints) and car_rect.colliderect(checkpoints[state["next_cp"]]["rect"]):
        state["next_cp"] += 1
    cp_advanced = state["next_cp"] > prev_cp

    # Finish Line
    on_finish = finish_rect.colliderect(car_rect)
    finished_lap = False
    if on_finish and not state["was_on_finish"]:
        if not state["timer_running"]:
            state["timer_running"] = True
            state["start_time"] = pygame.time.get_ticks()
        elif state["next_cp"] == len(checkpoints):
            finished_lap = True
            state["start_time"] = pygame.time.get_ticks()
            state["next_cp"] = 0
    state["was_on_finish"] = on_finish

    # Observe (for returning info)
    obs, rays, hits, ray_start = get_observation(state["x"], state["y"], state["angle"], state["speed"], border_mask,
                                                 track_border_rect, RAY_OFFSETS)
    reward = compute_reward(obs[7], rays, collided, cp_advanced, finished_lap)

    info = {
        "car_rot": car_rot,
        "car_rect": car_rect,
        "hits": hits,
        "ray_start": ray_start,
        "reward": reward,
        "finished_lap": finished_lap,
        "cp_advanced": cp_advanced  # Expose for epsilon decay
    }

    return obs, reward, collided, info


# --- INITIALIZATION ---

# 1. Initialize Shared Brain
agent = QLearningAgent(n_actions=N_ACTIONS, epsilon_decay=SWARM_DECAY)

# 2. Initialize Swarm
cars = [create_new_state(i) for i in range(N_AGENTS)]

running = True
episode_total = 0
leader_crossed_cps = set()  # Track which checkpoints the leader has crossed (visual highlight)

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # --- UPDATE SWARM ---
    for i, car in enumerate(cars):
        # A. Observe
        obs, rays, hits, ray_start = get_observation(car["x"], car["y"], car["angle"], car["speed"], border_mask,
                                                     track_border_rect, RAY_OFFSETS)
        curr_key = discretize_state(obs)

        # B. Decide (Shared Brain)
        action = agent.choose_action(curr_key)

        # C. Act (Physics)
        next_obs, reward, done, info = step(car, action, dt)
        next_key = discretize_state(next_obs)

        # D. Learn (Shared Brain) — terminal state if crashed
        if done:
            agent.learn(curr_key, action, reward, None)  # No future reward on crash
        else:
            agent.learn(curr_key, action, reward, next_key)

        # E. Save Visuals
        car["info"] = info
        car["action"] = action  # Save for UI display

        # F. Reset if Done
        if done or info["finished_lap"]:
            if info["finished_lap"]:
                print(f"!!! Car {i} FINISHED LAP !!!")

            # Reset ONLY this car
            cars[i] = create_new_state(i)
            episode_total += 1

        # Decay epsilon on checkpoint progress (not crashes)
        if info.get("cp_advanced"):
            agent.decay()

    # Update leader checkpoint crossings (visual highlight)
    leader_rect = cars[0]["info"]["car_rect"] if cars[0]["info"] else None
    if leader_rect is not None:
        for cp_idx, cp in enumerate(checkpoints):
            if cp_idx not in leader_crossed_cps and leader_rect.colliderect(cp["rect"]):
                leader_crossed_cps.add(cp_idx)

    # --- DRAWING ---
    screen.fill((30, 30, 30))
    screen.blit(track_scaled, track_rect)
    screen.blit(rotated_finish_line, finish_rect)

    # Draw Checkpoints as thin line pairs
    for cp_idx, cp in enumerate(checkpoints):
        color = (255, 162, 19) if cp_idx in leader_crossed_cps else (145, 211, 242)
        for line_start, line_end in cp["lines"]:
            pygame.draw.line(screen, color, line_start, line_end, 3)

    # Draw Cars
    for i, car in enumerate(cars):
        if car["info"] is None: continue

        info = car["info"]

        # Draw Leader (Car 0) with Rays and Red Color
        if i == 0:
            # Tint the rotated car red
            leader_car = info["car_rot"].copy()
            leader_car.fill((255, 50, 50, 255), special_flags=pygame.BLEND_RGBA_MULT)
            sx, sy = info["ray_start"]
            for hx, hy in info["hits"]:
                pygame.draw.line(screen, (0, 200, 255), (sx, sy), (hx, hy), 1)
                pygame.draw.circle(screen, (255, 50, 50), (hx, hy), 2)
            screen.blit(leader_car, info["car_rect"])
        else:
            # Draw Swarm (Green) without rays
            screen.blit(info["car_rot"], info["car_rect"])

    # UI (Stats from Leader)
    leader_info = cars[0]["info"]
    leader_reward = leader_info["reward"] if leader_info else 0
    leader_action = cars[0].get("action", 0)

    ui_text = [
        f"Agents: {N_AGENTS}",
        f"Total Episodes: {episode_total}",
        f"Epsilon: {agent.epsilon:.4f}",
        f"Leader Reward: {leader_reward:.1f}",
        f"Leader Action: {leader_action}"
    ]
    for i, line in enumerate(ui_text):
        screen.blit(font.render(line, True, (255, 255, 255)), (10, 10 + i * 20))

    pygame.display.flip()

pygame.quit()