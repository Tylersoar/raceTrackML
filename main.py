import math

import pygame
from pygame.locals import *

pygame.init()

# Sets window size
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Race Track ML - stage 0")

# Sets frame rate
clock = pygame.time.Clock()
# for displaying the timer
font = pygame.font.SysFont(None, 24)



def fmt(ms):
    if ms is None:
        return "--"
    return f"{ms / 1000:.2f}s"


def scale_to_fit(img: pygame.Surface, max_w: int, max_h: int) -> pygame.Surface:
    iw, ih = img.get_size()
    scale = min(max_w / iw, max_h / ih)
    new_size = (int(iw * scale), int(ih * scale))
    return pygame.transform.smoothscale(img, new_size)


# Load and rotate finish line image
finish_line = pygame.image.load('res/finish.png').convert_alpha()
rotated_finish_line = pygame.transform.rotate(finish_line, 90)

# Define checkpoints for lap timing (not implemented yet)
checkpoints = [
    pygame.Rect(600, 220, 40, 100),  # checkpoint 1
    pygame.Rect(450, 570, 100, 40),  # checkpoint 2
    pygame.Rect(230, 220, 100, 40)  # checkpoint 3
]

# Load track image and scale to fit window
track = pygame.image.load('res/track.png')
track_scaled = scale_to_fit(track, WIDTH, HEIGHT)
track_rect = track_scaled.get_rect(center=(WIDTH // 2, HEIGHT // 2))

# Position finish line
finish_rect = rotated_finish_line.get_rect(topleft=(570, 20))

# Load track border image
track_border = pygame.image.load('res/track-border.png').convert_alpha()
track_border_scaled = scale_to_fit(track_border, WIDTH, HEIGHT)
track_border_rect = track_border_scaled.get_rect(center=(WIDTH // 2, HEIGHT // 2))
border_mask = pygame.mask.from_surface(track_border_scaled)

# Load car image and scale it down
car_img = pygame.image.load('res/green-car.png')
car_scaled = pygame.transform.scale_by(car_img, 0.3)

# Car state
x, y = 550, 80
speed = 0
angle = 0

# Car constants
TURN_RATE = 4 #
MAX_SPEED = 250
ACCEL = 150
FRICTION = 450

# Timer variables
timer_running = False  # are we timing the lap?
start_time = 0  # when the lap started in ms
current_time = 0  # current lap time in ms

# Lap timing variables
prev_cp = 0 # index of previous checkpoint hit
next_cp = 0  # index of next checkpoint to hit
lap_armed = False  # becomes true after crossing start line
best_lap = None  # best lap time in ms
last_lap = None  # last lap time in ms
was_on_finish = False  # was the car on the finish line last frame

ray_offsets_rad = [-1.2, -1.0, -0.8, 0.0, 0.8, 1.0, 1.2] # ray angles relative to car angle

def compute_reward(
        speed_norm: float,
        ray_distances: list[int],
        max_distance: int,
        cp_advanced: bool,
        finished_lap: bool,
        collided: bool
)->float:
    SPEED_REWARD_SCALE = 0.05
    WALL_PENALTY_SCALE = 0.03
    CP_REWARD = 1.0
    LAP_REWARD = 10.0
    CRASH_PENALTY = 10.0
    PROGRESS_REWARD = 0.1
    reward = 0.0

    reward += max(0.0, speed_norm) * SPEED_REWARD_SCALE # reward for moving forward

    # wall proximity penality (encourage staying centered)
    min_ray = min(ray_distances) # minimum ray distance - closest to the wall
    closeness = 1.0 - (min_ray/max_distance)
    reward -= closeness * WALL_PENALTY_SCALE

    progress = next_cp / len(checkpoints)
    reward += progress * PROGRESS_REWARD

    # crossing checkpoints and finishing lap and collisions
    if cp_advanced:
        reward += CP_REWARD
    if finished_lap:
        reward += LAP_REWARD
    if collided:
        reward -= CRASH_PENALTY

    return reward




def get_observation(
    x, y, angle, speed,
    border_mask, border_rect,
    offsets,
    front_offset=15,
    max_distance=150,
    step=4,
    max_speed=250
):
    # start from the nose of the car (screen coords)
    sx = x + math.cos(angle) * front_offset
    sy = y + math.sin(angle) * front_offset

    w, h = border_mask.get_size()

    ray_distances = []
    ray_hits = []

    for off in offsets:
        ray_angle = angle + off
        dx = math.cos(ray_angle) # ray direction vector
        dy = math.sin(ray_angle) # ray direction vector

        hit_dist = max_distance

        for d in range(0, max_distance + 1, step):
            px = sx + dx * d
            py = sy + dy * d

            mx = int(px - border_rect.left)
            my = int(py - border_rect.top)

            # out of bounds counts as hit
            if mx < 0 or my < 0 or mx >= w or my >= h:
                hit_dist = d
                break

            if border_mask.get_at((mx, my)):
                hit_dist = d
                break

        hit_x = int(sx + dx * hit_dist)
        hit_y = int(sy + dy * hit_dist)

        ray_distances.append(hit_dist) # distance to hit in pixels
        ray_hits.append((hit_x, hit_y)) # hit position in screen coords

    # normalize rays to [0,1] from pixels
    rays_norm = [d / max_distance for d in ray_distances]

    # normalize speed to [-1,1]
    speed_norm = max(-1.0, min(1.0, speed / max_speed))

    # observation vector (ML-friendly)
    obs = rays_norm + [speed_norm, math.sin(angle), math.cos(angle)] # normalised values rays + speed + angle (radians)

    return obs, ray_distances, ray_hits, (sx, sy) # return sensor start position


running = True
while running:
    dt = clock.tick(60) / 1000  # Delta time in seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if timer_running:
        current_time = pygame.time.get_ticks() - start_time

    keys = pygame.key.get_pressed()
    # move car based on key presses

    throttle = 0.0
    if keys[K_UP] or keys[K_w]:
        throttle += 1.0
    if keys[K_DOWN] or keys[K_s]:
        throttle -= 1.0

    if throttle != 0.0:
        speed += throttle * ACCEL * dt

    else:
        if speed > 0:
            speed = max(0, speed - FRICTION * dt)
        elif speed < 0:
            speed = min(0, speed + FRICTION * dt)

    speed = max(-MAX_SPEED, min(MAX_SPEED, speed))

    steer = 0.0
    if keys[K_LEFT] or keys[K_a]:
        steer -= 1.0
    if keys[K_RIGHT] or keys[K_d]:
        steer += 1.0

    steer_strength = min(1.0, abs(speed) / MAX_SPEED)
    angle += steer * TURN_RATE * steer_strength * dt

    dx = math.cos(angle)
    dy = math.sin(angle)

    x += dx * speed * dt
    y += dy * speed * dt

    screen.fill((25, 25, 25))  # Fill the screen with black
    screen.blit(track_scaled, track_rect)  # Draw the track

    car_rot = pygame.transform.rotate(car_scaled, -math.degrees(angle) - 90)
    car_rect = car_rot.get_rect(center=(x, y))
    car_mask = pygame.mask.from_surface(car_rot)

    # checkpoint logic
    finished_lap = False
    cp_advanced = False
    prev_cp = next_cp

    # checkpoints in order
    if timer_running and next_cp < len(checkpoints) and car_rect.colliderect(checkpoints[next_cp]):
        next_cp += 1

    cp_advanced = (next_cp > prev_cp)

    # check for finish line crossing
    on_finish_line = finish_rect.colliderect(car_rect)
    crossed_finish = (not was_on_finish) and on_finish_line # checks if the car just crossed the finish line

    if crossed_finish:
        if not timer_running:
            # START a new lap
            timer_running = True
            start_time = pygame.time.get_ticks()
            current_time = 0
            next_cp = 0

        else:
            # FINISH lap only if all checkpoints hit
            if next_cp == len(checkpoints):
                finished_lap = True # for reward logic

                last_lap = current_time
                if best_lap is None or last_lap < best_lap:
                    best_lap = last_lap

                # IMMEDIATELY start next lap from 0 (normal racing)
                start_time = pygame.time.get_ticks()
                current_time = 0
                next_cp = 0
            else:
                # crossed finish too early
                pass

    was_on_finish = on_finish_line

    # if there is overlap between car and track border, reset position
    mask_offset = (car_rect.left - track_border_rect.left, car_rect.top - track_border_rect.top) # 
    hit = border_mask.overlap(car_mask, mask_offset)
    collided = (hit is not None)

    if hit:
        print("Collision detected!")
        x, y = 550, 80
        angle = 0
        speed = 0
        timer_running = False
        current_time = 0
        was_on_finish = False
        next_cp = 0


    for i, cp in enumerate(checkpoints):
        if i < next_cp:
            color = (0, 255, 0)  # green for passed checkpoints
        else:
            color = (255, 255, 0)  # yellow for upcoming checkpoints

        pygame.draw.rect(screen, color, cp, 2)

    screen.blit(rotated_finish_line, (570, 20))  # Draw the finish line
    screen.blit(car_rot, car_rect)  # Draw the car
    screen.blit(track_border_scaled, track_border_rect)  # Draw the track border

    # draw sensor rays
    obs, ray_distances, ray_hits, ray_start = get_observation(
        x, y, angle, speed,
        border_mask, track_border_rect,
        ray_offsets_rad,
        front_offset=15,
        max_distance=150,
        step=4,
        max_speed=MAX_SPEED
    )
    speed_norm = obs[7]  # 7 rays -> index 7 is speed_norm
    reward = compute_reward(
        speed_norm=speed_norm,
        ray_distances=ray_distances,
        max_distance=150,  # must match get_observation(max_distance=150)
        cp_advanced=cp_advanced,
        finished_lap=finished_lap,
        collided=collided
    )

    sx, sy = ray_start
    for hx, hy in ray_hits:
        pygame.draw.line(screen, (0, 200, 255), (sx, sy), (hx, hy), 2)
        pygame.draw.circle(screen, (255, 80, 80), (hx, hy), 3)


    # display timer
    seconds = current_time / 1000

    timer_text = font.render(f"Time: {seconds:.2f}s", True, (255, 255, 255))
    screen.blit(timer_text, (20, 20))

    cp_text = font.render(f"CP: {next_cp}/{len(checkpoints)}", True, (255, 255, 255))
    screen.blit(cp_text, (20, 45))

    last_lap_text = font.render(f"LL: {fmt(last_lap)}", True, (255, 255, 255))
    screen.blit(last_lap_text, (20, 60))

    best_lap_text = font.render(f"BL: {fmt(best_lap)}", True, (255, 255, 255))
    screen.blit(best_lap_text, (20, 75))

    rays_text = font.render("Rays: " + " ".join(f"{d:3d}" for d in ray_distances), True, (255, 255, 255))
    screen.blit(rays_text, (20, 95)) # display sensor distances

    obs_text = font.render("Obs: " + " ".join(f"{v:.2f}" for v in obs[:7]), True, (255, 255, 255))
    screen.blit(obs_text, (20, 120))

    reward_text = font.render(f"Reward: {reward:.3f}", True, (255, 255, 255))
    screen.blit(reward_text, (20, 145))

    pygame.display.flip()

pygame.quit()
