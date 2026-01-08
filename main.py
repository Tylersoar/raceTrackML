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

def scale_to_fit(img: pygame.Surface, max_w: int, max_h: int) -> pygame.Surface:
    iw, ih = img.get_size()
    scale = min(max_w / iw, max_h / ih)
    new_size = (int(iw * scale), int(ih * scale))
    return pygame.transform.smoothscale(img, new_size)

finish_line = pygame.image.load('res/finish.png').convert_alpha()
rotated_finish_line = pygame.transform.rotate(finish_line, 90)

# Load track image and scale to fit window
track = pygame.image.load('res/track.png')
track_scaled = scale_to_fit(track, WIDTH, HEIGHT)
track_rect = track_scaled.get_rect(center=(WIDTH // 2, HEIGHT // 2))
# Position finish line
finish_rect = rotated_finish_line.get_rect(topleft=(570, 20))

# Load track boarder image
track_border = pygame.image.load('res/track-border.png').convert_alpha()
track_border_scaled = scale_to_fit(track_border, WIDTH, HEIGHT)
track_border_rect = track_border_scaled.get_rect(center=(WIDTH // 2, HEIGHT // 2))
boarder_mask = pygame.mask.from_surface(track_border_scaled)

# Load car image and scale it down
car_img = pygame.image.load('res/green-car.png')
car_scaled = pygame.transform.scale_by(car_img, 0.5)


x, y = 550, 80
speed = 0
angle = 0

TURN_RATE = 4  # radians per frame at max speed
MAX_SPEED = 250
ACCEL = 150
FRICTION = 450

timer_running = False # are we timing the lap?
start_time = 0 # when the lap started in ms
current_time = 0 # current lap time in ms

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

    if car_rect.colliderect(finish_rect) and not timer_running:
        timer_running = True
        start_time = pygame.time.get_ticks()

    # if there is overlap between car and track boarder, slow car and reset position
    offset = (car_rect.left - track_border_rect.left, car_rect.top - track_border_rect.top)
    hit = boarder_mask.overlap(car_mask, offset)

    if hit:
        # hx, hy = hit
        # world_hit_x = hx + track_border_rect.left
        # world_hit_y = hy + track_border_rect.top
        # pygame.draw.circle(screen, (255, 0, 0), (world_hit_x, world_hit_y), 4)
        print("Collision detected!")
        x, y = 550, 80
        angle = 0
        speed = 0
        timer_running = False
        current_time = 0

    screen.blit(rotated_finish_line, (570, 20))  # Draw the finish line
    screen.blit(car_rot, car_rect)  # Draw the car
    screen.blit(track_border_scaled, track_border_rect)  # Draw the track boarder

    # display timer
    seconds = current_time / 1000
    timer_text = font.render(f"Time: {seconds:.2f}s", True, (255, 255, 255))
    screen.blit(timer_text, (20, 20))


    pygame.display.flip()

pygame.quit()
