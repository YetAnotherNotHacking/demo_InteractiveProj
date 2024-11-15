import pygame
import math
import sqlite3
import random

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

ball_radius = 15
ball_pos = [10, 10]
ball_vel = [0, 0]
cursor_radius = 1
friction = 1.03
max_speed = 120
push_force = 10

def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS data (x INTEGER, y INTEGER)')
    conn.commit()
    conn.close()


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_db_exists():
    conn = sqlite3.connect('data.db')
    data = None
    c = conn.cursor()
    c.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="data"')
    data = c.fetchall()
    conn.close()
    return len(data) > 0

def open_or_create_db():
    if not check_db_exists():
        init_db()

running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    mouse_pos = pygame.mouse.get_pos()
    dist = distance(mouse_pos, ball_pos)

    if dist < ball_radius + cursor_radius:
        dx, dy = mouse_pos[0] - ball_pos[0], mouse_pos[1] - ball_pos[1]
        angle = math.atan2(dy, dx)
        force_magnitude = push_force
        ball_vel[0] -= force_magnitude * math.cos(angle)
        ball_vel[1] -= force_magnitude * math.sin(angle)

    ball_vel[0] = max(min(ball_vel[0], max_speed), -max_speed)
    ball_vel[1] = max(min(ball_vel[1], max_speed), -max_speed)

    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]

    ball_vel[0] *= friction
    ball_vel[1] *= friction

    # Save positin
    open_or_create_db()
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('INSERT INTO data VALUES (?, ?)', (ball_pos[0], ball_pos[1]))
    conn.commit()
    conn.close()

    # Read db and draw points
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM data')
    data = c.fetchall()
    conn.close()
    # draw line
    for i in range(1, len(data)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pygame.draw.line(screen, color, data[i-1], data[i], 2)

    if ball_pos[0] - ball_radius < 0 or ball_pos[0] + ball_radius > 800:
        ball_vel[0] = -ball_vel[0]
        ball_pos[0] = max(ball_radius, min(800 - ball_radius, ball_pos[0]))

    if ball_pos[1] - ball_radius < 0 or ball_pos[1] + ball_radius > 600:
        ball_vel[1] = -ball_vel[1]
        ball_pos[1] = max(ball_radius, min(600 - ball_radius, ball_pos[1]))

    pygame.draw.circle(screen, RED, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)
    pygame.draw.circle(screen, BLACK, mouse_pos, cursor_radius, 1)

    pygame.display.flip()
    clock.tick(120)

pygame.quit()
