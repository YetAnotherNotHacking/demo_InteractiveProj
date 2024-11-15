import pygame

# Initialize Pygame
pygame.init()

# Set up the game window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Ball Game")

# Ball class
class Ball:
    def __init__(self, x, y, radius, speed_x, speed_y, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.color = color

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off the walls
        if self.x - self.radius < 0 or self.x + self.radius > WINDOW_WIDTH:
            self.speed_x *= -1
        if self.y - self.radius < 0 or self.y + self.radius > WINDOW_HEIGHT:
            self.speed_y *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# Game loop
running = True
clock = pygame.time.Clock()

# Create the ball
ball = Ball(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, 20, 5, 5, (255, 255, 255))

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the ball
    ball.update()

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the ball
    ball.draw(screen)

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()