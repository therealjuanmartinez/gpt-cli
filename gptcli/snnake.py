import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH = 800
HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
SNAKE_SPEED = 15

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Create the game window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Initialize clock for controlling game speed
clock = pygame.time.Clock()

# Snake class
class Snake:
	def __init__(self):
		self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
		self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
		self.color = GREEN

	def get_head_position(self):
		return self.positions[0]

	def move(self):
		cur = self.get_head_position()
		x, y = self.direction
		new = ((cur[0] + x) % GRID_WIDTH, (cur[1] + y) % GRID_HEIGHT)
		if new in self.positions[3:]:
			return False
		self.positions.insert(0, new)
		if len(self.positions) > len(self.positions) - 1:
			self.positions.pop()
		return True

	def reset(self):
		self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
		self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

	def draw(self, surface):
		for p in self.positions:
			pygame.draw.rect(surface, self.color, (p[0] * GRID_SIZE, p[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

	def handle_keys(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP and self.direction != (0, 1):
					self.direction = (0, -1)
				elif event.key == pygame.K_DOWN and self.direction != (0, -1):
					self.direction = (0, 1)
				elif event.key == pygame.K_LEFT and self.direction != (1, 0):
					self.direction = (-1, 0)
				elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
					self.direction = (1, 0)

# Food class
class Food:
	def __init__(self):
		self.position = (0, 0)
		self.color = RED
		self.randomize_position()

	def randomize_position(self):
		self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

	def draw(self, surface):
		pygame.draw.rect(surface, self.color, (self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Main game function
def main():
	snake = Snake()
	food = Food()
	score = 0

	while True:
		snake.handle_keys()
		draw_grid(window)
		
		if not snake.move():
			score = 0
			snake.reset()

		if snake.get_head_position() == food.position:
			snake.positions.append(snake.positions[-1])
			food.randomize_position()
			score += 1

		snake.draw(window)
		food.draw(window)
		draw_score(window, score)
		pygame.display.update()
		clock.tick(SNAKE_SPEED)

# Draw grid
def draw_grid(surface):
	surface.fill(BLACK)
	for y in range(0, HEIGHT, GRID_SIZE):
		for x in range(0, WIDTH, GRID_SIZE):
			rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
			pygame.draw.rect(surface, WHITE, rect, 1)

# Draw score
def draw_score(surface, score):
	font = pygame.font.Font(None, 36)
	text = font.render(f"Score: {score}", True, WHITE)
	surface.blit(text, (10, 10))

if __name__ == "__main__":
	main()
