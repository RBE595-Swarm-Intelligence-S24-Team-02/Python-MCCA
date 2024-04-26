import pygame
import numpy as np
from scipy.optimize import linprog

# Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define screen parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class ORCARobot:
    def __init__(self, position, velocity, radius):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.radius = radius

    def compute_new_velocity(self, robots, time_horizon, time_step):
        num_robots = len(robots)
        num_dims = len(self.position)
        
        # Construct linear programming problem
        c = np.zeros(num_dims)
        A = np.zeros((2*num_robots, num_dims))
        b = np.zeros(2*num_robots)
        
        for i, robot in enumerate(robots):
            relative_position = robot.position - self.position
            dist = np.linalg.norm(relative_position)
            time_to_collision = np.dot(relative_position, robot.velocity - self.velocity) / (dist ** 2 + 1e-5)
            if time_to_collision < 0:
                continue
            
            # Compute velocity constraints
            v_pref = robot.velocity
            u = (v_pref - self.velocity) / np.linalg.norm(v_pref - self.velocity)
            perpendicular = np.array([-u[1], u[0]])
            
            # ORCA half-planes
            A[i] = perpendicular
            b[i] = np.dot(perpendicular, self.position + self.velocity * time_to_collision + 0.5 * u * (time_to_collision ** 2)) + self.radius + robot.radius - dist
            
            A[num_robots + i] = -perpendicular
            b[num_robots + i] = -np.dot(-perpendicular, self.position + self.velocity * time_to_collision + 0.5 * u * (time_to_collision ** 2)) + self.radius + robot.radius - dist
        
        # Remove rows with nan or inf values
        valid_rows = np.logical_not(np.any(np.isnan(A) | np.isinf(A), axis=1))
        A= A[valid_rows]
        b = b[valid_rows]
        
        # Construct linear programming problem
        res = linprog(c, A_ub=A, b_ub=b)
        new_velocity = self.velocity + time_step * res.x
        
        return new_velocity

def draw_robots(robots, screen):
    for robot in robots:
        pygame.draw.circle(screen, BLACK, (int(robot.position[0]), int(robot.position[1])), int(robot.radius), 0)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("ORCA Robot Simulation")
    clock = pygame.time.Clock()
    
    # Define simulation parameters
    time_horizon = 1.0
    time_step = 0.1

    # Define robots
    robots = [
        ORCARobot(position=[100, 100], velocity=[1, 0], radius=20),
        ORCARobot(position=[700, 500], velocity=[-1, 0], radius=20)
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(WHITE)

        # Update robots
        for robot in robots:
            new_velocity = robot.compute_new_velocity(robots, time_horizon, time_step)
            robot.velocity = new_velocity
            robot.position = (robot.position + robot.velocity * time_step).astype(int)

        # Draw robots
        draw_robots(robots, screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
