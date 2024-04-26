import pygame 
import numpy as np 
from scipy.optimize import linprog 
import math

# SCREEN DIMENSIONS
WIDTH = 800
HEIGHT = 800

# Define colors
BLACK = (0, 0, 0)

class ORCARobot:
    def __init__(self, radius, max_speed, spawn_location, goal_location, time_step, id , color):
        self.current_location = np.array(spawn_location, dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.goal = np.array(goal_location, dtype=float)
        self.radius = radius
        self.max_speed = max_speed
        self.time_step = time_step
        self.neighbors = []
        self.ID = id
        self.color = color
        self.path = [self.current_location.copy()]  # Initialize path with current location
    
    def compute_new_velocity(self, robots, time_horizon, max_velocity_change):
        num_robots = len(robots)
        num_dims = len(self.current_location)
        
        # Construct the Linear Programming problem 
        c = np.zeros(num_dims)
        A = np.zeros((2*num_robots, num_dims))
        b = np.zeros(2*num_robots)
        
        for i, robot in enumerate(robots):
            relative_position = robot.current_location - self.current_location
            relative_velocity = np.array(robot.velocity) - self.velocity
            dist = np.linalg.norm(relative_position)
            time_to_collision = np.dot(relative_position, relative_velocity) / (dist**2 + 1e-6)
            if time_to_collision < 0:
                continue
            
            # Limit the time horizon
            time_to_collision = min(time_to_collision, time_horizon)
            
            #  Compute Velocity constraints
            v_pref = robot.velocity
            u = (v_pref - self.velocity) / np.linalg.norm(v_pref - self.velocity)
            perpendicular = np.array([-u[1], u[0]])
            
            # Adjust the radius of influence
            radius_of_influence = 2 * (self.radius + robot.radius)
            
            # Construct ORCA half-plane constraints
            A[i] = perpendicular
            b[i] = np.dot(perpendicular, self.current_location + self.velocity * time_to_collision + 0.5 * u * (time_to_collision**2)) + radius_of_influence - dist 
            
            A[i + num_robots] = -perpendicular
            b[i + num_robots] = -np.dot(-perpendicular, self.current_location + self.velocity * time_to_collision + 0.5 * u * (time_to_collision**2)) + radius_of_influence - dist
        
        # Remove rows with NaN values
        valid_rows = np.logical_not(np.any(np.isnan(A) | np.isinf(A), axis=1))
        A = A[valid_rows]
        b = b[valid_rows]
        
        # Construct the Linear Programming problem
        res = linprog(c, A_ub=A, b_ub=b)
        
        # Limit the maximum allowable change in velocity
        delta_velocity = np.clip(res.x, -max_velocity_change, max_velocity_change)
        
        # Update the velocity
        self.velocity = delta_velocity
        
        return self.velocity
    
    def update_position(self):
        # Update the position based on the current velocity
        self.current_location += self.velocity * self.time_step
        # Add current location to the path
        self.path.append(self.current_location.copy())

def draw_robot(robots, screen):
    for robot in robots:
        # Draw path
        if len(robot.path) > 1:
            pygame.draw.lines(screen, robot.color, False, robot.path, 2)
        # Draw robot
        pygame.draw.circle(screen, robot.color, robot.current_location.astype(int), robot.radius)
        # Draw heading direction
        pygame.draw.line(screen, robot.color, robot.current_location.astype(int), (robot.current_location + robot.velocity).astype(int), 2)
        
def main():
    pygame.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ORCA Algorithm")
    clock = pygame.time.Clock()
    
    # Define Simulation params 
    time_horizon = 1.0 
    time_step = 0.1
    max_velocity_change = 0.1  # Maximum allowable change in velocity
    
    num_robots = 6
    
    # Define Robots 
    robots = []
    
    # Define circle parameters
    circle_center = np.array([WIDTH // 2, HEIGHT // 2])
    circle_radius = min(WIDTH, HEIGHT) // 3
    
    # Define colors for each robot
    robot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Spawn robots evenly spaced on the circumference of a circle
    for i in range(num_robots):
        angle = 2 * math.pi * i / num_robots
        spawn_x = circle_center[0] + circle_radius * math.cos(angle)
        spawn_y = circle_center[1] + circle_radius * math.sin(angle)
        
        # Define opposite direction goal positions
        goal_x = circle_center[0] + circle_radius * math.cos(angle + math.pi)
        goal_y = circle_center[1] + circle_radius * math.sin(angle + math.pi)
        
        print(f"Robot {i} - Spawn: ({spawn_x}, {spawn_y}) - Goal: ({goal_x}, {goal_y})")
        robots.append(ORCARobot(10, 1, [spawn_x, spawn_y], [goal_x, goal_y], time_step, i, robot_colors[i]))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update robot velocities
        for robot in robots:
            robot.compute_new_velocity(robots, time_horizon, max_velocity_change)
        
        # Update robot positions towards the goal
        for robot in robots:
            # Calculate direction towards the goal
            direction = robot.goal - robot.current_location
            distance = np.linalg.norm(direction)
            if distance > 1:  # Check if the robot has reached its goal
                # Move towards the goal at maximum speed
                robot.current_location += direction / distance * robot.max_speed * time_step
                # Add current location to the path
                robot.path.append(robot.current_location.copy())
        
        screen.fill(BLACK)
        draw_robot(robots, screen)
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    main()



