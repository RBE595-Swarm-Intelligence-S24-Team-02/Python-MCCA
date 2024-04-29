import pygame
import numpy as np
from scipy.spatial import KDTree
import math
from scipy.optimize import linprog

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)

class Robot:
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

    def is_goal_reached(self):
        return np.linalg.norm(self.goal - self.current_location) < self.radius

    def compute_velocity(self, kd_tree, radius, time_horizon):
        self.neighbors = []
        neighbors_indices = kd_tree.query_ball_point(self.current_location, radius)
        for i in neighbors_indices:
            if i != self.ID:
                self.neighbors.append(robots[i])

        num_dims = len(self.current_location)
        num_neighbors = len(self.neighbors)

        c = np.zeros(num_dims)
        A = np.zeros((2 * num_neighbors, num_dims))
        b = np.zeros(2 * num_neighbors)

        for i, robot in enumerate(self.neighbors):
            relative_position = robot.current_location - self.current_location
            relative_velocity = np.array(robot.velocity) - self.velocity
            dist = np.linalg.norm(relative_position)
            time_to_collision = - np.dot(relative_position, relative_velocity) / (dist**2)
            time_to_collision = max(time_to_collision, 0)
            time_to_collision = min(time_to_collision, time_horizon)

            v_pref = robot.max_speed * relative_position / dist
            u = (v_pref - self.velocity) / np.linalg.norm(v_pref - self.velocity)
            perpendicular = np.array([-u[1], u[0]])

            radius_of_influence = 4 * (self.radius + robot.radius)

            A[i] = perpendicular
            b[i] = np.dot(perpendicular, self.current_location + self.velocity * time_to_collision +
                          0.5 * u * (time_to_collision ** 2)) + radius_of_influence - dist

            A[i + num_neighbors] = -perpendicular
            b[i + num_neighbors] = -np.dot(-perpendicular, self.current_location + self.velocity * time_to_collision +
                                            0.5 * u * (time_to_collision ** 2)) + radius_of_influence - dist

        valid_rows = np.logical_not(np.any(np.isnan(A) | np.isinf(A), axis=1))
        A = A[valid_rows]
        b = b[valid_rows]

        res = linprog(c, A_ub=A, b_ub=b)
        delta_velocity = np.clip(res.x, -self.max_speed, self.max_speed)

        if np.linalg.norm(delta_velocity) > 0 and time_to_collision > 0:
            self.velocity = self.velocity + 0.5 * delta_velocity

    def update_position(self):
        self.current_location += self.velocity * self.time_step
        self.path.append(self.current_location.copy())

    def move_towards_goal(self):
        direction = self.goal - self.current_location
        distance = np.linalg.norm(direction)
        if distance > self.radius:
            self.velocity = (direction / distance) * self.max_speed
        else:
            self.velocity = np.array([0, 0])

    def draw(self, screen):
        if len(self.path) > 1:
            pygame.draw.lines(screen, self.color, False, self.path, 2)
        pygame.draw.circle(screen, self.color, self.current_location.astype(int), self.radius)
        pygame.draw.line(screen, self.color, self.current_location.astype(int),
                         (self.current_location + self.velocity).astype(int), 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("RVO and ORCA")
    clock = pygame.time.Clock()

    num_robots = 6
    radius_of_influence = 100
    robots = []
    colors = [RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN]

    for i in range(num_robots):
        angle = 2 * math.pi * i / num_robots
        spawn_x = 400 + 300 * math.cos(angle)
        spawn_y = 400 + 300 * math.sin(angle)
        goal_x = 400 + 300 * math.cos(angle + math.pi)
        goal_y = 400 + 300 * math.sin(angle + math.pi)
        robots.append(Robot(10, 3, [spawn_x, spawn_y], [goal_x, goal_y], 0.1, i, colors[i]))

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        robot_locations = np.array([robot.current_location for robot in robots])
        kdtree = KDTree(robot_locations)

        for robot in robots:
            if not robot.is_goal_reached():
                robot.compute_velocity(kdtree, radius_of_influence, 2)
                robot.update_position()
            else:
                robot.velocity = np.array([0, 0])
            robot.move_towards_goal()

        screen.fill(BLACK)
        for robot in robots:
            robot.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
