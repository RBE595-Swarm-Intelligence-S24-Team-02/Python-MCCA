"""
ORCA Implementation 
Random

NON  Smooth
ORCA + VO Constraint Implemented

"""

import pygame
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import KDTree
import math

# Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60

IS_VO = True

class Robot:
    def __init__(self, radius, max_speed, spawn_location, goal_location, time_step, id, color):
        self.current_location = np.array(spawn_location, dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.goal = np.array(goal_location, dtype=float)
        self.radius = radius
        self.max_speed = max_speed
        self.time_step = time_step
        self.neighbours = []
        self.AV = []  # Added attribute for velocities
        self.ID = id
        self.color = color
        self.trail = [self.current_location.copy()]  # Trail to store previous locations

    def is_goal_reached(self):
        return np.linalg.norm(self.goal - self.current_location) < self.radius

    def update_current_location(self):
        self.trail.append(self.current_location.copy())  # Store the current location in the trail
        self.current_location += self.velocity * self.time_step

    def preferred_velocity(self):
        if not self.is_goal_reached():
            direction = self.goal - self.current_location
            direction_unit_vector = direction / np.linalg.norm(direction)
            self.pref_vel = self.max_speed * direction_unit_vector
        else:
            self.pref_vel = np.zeros(2)

    def calc_vel_penalty(self, robot2, new_vel, omega):
        new_vel = 2 * new_vel - self.velocity
        distance = (self.radius + robot2.radius) ** 2
        a = (new_vel[0] - robot2.velocity[0]) ** 2 + (new_vel[1] - robot2.velocity[1]) ** 2
        b = 2 * ((self.current_location[0] - robot2.current_location[0]) * (new_vel[0] - robot2.velocity[0]) +
                 (self.current_location[1] - robot2.current_location[1]) * (new_vel[1] - robot2.velocity[1]))
        c = (self.current_location[0] - robot2.current_location[0]) ** 2 + (self.current_location[1] -
                                                                            robot2.current_location[1]) ** 2 - distance
        d = b ** 2 - 4 * a * c
        if b > -1e-6 or d <= 0:
            return np.linalg.norm(self.pref_vel - new_vel)
        e = math.sqrt(d)
        t1 = (-b - e) / (2 * a)
        t2 = (-b + e) / (2 * a)
        if t1 < 0 < t2 and b <= -1e-6:
            return 0
        else:
            time_to_col = t1
        penalty = omega * (1 / time_to_col) + np.linalg.norm(self.pref_vel - new_vel)
        return penalty

    def get_neighbours(self, kd_tree, all_robots, radius):
        self.neighbours = []
        neighbours_indices = kd_tree.query_ball_point(self.current_location, radius)
        for i in neighbours_indices:
            if all_robots[i] != self:
                self.neighbours.append(all_robots[i])

    def useable_velocities(self, combined_VO):
        self.AV = [vel for vel in self.velocities if tuple(vel) not in map(tuple, combined_VO)]

    def choose_velocity(self, combined_RVO, VO=False):
        self.preferred_velocity()
        VO = IS_VO
        if (VO):
            optimal_vel = self.pref_vel
        if not self.AV:
            min_penalty = float('inf')
            for neighbour in self.neighbours:
                for vel in combined_RVO:
                    penalty = self.calc_vel_penalty(neighbour, vel, omega=1)
                    if penalty < min_penalty:
                        min_penalty = penalty
                        optimal_vel = vel
        else:
            min_closeness = float('inf')
            for vel in self.AV:
                closeness = np.linalg.norm(self.pref_vel - vel)
                if closeness < min_closeness:
                    min_closeness = closeness
                    optimal_vel = vel
        if (VO):
            self.velocity = optimal_vel
        else:
            self.velocity = (optimal_vel + self.velocity) / 2

    def compute_VO_and_RVO(self, robot2):
        VO = []
        RVO = []
        for vel in self.velocities:
            constraint_val = self.collision_cone_val(vel, robot2)
            if constraint_val < 0:
                VO.append(vel)
                RVO.append((vel + self.velocity) / 2)
        return VO, RVO

    def compute_combined_RVO(self, neighbour_robots):
        combined_VO = []
        combined_RVO = []
        for neighbour_robot in neighbour_robots:
            combined_VO.extend(self.compute_VO_and_RVO(neighbour_robot)[0])
            combined_RVO.extend(self.compute_VO_and_RVO(neighbour_robot)[1])
        return combined_VO, combined_RVO

    def collision_cone_val(self, vel, robot2):
        rx = self.current_location[0]
        ry = self.current_location[1]
        vrx = vel[0]
        vry = vel[1]
        obx = robot2.current_location[0]
        oby = robot2.current_location[1]
        vobx = robot2.velocity[0]
        voby = robot2.velocity[1]
        R = self.radius + robot2.radius + 10
        constraint_val = -((rx - obx) * (vrx - vobx) + (ry - oby) * (vry - voby)) ** 2 + (
                -R ** 2 + (rx - obx) ** 2 + (ry - oby) ** 2) * ((vrx - vobx) ** 2 + (vry - voby) ** 2)
        return constraint_val

    def draw(self, screen):
        if len(self.trail) >= 2:
            pygame.draw.lines(screen, self.color, False, self.trail, 2)
        pygame.draw.circle(screen, self.color, (int(self.current_location[0]), int(self.current_location[1])),
                           self.radius)
        end_point = self.current_location + 10 * self.velocity
        pygame.draw.line(screen, (0, 0, 0), self.current_location, end_point, 2)
        pygame.draw.circle(screen, self.color, (int(self.goal[0]), int(self.goal[1])), 3)
        
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RVO + ORCA")
    clock = pygame.time.Clock()

    # Simulation parameters
    num_robots = 6
    time_horizon = 2.0
    time_step = 0.1
    max_velocity_change = 3
    IS_VO = True  # Set True for VO and False for RVO

    # Define circle parameters
    # circle_center = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2])
    # circle_radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) // 3

    # Define colors for each robot
    robot_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Spawn robots evenly spaced on the circumference of a circle
    # robots = []
    # for i in range(num_robots):
    #     angle = 2 * math.pi * i / num_robots
    #     spawn_x = circle_center[0] + circle_radius * math.cos(angle)
    #     spawn_y = circle_center[1] + circle_radius * math.sin(angle)

    #     # Define opposite direction goal positions
    #     goal_x = circle_center[0] + circle_radius * math.cos(angle + math.pi)
    #     goal_y = circle_center[1] + circle_radius * math.sin(angle + math.pi)

    #     robots.append(Robot(10, max_velocity_change, [spawn_x, spawn_y], [goal_x, goal_y], time_step, i,
    #                         robot_colors[i]))
    
    # Let  us initialize three robots on left side and three on right side
    robots = []
    for i in range(num_robots):
        if i < 3:
            spawn_x = 100
            spawn_y = 100 + i * 100
            goal_x = 700
            goal_y = 100 + i * 100
        else:
            spawn_x = 700
            spawn_y = 100 + (i - 3) * 100
            goal_x = 100
            goal_y = 100 + (i - 3) * 100
        robots.append(Robot(10, max_velocity_change, [spawn_x, spawn_y], [goal_x, goal_y], time_step, i,
                            robot_colors[i]))
    
    # Let us add Static Obstacles
    # Obstacle 
    # MAP STRUCTURE
    '''
    000000000000000000000
    000000111111100000000
    000000000000000000000
    000000000000000000000
    000000111111100000000
    000000000000000000000
    '''
    # Above is the map structure where 0 is free space and 1 is obstacle
    
    # Define Obstacle
    
    

    running = True
    count = 0
    max_count = 1e5

    while running:
        if count > max_count:
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Build KD-tree
        robot_locations = np.array([robot.current_location for robot in robots])
        kdtree = KDTree(robot_locations)

        for robot in robots:
            robot.get_neighbours(kdtree, robots, 75)

        for robot in robots:
            robot.velocities = []
            for i in range(0, 360, 15):
                angle = math.radians(i)
                x = math.cos(angle) * robot.max_speed
                y = math.sin(angle) * robot.max_speed
                robot.velocities.append(np.array([x, y]))

        for robot in robots:
            combined_VO, combined_RVO = robot.compute_combined_RVO(robot.neighbours)
            robot.useable_velocities(combined_VO)
            robot.choose_velocity(combined_RVO, IS_VO)

        for robot in robots:
            robot.update_current_location()

        screen.fill(WHITE)
        for robot in robots:
            robot.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)
        count += 1

    pygame.quit()

if __name__ == "__main__":
    main()
