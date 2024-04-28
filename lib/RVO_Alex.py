import pygame
import numpy as np
import colorsys #HSV to RGB
import time
import math
from scipy.spatial import KDTree
from sympy import symbols, Eq, solve
from typing import List
# from pygame_screen_recorder import pygame_screen_recorder as pgr
# import pygame_screen_record.ScreenRecorder
# import ScreenRecorder
from pygame_screen_record import ScreenRecorder, RecordingSaver
import datetime

# Constants
WHITE = (255, 255, 255)
TRAIL_COLOR = (255, 0, 255)
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 120 # Game FPS, not real-time FPS

# Variables to experiment with [VELOCITY_COUNT, NEIGHBOUR_DIST, ROBOT_COUNT] [36,200,4] [24,200,8] [12,400,16] [28,200,8] [12,200,24], [16,150,28]
END_SIMULATION = False
VELOCITY_COUNT = 36 #36
NEIGHBOUR_DIST = 300 #300
ROBOT_COUNT = 6
IS_VO = False
TIME_STEP = 0.1
ROBOT_RADIUS = 10
ROBOT_MAX_VELOCITY = 5

#SPAWN CIRCLE RADIUS
CIRCLE_SPAWN_RADIUS = 200

# MOST RECENT FILE TO COMBINE OUR WORK

class Robot:
    # def __init__(self, radius, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color):
    def __init__(self, radius, shape, dimensions, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color):
        self.current_location = np.array(spawn_location, dtype=float)
        self.velocity = initial_velocity
        self.goal = np.array(goal_location, dtype=float)
        self.radius = radius        # the outer invisible bounding circle for collision and goal detection 
        self.shape = shape          # e.g. circle, square, rectangle
        self.dimensions = dimensions     # e.g. [radius] [side_length] [x_width y_height]
        self.max_speed = max_speed
        self.time_step = time_step
        self.neighbours = []
        self.AV = []
        self.preferred_velocity()
        # self.velocities = [np.zeros(2), self.pref_vel]
        self.velocities = [self.pref_vel]
        self.trail = []  # Trail to store previous locations
        self.ID = id
        self.color = color


    def is_goal_reached(self, shape):
        """Checks if the robot (or robot as obstacle) has reached its goal.

        Args:
            shape (String): Defines the robot's/obstacle's shape

        Returns:
            bool: True if goal reached.
        """
        if (shape=="circle"): return np.linalg.norm(self.goal - self.current_location) < self.radius
        if (shape=="square"): return np.linalg.norm(self.goal - self.current_location) < self.dimensions[0]
        if (shape=="rectangle"): return np.linalg.norm(self.goal - self.current_location) < min(self.dimensions[0], self.dimensions[1])


    def update_current_location(self):
        self.trail.append(self.current_location.copy())  # Store the current location in the trail
        self.current_location += self.velocity * self.time_step


    def preferred_velocity(self):
        shape = self.shape
        if not self.is_goal_reached(shape):
            direction = self.goal - self.current_location
            direction_unit_vector = direction / np.linalg.norm(direction)
            self.pref_vel = self.max_speed * direction_unit_vector
        else:
            self.pref_vel = np.zeros(2)


    def calc_vel_penalty(self, robot2, new_vel, omega):
        """Calculate the penalty of a velocity
        
        # part of the penalty function is taken from
        # https://stackoverflow.com/questions/43577298/calculating-collision-times-between-two-circles-physics

        Args:
            robot2 (Robot): Neighbor robot
            new_vel (np.array): Velocity inside the RVO
            omega (int): Weight to change the impact of time to impact on penalty

        Returns:
            int: Penalty value
        """
        
        new_vel = 2*new_vel - self.velocity
        
        # distance = (self.radius + robot2.radius) ** 2       # TODO: May need to modify this for non-moving obstacles, like long-skinny rectangles used for walls 
        if (self.shape=="circle" and robot2.shape=="circle"):
            distance = (self.radius + robot2.radius) ** 2
        elif (self.shape=="circle" and robot2.shape!="circle"):
            # distance = (self.radius + robot2.dimensions) ** 2
            distance = distance_to_rectangle(robot2.current_location,robot2.dimensions[0],robot2.dimensions[1],self.current_location,self.radius)
        elif (self.shape!="circle" and robot2.shape=="circle"):
            distance = distance_to_rectangle(self.current_location,self.dimensions[0],self.dimensions[1],robot2.current_location,robot2.radius)
            return 0
        else:
            # print("ERROR: At least one robot should be a circle!")
            distance = 0
            return 0   # return penalty of 0
        
        # Calculate coefficients for a quadratic equation representing the time of collision
        a = (new_vel[0] - robot2.velocity[0]) ** 2 + (new_vel[1] - robot2.velocity[1]) ** 2

        b = 2 * ((self.current_location[0] - robot2.current_location[0]) * (new_vel[0] - robot2.velocity[0]) +
                 (self.current_location[1] - robot2.current_location[1]) * (new_vel[1] - robot2.velocity[1]))

        c = (self.current_location[0] - robot2.current_location[0]) ** 2 + (self.current_location[1] -
                                                                            robot2.current_location[1]) ** 2 - distance

        # Calculate the discriminant of the quadratic equation
        d = b ** 2 - 4 * a * c

        # Ignore glancing collisions that may not cause a response due to limited precision and lead to an infinite loop
        if b > -1e-6 or d <= 0:
            return np.linalg.norm(self.pref_vel - new_vel)

        # Calculate the square root of the discriminant
        e = math.sqrt(d)

        # Calculate the two potential times of collision (t1 and t2)
        t1 = (-b - e) / (2 * a)  # Collision time, +ve or -ve
        t2 = (-b + e) / (2 * a)  # Exit time, +ve or -ve

        # Check conditions to determine the actual collision time
        # If we are overlapping and moving closer, collide now
        if t1 < 0 < t2 and b <= -1e-6:
            # time_to_col = 0 #removed/commented-out because this would cause division by zero.
            self.velocity = self.pref_vel
            return 0
        else:
            time_to_col = t1  # Return the time to collision
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
        # print("\r", end="")
        # if (self.shape=="circle" and (self.ID==1 or self.ID==2)):
        #     # print(f"\rpreferred_velocity:",self.preferred_velocity, end="", flush=True)
            # print(f"\rpref_vel:",self.pref_vel, end="", flush=True)
            # print("Circle ",self.ID,": ",self.AV)
            # draw_usable_velocities(self)
            


    def choose_velocity(self, combined_RVO, VO=False):
        self.preferred_velocity()       # updates self.pref_vel
        
        VO = IS_VO
        
        if (VO):
            optimal_vel = self.pref_vel #ADDED FOR VO
            
        if not self.AV:
            min_penalty = float('inf')
            for neighbour in self.neighbours:
                for vel in combined_RVO:
                    penalty = self.calc_vel_penalty(neighbour, vel, omega=1)
                    if penalty < min_penalty:
                        min_penalty = penalty
                        print('Robot ID: ', self.ID, 'Penalty: ', penalty, '\n')
                        if penalty==0:
                            optimal_vel = self.pref_vel
                        else:
                            optimal_vel = vel
        else:
            min_closeness = float('inf')
            for vel in self.AV:
                closeness = np.linalg.norm(self.pref_vel - vel)
                if closeness < min_closeness:
                    min_closeness = closeness
                    optimal_vel = vel

        if (VO):
            self.velocity = optimal_vel # VO part 
        else:
            self.velocity = (optimal_vel + self.velocity)/2 # RVO part (paper says this on page 3 definition 5)
            # self.velocity = np.min((optimal_vel + self.velocity)/2,self.max_speed) #this should stop nonmovable obstacles from moving weirdly
            # if (self.shape=="circle" and (self.ID==1 or self.ID==2)):
            #    print(f"Name:{self.shape}_{self.ID}: Velocity: {self.velocity}")
                
                # print(f"\rName:{self.shape}_{self.ID}: Velocity: {self.velocity}", end="", flush=True)
                # print(f"Name:{self.shape}_{self.ID}: Velocity: {self.velocity}", flush=True)
        # if optimal_vel.any() != None:
        #     self.velocity = (optimal_vel + self.velocity)/2+
        # else:
        #     self.velocity = self.velocity


    def compute_combined_RVO(self, neighbour_robots):
        combined_VO = []
        combined_RVO = []
        for neighbour_robot in neighbour_robots:
            combined_VO.extend(self.compute_VO_and_RVO(neighbour_robot)[0])
            combined_RVO.extend(self.compute_VO_and_RVO(neighbour_robot)[1])
        return combined_VO, combined_RVO


    def compute_VO_and_RVO(self, robot2):

        VO = []
        RVO = []

        for vel in self.velocities:
            constraint_val = self.collision_cone_val(vel, robot2)
            if constraint_val < 0:
                VO.append(vel)
                RVO.append((vel + self.velocity) / 2)
        return VO, RVO


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
        # if constraint_val >= 0, no collision , else there will be a collision in the future
        constraint_val = -((rx - obx) * (vrx - vobx) + (ry - oby) * (vry - voby)) ** 2 + (
                -R ** 2 + (rx - obx) ** 2 + (ry - oby) ** 2) * ((vrx - vobx) ** 2 + (vry - voby) ** 2)
        # print("constraint_val: ", constraint_val)
        return constraint_val


    def draw(self, screen, shape):
        # Draw the trail if there are at least two points
        if len(self.trail) >= 2:
            # pygame.draw.lines(screen, TRAIL_COLOR, False, self.trail, 2)
            pygame.draw.lines(screen, self.color, False, self.trail, 2)

        if (shape=="circle"):
            if (self.ID==1 and self.ID==2):
                # Draw the robot as a circle
                pygame.draw.circle(screen, self.color, (int(self.current_location[0]), int(self.current_location[1])),
                                self.radius)
            else:
                # Draw the obsticles
                pygame.draw.circle(screen, self.color, (int(self.current_location[0]), int(self.current_location[1])),
                                self.radius, width=3)
            
            # Draw a line representing the direction of the current velocity
            end_point = self.current_location + 10 * self.velocity
            pygame.draw.line(screen, (0, 0, 0), self.current_location, end_point, 2)
            
            # Draw Goal
            pygame.draw.circle(screen, self.color, (int(self.goal[0]), int(self.goal[1])),
                            self.radius, width=3)
            
            # Draw line(s) representing the direction of AV
            for av in self.AV:
                end_point = self.current_location + 5 * av
                pygame.draw.line(screen, (255, 0, 0), self.current_location, end_point, 2)
            
        elif (shape=="rectangle"):
            rect = pygame.Rect(self.current_location[0], self.current_location[1], self.dimensions[0], self.dimensions[1])
            pygame.draw.rect(screen, self.color, rect)
        else:
            print("ERROR: NO SHAPE FOR ROBOT",self.ID)

        # # Draw a line representing the direction of the current velocity
        # end_point = self.current_location + 10 * self.velocity
        # pygame.draw.line(screen, (0, 0, 0), self.current_location, end_point, 2)
        
        # # Draw Goal
        # pygame.draw.circle(screen, self.color, (int(self.goal[0]), int(self.goal[1])),
        #                    self.radius, width=3)


def draw_robots(robots, screen):
    for robot in robots:
        robot.draw(screen, robot.shape)


def velocities_list():
    angles = np.linspace(0, 2*np.pi, num=VELOCITY_COUNT)
    x_values = np.cos(angles)
    y_values = np.sin(angles)
    for i in np.arange(0, 6, 1):
        if i>0:
            velocities = [i * np.array([x,y]) for x,y in zip(x_values,y_values)]
    # print("velocities: ",velocities)
    # draw_possible_velocities(velocities)
    return velocities

# def draw_possible_velocities():
#     # Draw a line representing the direction of the current velocity
#     end_point = self.current_location + 10 * self.velocity
#     pygame.draw.line(screen, (0, 0, 0), self.current_location, end_point, 2)


def generate_rainbow_colors(num_colors: int) -> List[tuple]:
    """
    Generates a list of RGB values for a rainbow with the specified number of colors.

    Args:
        num_colors: The number of colors in the rainbow.

    Returns:
        A list of RGB values.
    """

    if num_colors == 1:
        return [(255, 0, 0)]

    hues = [i / (num_colors) for i in range(num_colors)]
    rgbs = [
        tuple(
            [int(c * 255) for c in colorsys.hsv_to_rgb(h, 1.0, 1.0)]
        ) for h in hues
    ]
    return rgbs

def create_circular_locations(num_robots, radius, center):
    """
    This function creates two lists:
    - spawn_locations: list of spawn locations for each robot
    - goal_locations: list of goal locations for each robot
    around a circle with a specific center.

    Args:
        num_robots: integer, number of robots
        radius: integer, radius of the circle
        center: tuple, (x, y) coordinates of the circle's center

    Returns:
        spawn_locations: list of tuples representing spawn locations
        goal_locations: list of tuples representing goal locations
    """
    spawn_locations = []
    goal_locations = []
    theta = np.linspace(0, 2*np.pi, num_robots + 1)[:-1]
    
    for i, angle in enumerate(theta):
        x_offset = int(radius * np.cos(angle))
        y_offset = int(radius * np.sin(angle))
        x_spawn = center[0] + x_offset
        y_spawn = center[1] + y_offset
        spawn_locations.append((x_spawn, y_spawn))

        x_offset = int(radius * np.cos(angle + np.pi))
        y_offset = int(radius * np.sin(angle + np.pi))
        x_goal = center[0] + x_offset
        y_goal = center[1] + y_offset
        goal_locations.append((x_goal, y_goal))
        
    return spawn_locations, goal_locations


def create_nonmoving_obstacle_locations(pattern):
    """Takes in a pattern and intended to create a field of nonmoving robots to be avoided.
        If modified, perhaps the obstacles to be avoided could move.

    Args:
        pattern (String): Field to create

    Returns:
        list: spawn_locations - where the obstacles to spawn
        list: goal_locations - where the obtacles move to
        
    TODO: Modify code to accept different types of layout for non-moving and/or moving obstacles.
    TODO: Consider using a PNG B+W map for non-moving obtacles (Like from WPI RBE UG A.I. for Robotics final project)
    """
    
    spawn_locations = []
    goal_locations = []
    
    if (pattern=="corridor"):
        vertical_wall_1 = [50,50]
        vertical_wall_2 = [50,SCREEN_HEIGHT-50]
        vertical_wall_3 = [SCREEN_WIDTH-50,50]
        verticle_wall_4 = [SCREEN_WIDTH-50,SCREEN_HEIGHT-50]
        horizontal_wall_1 = [SCREEN_WIDTH/2,200]
        horizontal_wall_2 = [SCREEN_WIDTH/2,400]
        
        spawn_locations.append(vertical_wall_1,vertical_wall_2,vertical_wall_3,verticle_wall_4,horizontal_wall_1,horizontal_wall_2)
        goal_locations.append(vertical_wall_1,vertical_wall_2,vertical_wall_3,verticle_wall_4,horizontal_wall_1,horizontal_wall_2)
    else:
        print("NO PATTERN FOR OBSTACLES")
    
    return spawn_locations, goal_locations


def create_robots(num_robots, radius):
    """
    This function creates a list of robots based on the number of robots needed.

    Args:
    num_robots: integer, number of robots
    radius: integer, radius of the circle
    rgb_list: list of tuples representing robot colors

    Returns:
    robots: list of Robot objects
    """
    robots = []
    rgb_list = generate_rainbow_colors(num_robots)
    center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    
    for i in range(num_robots):
        robot_name = "Robot{}".format(i+1)
        # radius, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color
        robot = Robot(
            radius=ROBOT_RADIUS,
            max_speed=ROBOT_MAX_VELOCITY,
            spawn_location=create_circular_locations(num_robots, radius, center)[0][i],
            goal_location=create_circular_locations(num_robots, radius, center)[1][i],
            time_step=TIME_STEP,
            initial_velocity=np.array([0, 0]),
            id=robot_name,
            color=rgb_list[i]
        )
        robots.append(robot)

    return robots


def create_obstacles(num_obstacles, pattern, shape, dimensions):
    obstacles = []
    # rgb_list = generate_rainbow_colors(num_obstacles)
    color = [0,100,0]
    center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    
    # VERTICAL RECTANGLES
    for i in range(num_obstacles,5):
        obstacle_name = "Obstacle{}".format(i+1)
        # radius, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color
        robot = Robot(
            radius=ROBOT_RADIUS,
            dimensions=[50,100],
            max_speed=0,                                                            # non-moving
            spawn_location=create_nonmoving_obstacle_locations(pattern)[0][i],
            goal_location=create_circular_locations(pattern)[1][i],
            time_step=TIME_STEP,
            initial_velocity=np.array([0, 0]),
            id=obstacle_name,
            color=color
        )
        obstacles.append(robot)
    
    # HORIONTAL RECTANGLES
    for i in range(num_obstacles+4,7):
        obstacle_name = "Obstacle{}".format(i+1)
        # radius, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color
        robot = Robot(
            radius=ROBOT_RADIUS,
            dimensions=[300,50],
            max_speed=0,                                                            # non-moving
            spawn_location=create_nonmoving_obstacle_locations(pattern)[0][i],
            goal_location=create_circular_locations(pattern)[1][i],
            time_step=TIME_STEP,
            initial_velocity=np.array([0, 0]),
            id=obstacle_name,
            color=color
        )
        obstacles.append(robot)   
        
    return obstacles


def distance_to_rectangle(center_rectangle, width, height, center_circle, radius):
    """
    This function takes the center point of a rectangle, its width, height, 
    the center point of a circle, and the circle's radius as input and returns 
    the shortest distance between the outer wall of the rectangle and the edge 
    of the circle.

    Args:
        center_rectangle (list): A list of length 2 containing the x and y coordinates of the rectangle's center point.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        center_circle (list): A list of length 2 containing the x and y coordinates of the circle's center point.
        radius (float): The radius of the circle.

    Returns:
        float: The shortest distance between the outer wall of the rectangle and the edge of the circle.
    """
    # Get the top left and bottom right corners of the rectangle
    top_left = [center_rectangle[0] - width / 2, center_rectangle[1] + height / 2]
    bot_right = [center_rectangle[0] + width / 2, center_rectangle[1] - height / 2]

    # Find the closest point on the rectangle's edge to the circle's center
    closest_point = center_rectangle.copy()
    
    # Check each side of the rectangle
    for i in range(2):
        if center_circle[i] < top_left[i]:
            closest_point[i] = top_left[i]
        elif center_circle[i] > bot_right[i]:
            closest_point[i] = bot_right[i]

    # Calculate the distance between the closest point and the circle's center
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(center_circle, closest_point)))

    # Check if the circle is completely inside the rectangle
    if distance > radius:
        return distance - radius
    else:
        return 0
        

def update_time_counter(screen, start_time):
    # Update the time counter
    elapsed_time = time.time() - start_time
    # Add a black rectangle for the time counter
    font = pygame.font.Font(None, 24)
    # text = font.render(f"Time: {time.time():.2f}", True, (0, 0, 0))
    text = font.render(f"Time Step: {TIME_STEP}, Game FPS: {FPS}, Time: {elapsed_time:.2f} s", True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.topleft = (10, 10)
    screen.blit(text, text_rect)
    return elapsed_time


def draw_lines(self, screen, inputs, magnitude=10):
    for robot in self:
        if (robot.ID==1 or robot.ID==2):
            for input in inputs:
                end_point = robot.current_location + magnitude * input
                pygame.draw.line(screen, (200, 200, 200), (int(robot.current_location[0]), int(robot.current_location[1])), (int(end_point[0]), int(end_point[1])),  )
                # pygame.draw.circle(screen, (0,0,0), (int(end_point[0]), int(end_point[1])),
                #                     robot.radius, width=3)
       

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Simulation")
    clock = pygame.time.Clock()
    
    # Start the timer
    start_time = time.time()

    # # Create robots with initial velocity
    # Diagonal Robots
    # robot1 = Robot(radius=10, max_speed=5, spawn_location=(50, 50), goal_location=(250, 250), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=1, color=(255,0,0))
    # robot2 = Robot(radius=10, max_speed=5, spawn_location=(250, 250), goal_location=(50, 50), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=2, color=(0,0,255))
    # robots = [robot1, robot2]
    
    # Create robots - spawn in circle
    num_robots = ROBOT_COUNT          # number of robots
    spawn_radius = CIRCLE_SPAWN_RADIUS      # radius of spawning circle #was 200 with screen (600,600)
    # robots = create_robots(num_robots, spawn_radius) # creates and sets spawn locations in a circle around the center of the screen
    
    
    # --------------- MANUALLY CREATING OBSTACLES AND ROBOTS - ATTEMPT: 1 ----------------------
    # wall_LEFT_UPPER = Robot(radius=100, shape="rectangle", dimensions=[50,100], max_speed=0, spawn_location=(100, 150), goal_location=(100, 150), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=1, color=(0,0,0))
    # wall_LEFT_LOWER = Robot(radius=100, shape="rectangle", dimensions=[50,100], max_speed=0, spawn_location=(100, 350), goal_location=(100, 350), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=2, color=(0,0,0))
    # wall_RIGHT_UPPER = Robot(radius=100, shape="rectangle", dimensions=[50,100], max_speed=0, spawn_location=(450, 150), goal_location=(450, 150), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=3, color=(0,0,0))
    # wall_RIGHT_LOWER = Robot(radius=100, shape="rectangle", dimensions=[50,100], max_speed=0, spawn_location=(450, 350), goal_location=(450, 350), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=4, color=(0,0,0))
    # wall_CENTER_UPPER = Robot(radius=100, shape="rectangle", dimensions=[300,50], max_speed=0, spawn_location=(SCREEN_WIDTH/2-150, 200), goal_location=(SCREEN_WIDTH/2-150, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=5, color=(0,0,0))
    # wall_CENTER_LOWER = Robot(radius=100, shape="rectangle", dimensions=[300,50], max_speed=0, spawn_location=(SCREEN_WIDTH/2-150, 350), goal_location=(SCREEN_WIDTH/2-150, 350), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=6, color=(0,0,0))
    
    # # robots = []
    # # # create non-moving obstacles
    # # for i in range(1,5):
        
    
    # robot1 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(50, 50), goal_location=(550, 550), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=7, color=(255,0,0))
    # robot2 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(50, 550), goal_location=(550, 50), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=8, color=(0,0,255))
    
    # robots = [wall_LEFT_UPPER,wall_LEFT_LOWER,wall_RIGHT_UPPER,wall_RIGHT_LOWER,wall_CENTER_UPPER,wall_CENTER_LOWER,robot1,robot2]
    # --------------- END: MANUALLY CREATING OBSTACLES AND ROBOTS - ATTEMPT: 1 ----------------------
    
    
    # --------------- MANUALLY CREATING OBSTACLES AND ROBOTS - ATTEMPT: 2 ----------------------
    # wall_UPPER_1 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(100, 200), goal_location=(100, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=21, color=(0,0,0))
    # wall_UPPER_2 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(200, 200), goal_location=(200, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=22, color=(0,0,0))
    # wall_UPPER_3 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(300, 200), goal_location=(300, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=23, color=(0,0,0))
    # wall_UPPER_4 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(400, 200), goal_location=(400, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=24, color=(0,0,0))
    # wall_UPPER_5 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(500, 200), goal_location=(500, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=25, color=(0,0,0))
    
    # wall_LOWER_1 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(100, 400), goal_location=(100, 400), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=31, color=(0,0,0))
    # wall_LOWER_2 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(200, 400), goal_location=(200, 400), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=32, color=(0,0,0))
    # wall_LOWER_3 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(300, 400), goal_location=(300, 400), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=33, color=(0,0,0))
    # wall_LOWER_4 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(400, 400), goal_location=(400, 400), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=34, color=(0,0,0))
    # wall_LOWER_5 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(500, 400), goal_location=(500, 400), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=35, color=(0,0,0))
    
    # # robots = []
    # # # create non-moving obstacles
    # # for i in range(1,5):
        
    
    # robot1 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(25, 50), goal_location=(550, 550), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=1, color=(255,0,0))
    # robot2 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(50, 550), goal_location=(550, 50), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=2, color=(0,0,255))
    
    # robots = [robot1,robot2,wall_LOWER_1,wall_LOWER_2,wall_LOWER_3,wall_LOWER_4,wall_LOWER_5,wall_UPPER_1,wall_UPPER_2,wall_UPPER_3,wall_UPPER_4,wall_UPPER_5]
    # --------------- END: MANUALLY CREATING OBSTACLES AND ROBOTS - ATTEMPT: 2 ----------------------
    
    
    
    
    
    
        # --------------- MANUALLY CREATING OBSTACLES AND ROBOTS - ATTEMPT: 3 ----------------------
    wall_UPPER_1 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 0), goal_location=(SCREEN_WIDTH/2, 0), time_step=0,
                   initial_velocity=np.array([0, 0]), id=21, color=(0,0,0))
    wall_UPPER_2 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 100), goal_location=(SCREEN_WIDTH/2, 100), time_step=0,
                   initial_velocity=np.array([0, 0]), id=22, color=(0,0,0))
    wall_UPPER_3 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 200), goal_location=(SCREEN_WIDTH/2, 200), time_step=0,
                   initial_velocity=np.array([0, 0]), id=23, color=(0,0,0))
    # wall_UPPER_4 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 300), goal_location=(SCREEN_WIDTH/2, 300), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=24, color=(0,0,0))
    # wall_UPPER_5 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 400), goal_location=(SCREEN_WIDTH/2, 400), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=25, color=(0,0,0))
    
    # wall_LOWER_1 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 400), goal_location=(SCREEN_WIDTH/2, 200), time_step=0,
    #                initial_velocity=np.array([0, 0]), id=31, color=(0,0,0))
    # wall_LOWER_2 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 400), goal_location=(SCREEN_WIDTH/2, 300), time_step=0,
                #    initial_velocity=np.array([0, 0]), id=32, color=(0,0,0))
    wall_LOWER_3 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 400), goal_location=(SCREEN_WIDTH/2, 400), time_step=0,
                   initial_velocity=np.array([0, 0]), id=33, color=(0,0,0))
    wall_LOWER_4 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, 400), goal_location=(SCREEN_WIDTH/2, 500), time_step=0,
                   initial_velocity=np.array([0, 0]), id=34, color=(0,0,0))
    wall_LOWER_5 = Robot(radius=50, shape="circle", dimensions=[50], max_speed=0, spawn_location=(SCREEN_WIDTH/2, SCREEN_HEIGHT), goal_location=(SCREEN_WIDTH/2, SCREEN_HEIGHT), time_step=0,
                   initial_velocity=np.array([0, 0]), id=35, color=(0,0,0))
    
    # robots = []
    # # create non-moving obstacles
    # for i in range(1,5):
        
    
    robot1 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(15, 50), goal_location=(550, 550), time_step=TIME_STEP,
                   initial_velocity=np.array([0, 0]), id=1, color=(255,0,0))
    robot2 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(18, 150), goal_location=(550, 450), time_step=TIME_STEP,
                   initial_velocity=np.array([0, 0]), id=2, color=(0,0,255))
    robot3 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(20, 300), goal_location=(550, 300), time_step=TIME_STEP,
                   initial_velocity=np.array([0, 0]), id=1, color=(0,255,0))
    robot4 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(23, 450), goal_location=(550, 150), time_step=TIME_STEP,
                   initial_velocity=np.array([0, 0]), id=2, color=(255,0,255))
    robot5 = Robot(radius=10, shape="circle", dimensions=[10], max_speed=5, spawn_location=(25, 550), goal_location=(550, 50), time_step=TIME_STEP,
                   initial_velocity=np.array([0, 0]), id=1, color=(255,255,0))

    
    robots = [robot1,robot2,robot3,robot4,robot5,wall_LOWER_3,wall_LOWER_4,wall_LOWER_5,wall_UPPER_1,wall_UPPER_2,wall_UPPER_3]
    # --------------- END: MANUALLY CREATING OBSTACLES AND ROBOTS - ATTEMPT: 3 ----------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
    # pattern = "corridor"
    # obstacles = create_nonmoving_obstacle_locations(pattern=="corridor")
    
    
    

    # Compute velocities list for all robots
    velocities = velocities_list()

    for robot in robots:
        if (robot.shape=="circle"):
            robot.velocities += velocities

    running = True
    time_elapsed_shown = False

    ### RECORD
    # recorder = ScreenRecorder(30) # Pass your desired fps
    # recorder.start_rec() # Start recording

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        count = 0
        for robot in robots:
            if robot.is_goal_reached(robot.shape):
                count += 1
        if count == len(robots):
            if not time_elapsed_shown:
                print(f"Time Elapsed: {elapsed_time:.2f} [s]")
                time_elapsed_shown = True
            if END_SIMULATION:
                running = False

        # Update KDTree
        kd_tree = KDTree([robot.current_location for robot in robots])
        combined_vos = []
        combined_rvos = []
        for robot in robots:
            if (robot.ID==1 or robot.ID==2):
                robot.get_neighbours(kd_tree, robots, radius=NEIGHBOUR_DIST) #increased from 100
                vo, rvo = robot.compute_combined_RVO(robot.neighbours)
                # print("rvo",rvo)
                # draw_lines(robot,screen,rvo,10)
                
                # for input in rvo:
                #     end_point = robot.current_location + 5 * input
                #     # pygame.draw.line(screen, (200, 200, 200), (int(self.current_location[0]), int(self.current_location[1])), (int(end_point[0]), int(end_point[1])), 2)
                #     pygame.draw.circle(screen, (100,100,100), (int(end_point[0]), int(end_point[1])),
                #                         robot.radius, width=3)
                
                
                combined_vos.append(vo)
                combined_rvos.append(rvo)

        # Update robots
        i = 0
        for robot in robots:
            if (robot.ID==1 or robot.ID==2):
                robot.useable_velocities(combined_vos[i])
                robot.choose_velocity(combined_rvos[i])
                robot.update_current_location()
            i += 1

        # Draw on the screen
        screen.fill(WHITE)
        draw_lines(robots,screen,rvo,10) #draws RVO lines to bounding points?
        draw_robots(robots, screen)
        elapsed_time = update_time_counter(screen, start_time)
        
        

        pygame.display.flip()
        clock.tick(FPS)

    # print("Start Time:",start_time)
    print(f"Time Elapsed: {elapsed_time:.3f} [s]")
    
    
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    ### END RECORDING
    # recorder.stop_rec()	# stop recording
    # recording = recorder.get_single_recording() # returns a Recording
    # save(recording,(f"{formatted_datetime} my_recording","mp4")) # save the recording as mp4
    
    # recorder.stop_rec().get_single_recording().save((f"{formatted_datetime} my_recording","mp4"))
    
    # recordings = recorder.get_recordings()
    # saver = RecordingSaver(recordings, "mp4", "saved_files")
    # saver.save()
    
    pygame.display.quit()
    pygame.quit()


main()
