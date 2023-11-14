"""This script simulates a collision between billiard balls, animated on a graph using matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Set up initial conditions
alpha = 0.8  # damping for normal velocity
beta = 0.98  # damping for tangential velocity

radius = 0.05  # radius of both balls
start_time = 0
end_time = 50
delta_t = 0.01  # Euler time step
interval = 10  # interval between frames in ms

red_coordinates = np.array([0.75, 5 * radius])  # initial position of red ball
red_velocity = 3*np.array([-0.1, 0.5])  # initial velocity of red ball

blue_coordinates = np.array([0.25, 5.5 * radius])  # initial position of blue ball
blue_velocity = 3*np.array([0.11, 0.2])  # initial velocity of blue ball

# Set up plot
fig, ax = plt.subplots()
x_limit = 1
y_limit = 1
ax.set_xlim([0, x_limit])
ax.set_ylim([0, y_limit])
ax.set_aspect('equal')
ax.set_title('Billiard Balls')
ax.set_xlabel('x')
ax.set_ylabel('y')

# draw balls
red_ball = plt.Circle(red_coordinates, radius, color='r')
blue_ball = plt.Circle(blue_coordinates, radius, color='b')

ax.add_artist(red_ball)
ax.add_artist(blue_ball)

current_time = 0


def animate(i):
    new_red_coordinates = red_ball.center + red_velocity * delta_t
    new_blue_coordinates = blue_ball.center + blue_velocity * delta_t

    # check for collision with walls
    if new_red_coordinates[0] - radius < 0:  # collision with left wall
        # find new timestep to place in contact with wall
        delta_t_new = abs(new_red_coordinates[0] - radius) / abs(red_velocity[0])
        # update positions with new timestep
        new_red_coordinates[0] = radius
        new_red_coordinates[1] = red_ball.center[1] + red_velocity[1] * delta_t_new

        red_velocity[0] = -alpha * red_velocity[0]
        red_velocity[1] = beta * red_velocity[1]

    elif new_red_coordinates[0] + radius > x_limit:  # collision with right wall
        delta_t_new = abs(x_limit - radius - new_red_coordinates[0]) / abs(red_velocity[0])
        new_red_coordinates[0] = x_limit - radius
        new_red_coordinates[1] = red_ball.center[1] + red_velocity[1] * delta_t_new

        red_velocity[0] = -alpha * red_velocity[0]
        red_velocity[1] = beta * red_velocity[1]

    elif new_red_coordinates[1] - radius < 0:  # collision with bottom wall
        delta_t_new = abs(new_red_coordinates[1] - radius) / abs(red_velocity[1])
        new_red_coordinates[0] = red_ball.center[0] + red_velocity[0] * delta_t_new
        new_red_coordinates[1] = radius

        red_velocity[1] = -alpha * red_velocity[1]
        red_velocity[0] = beta * red_velocity[0]

    elif new_red_coordinates[1] + radius > y_limit:  # collision with top wall
        delta_t_new = abs(y_limit - radius - new_red_coordinates[1]) / abs(red_velocity[1])
        new_red_coordinates[0] = red_ball.center[0] + red_velocity[0] * delta_t_new
        new_red_coordinates[1] = y_limit - radius

        red_velocity[1] = -alpha * red_velocity[1]
        red_velocity[0] = beta * red_velocity[0]

    if new_blue_coordinates[0] - radius < 0:  # collision with left wall
        delta_t_new = abs(new_blue_coordinates[0] - radius) / abs(blue_velocity[0])
        new_blue_coordinates[0] = radius
        new_blue_coordinates[1] = blue_ball.center[1] + blue_velocity[1] * delta_t_new

        blue_velocity[0] = -alpha * blue_velocity[0]
        blue_velocity[1] = beta * blue_velocity[1]

    elif new_blue_coordinates[0] + radius > x_limit:  # collision with right wall
        delta_t_new = abs(x_limit - radius - new_blue_coordinates[0]) / abs(blue_velocity[0])
        new_blue_coordinates[0] = x_limit - radius
        new_blue_coordinates[1] = blue_ball.center[1] + blue_velocity[1] * delta_t_new

        blue_velocity[0] = -alpha * blue_velocity[0]
        blue_velocity[1] = beta * blue_velocity[1]

    elif new_blue_coordinates[1] - radius < 0:  # collision with bottom wall
        delta_t_new = abs(new_blue_coordinates[1] - radius) / abs(blue_velocity[1])
        new_blue_coordinates[0] = blue_ball.center[0] + blue_velocity[0] * delta_t_new
        new_blue_coordinates[1] = radius

        blue_velocity[1] = -alpha * blue_velocity[1]
        blue_velocity[0] = beta * blue_velocity[0]

    elif new_blue_coordinates[1] + radius > y_limit:  # collision with top wall
        delta_t_new = abs(y_limit - radius - new_blue_coordinates[1]) / abs(blue_velocity[1])
        new_blue_coordinates[0] = blue_ball.center[0] + blue_velocity[0] * delta_t_new
        new_blue_coordinates[1] = y_limit - radius

        blue_velocity[1] = -alpha * blue_velocity[1]
        blue_velocity[0] = beta * blue_velocity[0]

    # check for collision with other ball
    distance = np.linalg.norm(new_red_coordinates - new_blue_coordinates)
    relative_velocity = np.linalg.norm(red_velocity - blue_velocity)
    relative_distance = np.linalg.norm(red_ball.center - blue_ball.center)

    if distance <= 2 * radius:
        # calculate new timestep to place in contact
        delta_t_new = delta_t * (relative_distance - 2 * radius) / relative_velocity
        # update positions with new timestep
        new_red_coordinates = red_ball.center + red_velocity * delta_t_new
        new_blue_coordinates = blue_ball.center + blue_velocity * delta_t_new

        # calculate new velocities

        # find direction of the normal vector and convert to unit normal vector
        unit_normal = (new_red_coordinates - new_blue_coordinates) / np.linalg.norm(
            new_red_coordinates - new_blue_coordinates)
        tangent_vector = np.array([unit_normal[1], -unit_normal[0]])

        # find normal and tangential components of velocity
        red_velocity_in_norm = np.dot(red_velocity, unit_normal)
        red_velocity_in_tang = np.dot(red_velocity, tangent_vector)
        blue_velocity_in_norm = np.dot(blue_velocity, unit_normal)
        blue_velocity_in_tang = np.dot(blue_velocity, tangent_vector)

        # output velocities: normal components are swapped, tangential components are unchanged
        velocity_red_out = np.array([blue_velocity_in_norm, red_velocity_in_tang])
        velocity_blue_out = np.array([red_velocity_in_norm, blue_velocity_in_tang])

        # convert back to x-y reference frame
        red_velocity[0] = np.dot(velocity_red_out, unit_normal)
        red_velocity[1] = np.dot(velocity_red_out, tangent_vector)
        blue_velocity[0] = np.dot(velocity_blue_out, unit_normal)
        blue_velocity[1] = np.dot(velocity_blue_out, tangent_vector)

    red_ball.center = new_red_coordinates
    blue_ball.center = new_blue_coordinates


anim = FuncAnimation(fig, animate, frames=np.arange(start_time, end_time, delta_t), interval=interval)
# save animation
anim.save('billiards.mp4', writer='ffmpeg', fps=end_time*2)

