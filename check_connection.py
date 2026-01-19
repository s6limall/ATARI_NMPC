from unitree_go2 import Go2Robot

# Initialize the robot
robot = Go2Robot()
robot.start()

while True:
    # Get the current joint states
    # This usually returns positions, velocities, torques
    joint_states = robot.get_joint_state()
    print("Positions:", joint_states.q)
    print("Velocities:", joint_states.dq)
    print("Torques:", joint_states.tau)
