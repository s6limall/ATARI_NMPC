import pygame

# Initialize Pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Check if any joystick is connected
if pygame.joystick.get_count() == 0:
    print("No joystick detected.")
    exit()

# Use the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick detected: {joystick.get_name()}")
print(f"Number of axes: {joystick.get_numaxes()}")
print(f"Number of buttons: {joystick.get_numbuttons()}")
print(f"Number of hats: {joystick.get_numhats()}")

try:
    while True:
        pygame.event.pump()  # Process event queue

        # Print axes
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        print("Axes:", axes)

        # Print buttons
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        print("Buttons:", buttons)

        # Print hat switches (D-pad)
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
        print("Hats:", hats)

        print("-" * 40)
        pygame.time.wait(200)  # Small delay to avoid flooding
except KeyboardInterrupt:
    print("Exiting...")
finally:
    joystick.quit()
    pygame.quit()
