import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

# Storage
true_x,   true_y   = [], []
noisy_x,  noisy_y  = [], []
kalman_x, kalman_y = [], []

# Parse all stdin upfront
for line in sys.stdin:
    line = line.strip()
    # Match data rows: starts with a time value
    nums = re.findall(r'-?\d+\.\d+', line)
    if len(nums) >= 6:
        try:
            t = float(nums[0])
            true_x.append(float(nums[1]));   true_y.append(float(nums[2]))
            noisy_x.append(float(nums[3]));  noisy_y.append(float(nums[4]))
            kalman_x.append(float(nums[5])); kalman_y.append(float(nums[6]))
        except (IndexError, ValueError):
            continue

# ─── Animate ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Kalman Filter -- 2D Projectile Tracking")
ax.set_xlim(min(true_x) - 5, max(true_x) + 5)
ax.set_ylim(-5, max(true_y) + 10)
ax.grid(True, alpha=0.3)

line_true,   = ax.plot([], [], 'g-',  linewidth=2,   label="True")
line_noisy,  = ax.plot([], [], 'r.',  markersize=4,  label="Noisy")
line_kalman, = ax.plot([], [], 'b-',  linewidth=2,   label="Kalman")
ax.legend()

def update(frame):
    line_true.set_data(true_x[:frame],   true_y[:frame])
    line_noisy.set_data(noisy_x[:frame],  noisy_y[:frame])
    line_kalman.set_data(kalman_x[:frame], kalman_y[:frame])
    return line_true, line_noisy, line_kalman

ani = animation.FuncAnimation(
    fig, update,
    frames=len(true_x) + 1,
    interval=100,  # ms per frame, lower = faster
    blit=True
)

plt.tight_layout()
plt.show()