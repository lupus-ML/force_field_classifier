import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Parameters
# -----------------------------
number_particles = 100
time_steps = 200
dt = 0.05
g = 1.0
m = 10.0
softening = 0.1
STABLE_RADIUS = 15.0  # Adjusted for realistic label distribution

# -----------------------------
# Initialize Particles
# -----------------------------
np.random.seed(42)
positions = np.random.uniform(-5, 5, size=(number_particles, 2))
velocities = np.random.uniform(-1, 1, size=(number_particles, 2))
trajectories = np.zeros((number_particles, time_steps, 2))
trajectories[:, 0, :] = positions

# -----------------------------
# Force Function
# -----------------------------
def compute_gravitational_force(pos):
    r = np.linalg.norm(pos, axis=1).reshape(-1, 1)
    force_magnitude = -g * m / (r**2 + softening**2)
    force_direction = -pos / (r + softening)
    return force_magnitude * force_direction

# -----------------------------
# Simulate Particle Motion
# -----------------------------
for t in range(1, time_steps):
    forces = compute_gravitational_force(positions)
    velocities += forces * dt
    positions += velocities * dt
    trajectories[:, t, :] = positions

# ============================================================
# FORCE FIELD + EQUATION VISUALIZATION
# ============================================================

x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
pos_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
r_grid = np.linalg.norm(pos_grid, axis=1).reshape(-1, 1)

force_magnitude_grid = -g * m / (r_grid**2 + softening**2)
force_direction_grid = -pos_grid / (r_grid + softening)
F_grid = force_magnitude_grid * force_direction_grid
Fx = F_grid[:, 0].reshape(X.shape)
Fy = F_grid[:, 1].reshape(Y.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.quiver(X, Y, Fx, Fy, color='white', alpha=0.7, scale=20, width=0.003)
ax1.set_title("Force Field Visualization")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

ax2.axis('off')
ax2.text(0.05, 0.85, r"$\vec{F} = -\frac{GMm}{r^2} \hat{r}$", fontsize=12, color='white')
ax2.text(0.05, 0.70, r"$\vec{F} = -\nabla V(x, y)$", fontsize=12, color='white')
ax2.text(0.05, 0.55, r"$V(x, y) = -\frac{1}{\sqrt{x^2 + y^2 + \epsilon}}$", fontsize=12, color='white')
ax2.set_title("Equations")
plt.style.use('dark_background')
plt.show()

# ============================================================
# PARTICLE TRAJECTORY VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 8))
colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF0000', '#00FF00',
          '#0000FF', '#FFA500', '#800080', '#FFC0CB', '#008000']
for i in range(10):
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], color=colors[i], label=f'Particle {i+1}')
plt.scatter(0, 0, color='red', label='Central Mass')
plt.title("Particle Trajectories in Force Field", color='white')
plt.xlabel("X", color='white')
plt.ylabel("Y", color='white')
plt.legend(labelcolor='white')
plt.axis('equal')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.style.use('dark_background')
plt.show()

# ============================================================
# STABILITY LABELING + DATASET CREATION
# ============================================================

labels = []
initial_conditions = []
final_distances = []

for i in range(number_particles):
    x0, y0 = trajectories[i, 0]
    vx0, vy0 = velocities[i]
    final_pos = trajectories[i, -1]
    distance = np.linalg.norm(final_pos)
    final_distances.append(distance)
    
    label = 'Stable' if distance < STABLE_RADIUS else 'Chaotic'
    labels.append(label)
    initial_conditions.append([x0, y0, vx0, vy0])

# Print distance stats
print(f"\nFinal Distance Stats:")
print(f"Min: {min(final_distances):.2f}, Max: {max(final_distances):.2f}, Mean: {np.mean(final_distances):.2f}")

# Save dataset
df = pd.DataFrame(initial_conditions, columns=['x0', 'y0', 'vx0', 'vy0'])
df['label'] = labels
df.to_csv('particle_dataset.csv', index=False)

# Label summary
print("\nLabel Distribution:")
print(df['label'].value_counts())
print("\nDataset saved to 'particle_dataset.csv'")