# force_field_classifier

Force Field Classifier – Predicting Chaos with Physics + Machine Learning

⸻

📜 Intro:

What if we could teach a machine to understand gravity?
Not by equations.
But by watching particles move…
…and learning when they’ll orbit peacefully — or fly into chaos.

⸻

💡 The Idea:

I built a simulator where particles move under gravitational pull, just like planets around a star.
Each particle starts at a random position, with random velocity.
At the end of its journey, it’s labeled:
🟢 Stable – if it stayed close
🔴 Chaotic – if it escaped orbit

Then I trained a Machine Learning model to predict chaos from initial conditions.

⸻

🧠 What’s Inside:
🌌 Physics-inspired data from vector fields
⚙️ Stability labeling using distance thresholds
🌲 Random Forest Classifier (100% accuracy on test set!)
🎨 Force field visualization with quiver plots
🧭 Decision boundary mapping
🔁 Real-time animation (coming next)

⸻

🧪 Technologies:
Python / NumPy / Pandas / Matplotlib
scikit-learn
Pure vector calculus + Newtonian physics
