#!/usr/bin/env python3
import antimony
import roadrunner
import numpy as np

# ── Minimal Antimony model ──
ANT_MODEL = """
model periodic_test()
  U = 50;
  V = 50;
  W = 50;
  X = 50;
  Y = 50;
end
"""

# 1) Load model into RoadRunner
antimony.loadAntimonyString(ANT_MODEL)
rr = roadrunner.RoadRunner(antimony.getSBMLString())

# 2) Switch to the Gillespie (SSA) integrator
rr.integrator = 'gillespie'

# 3) Simulate first segment (t=0 to t=10, with 100 output points)
seg1 = rr.simulate(0, 10, 100)
print("First segment completed. Time range:", seg1[0,0], "to", seg1[-1,0])

# 4) Extract last state and apply 60% drop to Y
last_counts = seg1[-1, 1:].copy()   # ignore the time column
species     = rr.getFloatingSpeciesIds()
y_idx       = species.index('Y')
print(f"Before perturbation: Y = {last_counts[y_idx]}")
last_counts[y_idx] *= 0.4
print(f"After  60% drop:    Y = {last_counts[y_idx]}")

# 5) Reset the integrator state *but keep* the current species values
rr.resetAll()  
for idx, sp in enumerate(species):
    rr[sp] = last_counts[idx]

# 6) Simulate second segment (t=10 to t=20, with 100 output points)
seg2 = rr.simulate(10, 20, 100)
print("Second segment completed. First few time points:", seg2[:5,0])