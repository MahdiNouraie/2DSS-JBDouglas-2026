"""Print every simulation number that appears on the slides, from the saved output."""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
d = json.load(open(os.path.join(OUT, "main_setting.json")))
F = np.array(d["freq8"]); dl = d["deltas"]   # columns: x1..x5 (relevant), then 3 impostors
i = dl.index
print("Main setting: n = 100, p = 1000, s = 5 relevant variables, 3 impostors, lambda = sqrt(log p / n1) = %.3f" % d["lam"])
print("\nSlide 8 (Stability Selection, delta = 0, 200 half-samples)")
print("  relevant x1..x5 :", F[0, :5].round(3), " average %.3f" % F[0, :5].mean())
print("  impostors       :", F[0, 5:].round(3), " average %.3f" % F[0, 5:].mean())
print("\nSlide 10 (noise sweep)")
for dd in (0, 0.5, 0.75):
    print("  delta = %.2f: weaker relevant (x1..x4) avg %.3f, impostors avg %.3f, strong x5 %.3f" % (dd, F[i(dd), :4].mean(), F[i(dd), 5:].mean(), F[i(dd), 4]))
print("\nSlide 13 (noise path), averages per delta")
print("  delta    :", dl)
print("  weak relevant:", np.round(F[:, :4].mean(1), 3).tolist())
print("  impostors:", np.round(F[:, 5:].mean(1), 3).tolist())
print("  at delta = 3 the weaker relevant variables are selected %.3f of the time" % F[i(3), :4].mean())
