#!/bin/bash
# Re-runs every simulation used in the talk and prints the numbers shown on the slides.
set -e
cd "$(dirname "$0")/simulations"
python3 main_setting.py          # slides 5, 8, 10 and 13 (about 30 s)
python3 illustration_ss.py       # slide 6 (illustration)
python3 check_slide_numbers.py
# Optional: python3 robustness_seeds.py   (same setting over seeds 1-12, about 10 min)
# Animations (optional, needs manim):  cd ../animations && manim -qh scenes.py NoiseSweep
