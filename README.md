# Can Adding Noise Help Us Select the Right Variables?

Code for the talk given at the J.B. Douglas Postgraduate Awards 2026 (Statistical Society of Australia, NSW Branch) by Mahdi Nouraie, Macquarie University.

The talk is based on:

> Nouraie, M., Zhu, H. and Muller, S. *2D Stability Selection: Design Jittering to Exclude Correlated Irrelevant Variables.* arXiv:2605.02205.

The talk presents the main idea of the paper with a single simulated example. The full theory, the simulation studies and the real-data analysis are in the paper.

This repository reproduces every simulated number and animation shown in the talk:

```bash
pip install -r requirements.txt
./run_all.sh
```

## Simulation setting

All result slides (8, 10 and 13) use one setting, the motivating example of the paper with three impostors:

- n = 100 observations and p = 1000 variables.
- Five relevant variables, independent standard normal, with coefficients β = λ·(4, 4, 4, 4, 15), where λ = √(log p / n₁) and n₁ = 50.
- Three impostors. Each is 0.4·(x₁ + x₂ + x₃ + x₄) − 0.447·x₅ plus independent noise, with unit variance and zero coefficient. (The paper calls these variables *confounders*, in the sense of correlation only.)
- The remaining 992 irrelevant variables are equicorrelated with correlation 0.1.
- Response noise ε ~ N(0, 1).
- Lasso at the fixed λ, columns re-standardised after adding noise, 200 half-samples per noise level, 18 noise levels δ in [0, 3].

The data set is generated with seed 6. `robustness_seeds.py` repeats the setting over seeds 1–12. The averages of the relevant variables and of the impostors separate as the noise grows in all twelve data sets.

## Slides and scripts

| Slide | Content | Script |
|---|---|---|
| 3 | Grid of 1,000 variables (schematic) | `animations/scenes.py: Grid` |
| 5 | Lasso path on a random half-sample of the main data (5 relevant and 35 irrelevant variables) | `simulations/main_setting.py`, `scenes.py: LassoPath` |
| 6 | Stability Selection, illustrative design (see below) | `simulations/illustration_ss.py`, `scenes.py: SSAnim` |
| 8 | Selection frequencies at δ = 0 for the 5 relevant variables and the 3 impostors | `simulations/main_setting.py` |
| 10 | Selection frequencies as δ increases from 0 to 0.75 | `main_setting.py`, `scenes.py: NoiseSweep` |
| 11 | Noise added to the design is a ridge penalty in expectation (Proposition 1 of the paper) | `scenes.py: Proof` |
| 12 | Population irrepresentable statistic c(ν) = (1.153 − 0.305ν²)/(1 + ν²) from the paper; the diagram is schematic | `scenes.py: Shortcut` |
| 13 | Average selection frequencies over δ in [0, 3] | `main_setting.py` |
| 14 | Results quoted from the paper (Studies 2 and 3) | — |

`simulations/check_slide_numbers.py` prints every simulated number that appears on the slides.

## Notes on the figures

- **Slide 6** illustrates how Stability Selection works. To show single Lasso fits disagreeing with each other, it uses a separate design in which the five relevant variables are correlated (correlation 0.8, β = 1.4λ, seed 3). `illustration_design.py` reports this behaviour for several correlation and signal levels. The animation steps through three of the 100 half-samples one at a time and then adds the rest. The grey bars are the five irrelevant variables selected most often.
- **Slide 8** shows the variables in a fixed shuffled order (A–H).
- **Slide 5** uses a square-root scale for the bar heights, so that the strong variable does not dominate the display.

## Software

Python 3.13, NumPy 2.5.3, scikit-learn 1.9.1 and manim 0.22.0 (animations only). The results are deterministic for these versions. Other versions of scikit-learn may differ in the last digits.
