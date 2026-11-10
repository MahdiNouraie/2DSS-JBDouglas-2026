import numpy as np
from manim import *
import os
SIM_DIR = os.environ.get("SIM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations", "output"))
from sklearn.linear_model import lasso_path, lars_path

BG="#14161A"; GREY="#4A4F57"; OR="#FF8C42"; TXT="#ECECEC"; MUT="#9AA0A6"; YEL="#FFD866"
config.background_color = BG
config.frame_height = 8.0
config.frame_width = 8.0 * config.pixel_width / config.pixel_height

class Grid(Scene):
    def construct(self):
        cols, rows, sp = 40, 25, 0.33
        pos = lambda c, r: np.array([(c-19.5)*sp, (12-r)*sp, 0])
        # Crux: RA (h), Dec (deg), magnitude.  x = -RA*cos(dec) (east to the left), y = Dec
        stars = {"Acrux":(12+26.6/60,-63.10,0.8), "Mimosa":(12+47.7/60,-59.69,1.25), "Gacrux":(12+31.2/60,-57.11,1.6),
                 "Delta":(12+15.1/60,-58.75,2.8), "Epsilon":(12+21.4/60,-60.40,3.6)}
        xy = {k:(-(v[0]-12.5)*15*np.cos(np.deg2rad(v[1])), v[1]) for k,v in stars.items()}
        xs = np.array([p[0] for p in xy.values()]); ys = np.array([p[1] for p in xy.values()])
        scale = 17 / (ys.max() - ys.min())                     # 17 rows tall
        cx, cy = (xs.max()+xs.min())/2, (ys.max()+ys.min())/2
        cell = {k:(int(round(19.5 + (p[0]-cx)*scale)), int(round(12 - (p[1]-cy)*scale))) for k,p in xy.items()}
        dots = VGroup(*[Dot(pos(c, r), radius=0.045, color=GREY) for r in range(rows) for c in range(cols)])
        idx = {k:r*cols + c for k,(c,r) in cell.items()}
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.0015, run_time=3))
        self.wait(0.5)
        order = ["Gacrux", "Acrux", "Mimosa", "Delta", "Epsilon"]
        for k in order:
            d = dots[idx[k]]; mag = stars[k][2]
            sc = 2.6 - 0.35*mag
            glow = Dot(d.get_center(), radius=0.09*sc + 0.08, color=OR, fill_opacity=0.25)
            self.play(d.animate.set_color(OR).scale(sc), FadeIn(glow, scale=0.3), run_time=0.6)
            self.wait(0.2)
        self.wait(1.5)


class LassoPath(Scene):
    def construct(self):
        D = np.load(os.path.join(SIM_DIR, "main_setting_data.npz"))
        r = np.random.default_rng(0); I = np.sort(r.choice(100, 50, replace=False)); oth = r.choice(np.arange(8, 1000), 35, replace=False)
        cols = [0, 1, 2, 3, 4] + oth.tolist()
        Xa = D["X"][np.ix_(I, cols)]; Xa = (Xa - Xa.mean(0)) / Xa.std(0); y = D["y"][I] - D["y"][I].mean()
        p = 40; relevant = [3, 11, 19, 27, 35]
        order = [None]*p; others = iter(range(5, 40)); ti = iter(range(5))
        for j in range(p): order[j] = next(ti) if j in relevant else next(others)
        X = Xa[:, order]; n = X.shape[0]
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        lam_max = np.abs(X.T @ y).max() / n
        grid = np.geomspace(lam_max, 1e-3, 400)
        _, cg, _ = lasso_path(X, y, alphas=grid)
        supg = [set(np.flatnonzero(np.abs(cg[:, i]) > 1e-10)) for i in range(len(grid))]
        lam_stop = max(grid[i] for i in range(len(grid)) if supg[i] == set(relevant)) * 0.8
        dense = np.geomspace(lam_stop, 1e-3, 3000)
        ka, _, kc = lars_path(X, y, method="lasso")          # exact piecewise-linear path
        dense = dense[::-1]                                  # ascending λ
        cd = np.array([np.interp(dense, ka[::-1], kc[j, ::-1]) for j in range(p)])
        noise = [j for j in range(p) if j not in relevant]
        l1 = np.abs(cd[noise]).sum(0); l1 = l1 / l1[0]
        logp = (np.log(dense) - np.log(dense[0])) / (np.log(dense[-1]) - np.log(dense[0]))
        prog = 0.7 * (1 - l1) + 0.3 * logp
        prog = np.convolve(np.pad(prog, 60, mode="edge"), np.ones(121) / 121, mode="valid")
        prog = np.maximum.accumulate(prog); prog = (prog - prog[0]) / (prog[-1] - prog[0])
        prog = prog + np.linspace(0, 1e-6, len(prog))
        pos = np.arange(len(dense), dtype=float)
        coefs = cd; lams = dense
        def at(u):
            x = float(np.interp(u, prog, pos)); i0 = int(np.floor(x)); i1 = min(i0 + 1, len(dense) - 1); f = x - i0
            return i0, i1, f
        W, base_y = 15.0, -1.6
        strong = relevant[-1]
        H = 3.9 / np.sqrt(np.abs(coefs).max())
        CAP = 99
        step = W / p; bw = step * 0.6
        axis = Line([-W/2 - 0.2, base_y, 0], [W/2 + 0.2, base_y, 0], color=MUT, stroke_width=2)
        t = ValueTracker(0)
        def val(j):
            i0, i1, f = at(t.get_value())
            return (1 - f) * coefs[j, i0] + f * coefs[j, i1]
        bars = VGroup()
        for j in range(p):
            col = OR if j in relevant else GREY
            def mk(j=j, col=col):
                v = val(j); h = min(max(np.sqrt(abs(v)) * H, 0.02), CAP)
                r = Rectangle(width=bw, height=h, fill_color=col, fill_opacity=1, stroke_width=0)
                x = -W/2 + step*(j + 0.5)
                r.move_to([x, base_y + np.sign(v if abs(v) > 1e-9 else 1) * h / 2, 0])
                return r
            bars.add(always_redraw(mk))
        lam_txt = always_redraw(lambda: Text(
            f"λ = {np.exp((1-at(t.get_value())[2])*np.log(lams[at(t.get_value())[0]]) + at(t.get_value())[2]*np.log(lams[at(t.get_value())[1]])):.2f}", font="Carlito", color=YEL, font_size=40
        ).to_corner(UL, buff=0.5).shift(RIGHT*7.2))
        kept = always_redraw(lambda: Text(
            f"variables in the model: {int(sum(abs(val(j)) > 1e-6 for j in range(p)))}", font="Carlito", color=TXT, font_size=34
        ).to_corner(UL, buff=0.5))
        xs_ = -W/2 + step*(strong + 0.5)
        self.add(axis, bars, lam_txt, kept)
        self.wait(0.4)
        self.play(t.animate.set_value(1), run_time=8, rate_func=linear)
        self.wait(2)


class NoiseSweep(Scene):
    def construct(self):
        import json
        d = json.load(open(os.path.join(SIM_DIR, "main_setting.json")))
        deltas = np.array(d["deltas"]); F = np.array(d["freq8"])
        BLUE = "#58C4DD"
        order = [4, 0, 1, 2, 3, 5, 6, 7]
        W, H, base_y = 11.0, 4.6, -2.6
        step = W / 9; bw = step * 0.6
        t = ValueTracker(0.0); dmax = 0.75
        fr = lambda j: float(np.interp(t.get_value() * dmax, deltas, F[:, j]))
        axis = Line([-W/2, base_y, 0], [W/2, base_y, 0], color=MUT, stroke_width=2)
        bars = VGroup()
        for k, j in enumerate(order):
            col = OR if j < 5 else BLUE
            x = -W/2 + step * (k + 0.5 + (1 if k >= 5 else 0))
            def mk(j=j, x=x, col=col):
                h = max(fr(j) * H, 0.02)
                return Rectangle(width=bw, height=h, fill_color=col, fill_opacity=1, stroke_width=0).move_to([x, base_y + h/2, 0])
            bars.add(always_redraw(mk))
        for v in (0.5, 1.0):
            ln = DashedLine([-W/2, base_y + v*H, 0], [W/2, base_y + v*H, 0], color=GREY, stroke_width=1.5, dash_length=0.12)
            lab = Text(f"{int(v*100)}%", font="Carlito", color=MUT, font_size=24).next_to(ln, LEFT, buff=0.15)
            self.add(ln, lab)
        lab_t = Text("relevant variables", font="Carlito", color=OR, font_size=30).move_to([-W/2 + step*2.5, base_y - 0.45, 0])
        lab_c = Text("impostors", font="Carlito", color=BLUE, font_size=30).move_to([-W/2 + step*7.5, base_y - 0.45, 0])
        ylab = Text("how often selected", font="Carlito", color=MUT, font_size=26).move_to([-W/2 + 1.3, base_y + H + 0.45, 0])
        dtxt = always_redraw(lambda: Text(f"noise  δ = {t.get_value()*dmax:.2f}", font="Carlito", color=YEL, font_size=40).to_corner(UR, buff=0.45))
        self.add(axis, bars, lab_t, lab_c, ylab, dtxt)
        self.wait(0.6)
        self.play(t.animate.set_value(1), run_time=10, rate_func=linear)
        self.wait(2)


class SSBase(Scene):
    NOISE = False
    def setup_data(self):
        import json
        J = json.load(open(os.path.join(SIM_DIR, "main_setting.json")))
        BLUE = "#58C4DD"
        if not self.NOISE:
            I = json.load(open(os.path.join(SIM_DIR, "illustration_ss.json")))
            self.vars = [("x₁",OR),("x₂",OR),("x₃",OR),("x₄",OR),("x₅",OR)] + [("",GREY)]*5
            R = [(r, s) for r, s in I["recs"]]
            first = [6, 27, 4]
            self.recs = [R[i] for i in first] + [R[i] for i in range(len(R)) if i not in first]
        else:
            self.vars = [("x₁",OR),("x₂",OR),("x₃",OR),("x₄",OR),("x₅",OR),("",BLUE),("",BLUE),("",BLUE)]
            self.recs = [(r, s8) for r, s8, _ in J["sel05"]]
    def construct(self):
        self.setup_data()
        nv = len(self.vars); B = len(self.recs)
        # data matrix (schematic): 20 display rows x 10 columns
        R, Cc, cs = 10, 20, 0.2
        mx, my = -5.2, 0.1
        cells = [[Square(cs*0.9, stroke_width=0, fill_color="#2A2F37", fill_opacity=1).move_to([mx + (c-(Cc-1)/2)*cs, my + ((R-1)/2-r)*cs, 0]) for c in range(Cc)] for r in range(R)]
        mat = VGroup(*[sq for row in cells for sq in row])
        mlab = VGroup(Text("data: 100 observations", font="Carlito", font_size=24, color=MUT),
                      Text("× 1,000 variables", font="Carlito", font_size=24, color=MUT)).arrange(DOWN, buff=0.1).next_to(mat, DOWN, buff=0.25)
        lasso = RoundedRectangle(width=1.7, height=0.9, corner_radius=0.15, color=TXT, stroke_width=2).move_to([-2.0, my, 0])
        ltxt = Text("Lasso", font="Carlito", font_size=30, color=TXT).move_to(lasso)
        a1 = Arrow(mat.get_right(), lasso.get_left(), buff=0.15, color=MUT, stroke_width=3)
        # variables, bars, indicators
        x0, x1 = 0.4, 6.6; base = -2.3; H = 3.8
        xs = [x0 + (x1-x0)*k/(nv-1) for k in range(nv)]
        axis = Line([x0-0.4, base, 0], [x1+0.4, base, 0], color=MUT, stroke_width=2)
        ind = [Square(0.32, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0).move_to([x, base-0.4, 0]) for x, (_, col) in zip(xs, self.vars)]
        labs = VGroup(*[Text(l, font="Carlito", font_size=24, color=col).move_to([x, base-0.85, 0]) for x, (l, col) in zip(xs, self.vars) if l])
        gx = [x for x, (l, c) in zip(xs, self.vars) if not l]
        if gx:
            gcol = [c for (l, c) in self.vars if not l][0]
            labs.add(Text("impostors" if self.NOISE else "irrelevant variables", font="Carlito", font_size=24, color=gcol).move_to([(gx[0]+gx[-1])/2, base-0.85, 0]))
        a2 = Arrow(lasso.get_right(), [x0-0.6, base-0.4, 0], buff=0.15, color=MUT, stroke_width=3)
        cnt = np.zeros(nv); bt = ValueTracker(0)
        hist = np.zeros((B+1, nv))
        for b, (_, sel) in enumerate(self.recs):
            hist[b+1] = hist[b]; hist[b+1, sel] += 1
        def freq_at(v):
            b = int(np.clip(np.floor(v), 0, B)); f = v - b
            b2 = min(b+1, B)
            num = (1-f)*hist[b] + f*hist[b2]; den = max(v, 1e-9)
            return num/den if v >= 1 else np.zeros(nv)
        bars = VGroup()
        for k, (x, (_, col)) in enumerate(zip(xs, self.vars)):
            bars.add(always_redraw(lambda k=k, x=x, col=col: Rectangle(width=0.42, height=max(freq_at(bt.get_value())[k]*H, 0.01), fill_color=col, fill_opacity=1, stroke_width=0).move_to([x, base + max(freq_at(bt.get_value())[k]*H, 0.01)/2, 0])))
        ylab = Text("how often selected", font="Carlito", font_size=24, color=MUT).move_to([x0+0.9, base+H+0.35, 0])
        counter = always_redraw(lambda: Text(f"resample {int(round(bt.get_value()))} / {B}", font="Carlito", font_size=30, color=YEL).to_corner(UR, buff=0.45))
        cap = Text("Stability Selection", font="Carlito", font_size=32, color=TXT).to_corner(UL, buff=0.45).set_opacity(0)
        self.add(mat, mlab, lasso, ltxt, a1, a2, axis, *ind, labs, bars, ylab, counter, cap)
        def row_set(rows):
            hit = np.zeros(R)
            for i in rows: hit[min(i//10, R-1)] += 1
            return sorted(np.argsort(-hit, kind="stable")[:R//2].tolist())
        steps = [("1. take a random half of the observations", None), ("2. add noise to the data (δ = 0.5)", "noise"), ("3. fit the Lasso, record what is selected", None)]
        if not self.NOISE: steps = [s for s in steps if s[1] is None]; steps[1] = ("2. fit the Lasso, record what is selected", None)
        for b in range(3):
            rows, sel = self.recs[b]; hr = row_set(rows)
            on = [cells[r][c] for r in hr for c in range(Cc)]
            capb = Text(steps[0][0], font="Carlito", font_size=30, color=TXT).to_corner(UL, buff=0.45).set_opacity(0)
            self.play(FadeOut(cap), FadeIn(capb), *[sq.animate.set_fill("#6B717B") for sq in on], run_time=0.6 if b else 0.9)
            cap = capb
            if self.NOISE:
                capn = Text(steps[1][0], font="Carlito", font_size=30, color=YEL).to_corner(UL, buff=0.45).set_opacity(0)
                rng = np.random.default_rng(b)
                self.play(FadeOut(cap), FadeIn(capn), *[sq.animate.set_fill(interpolate_color(ManimColor("#6B717B"), ManimColor(YEL), rng.uniform(0.2, 0.8))) for sq in on], run_time=0.7)
                cap = capn
            capl = Text(steps[-1][0], font="Carlito", font_size=30, color=TXT).to_corner(UL, buff=0.45).set_opacity(0)
            self.play(FadeOut(cap), FadeIn(capl), Indicate(lasso, color=YEL, scale_factor=1.08), run_time=0.6)
            cap = capl
            self.play(*[ind[k].animate.set_fill(opacity=1 if k in sel else 0) for k in range(nv)], bt.animate.set_value(b+1), run_time=0.6)
            self.wait(0.6)
            self.next_section(f"after_r{b+1}")
            self.play(*[sq.animate.set_fill("#2A2F37") for sq in on], *[i.animate.set_fill(opacity=0) for i in ind], run_time=0.3)
        capr = Text("4. repeat 100 times", font="Carlito", font_size=30, color=TXT).to_corner(UL, buff=0.45).set_opacity(0)
        self.play(FadeOut(cap), FadeIn(capr), run_time=0.4); cap = capr
        self.play(bt.animate.set_value(B), run_time=4, rate_func=linear)
        if self.NOISE:
            capk = Text("5. repeat for a grid of noise levels δ", font="Carlito", font_size=30, color=YEL).to_corner(UL, buff=0.45).set_opacity(0)
            self.play(FadeOut(cap), FadeIn(capk), run_time=0.8)
        else:
            thr = DashedLine([x0-0.4, base+0.6*H, 0], [x1+0.4, base+0.6*H, 0], color=YEL, stroke_width=2.5)
            capk = Text("5. keep the variables selected most of the time", font="Carlito", font_size=30, color=YEL).to_corner(UL, buff=0.45).set_opacity(0)
            self.play(Create(thr), FadeOut(cap), FadeIn(capk), run_time=1.0)
        self.wait(2.5)


class SSAnim(SSBase):
    NOISE = False


class ShakeAnim(SSBase):
    NOISE = True


class Proof(Scene):
    def construct(self):
        tp = TexTemplate(); tp.add_to_preamble(r"\usepackage{amsmath,amssymb,bm}")
        def T(*a, **k): return MathTex(*a, tex_template=tp, **k)
        l1 = T(r"\tfrac{1}{2n}\,\big\|\,y-(X+W)\beta\,\big\|^2", r"+\lambda\|\beta\|_1", font_size=54).shift(UP*2.75)
        cap1 = Text("Lasso on the shaken data", font="Carlito", font_size=30, color=MUT).next_to(l1, UP, buff=0.35)
        self.play(FadeIn(cap1), Write(l1), run_time=2); self.wait(2)
        l2 = T(r"\tfrac{1}{2n}\,\big\|\,(y-X\beta)", r"-W\beta", r"\,\big\|^2", r"+\lambda\|\beta\|_1", font_size=54).shift(UP*1.25)
        l2[1].set_color(YEL)
        self.play(TransformFromCopy(l1, l2), run_time=1.5); self.wait(1.5)
        n1 = T(r"\mathbb{E}[W]=0", r"\;\Rightarrow\;\text{cross term vanishes}", font_size=36, color=MUT).shift(DOWN*0.05+LEFT*2.6)
        n2 = T(r"\mathbb{E}\|W\beta\|^2 = n\,\delta^2\|\beta\|^2", font_size=36, color=YEL).shift(DOWN*0.05+RIGHT*3.4)
        self.play(FadeIn(n1, shift=UP*0.2)); self.wait(1.2)
        self.play(FadeIn(n2, shift=UP*0.2)); self.wait(1.8)
        l3 = T(r"\tfrac{1}{2n}\|y-X\beta\|^2", r"+\lambda\|\beta\|_1", r"+\tfrac{\delta^2}{2}\|\beta\|^2", font_size=58).shift(DOWN*1.45)
        ref = l3.copy()
        l3[2].set_color(YEL)
        eq = T(r"\mathbb{E}\,[\;\cdot\;]\;=", font_size=46, color=MUT).next_to(l3, LEFT, buff=0.3)
        self.play(FadeIn(eq), TransformFromCopy(l2, l3), run_time=1.8); self.wait(1.2)
        b1 = Brace(VGroup(ref[0], ref[1]), DOWN, color=MUT); t1 = Text("Lasso", font="Carlito", font_size=32, color=TXT).next_to(b1, DOWN, buff=0.15)
        b2 = Brace(ref[2], DOWN, color=YEL); t2 = Text("ridge", font="Carlito", font_size=32, color=YEL).next_to(b2, DOWN, buff=0.15)
        self.play(GrowFromCenter(b1), FadeIn(t1), GrowFromCenter(b2), FadeIn(t2)); self.wait(1.2)
        box = SurroundingRectangle(VGroup(ref, b1, b2, t1, t2), color=OR, buff=0.2, corner_radius=0.1)
        en = Text("= an elastic net", font="Carlito", font_size=40, color=OR).next_to(box, DOWN, buff=0.25)
        self.play(Create(box), Write(en), run_time=1.5); self.wait(3)


class CnuCurve(Scene):
    def construct(self):
        BLUE = "#58C4DD"
        ax = Axes(x_range=[0, 1.5, 0.25], y_range=[0, 1.4, 0.5], x_length=11, y_length=5.2,
                  axis_config={"color": MUT, "stroke_width": 2, "include_ticks": True},
                  x_axis_config={"numbers_to_include": [0, 0.5, 1, 1.5], "font_size": 28, "decimal_number_config": {"num_decimal_places": 1, "color": MUT}},
                  y_axis_config={"numbers_to_include": [0.5, 1], "font_size": 28, "decimal_number_config": {"num_decimal_places": 1, "color": MUT}},
                  tips=False).shift(DOWN*0.4)
        f = lambda v: abs(1.153 - 0.3*v*v) / (1 + v*v)
        one = DashedLine(ax.c2p(0, 1), ax.c2p(1.5, 1), color=TXT, stroke_width=2)
        lab1 = Text("impostor gets selected", font="Carlito", font_size=26, color=BLUE).next_to(ax.c2p(1.5, 1), UP+LEFT, buff=0.15).shift(LEFT*0.2)
        lab2 = Text("impostor is rejected", font="Carlito", font_size=26, color=OR).next_to(ax.c2p(1.5, 1), DOWN+LEFT, buff=0.15).shift(LEFT*0.2)
        xl = Text("noise level", font="Carlito", font_size=28, color=MUT).next_to(ax.x_axis, DOWN, buff=0.55)
        yl = Text("irrepresentable statistic", font="Carlito", font_size=28, color=MUT).next_to(ax, UP, buff=0.15).align_to(ax, LEFT)
        self.add(ax, one, lab1, lab2, xl, yl)
        t = ValueTracker(0.0)
        curve = always_redraw(lambda: ax.plot(f, x_range=[0, max(t.get_value(), 1e-3)], color=YEL, stroke_width=5))
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), f(t.get_value())), radius=0.11, color=OR if f(t.get_value()) < 1 else BLUE))
        rd = always_redraw(lambda: Text(f"δ = {t.get_value():.2f}", font="Carlito", font_size=34, color=YEL).to_corner(UR, buff=0.45))
        self.add(curve, dot, rd)
        self.wait(0.5)
        self.play(t.animate.set_value(0.34), run_time=3, rate_func=linear)
        cross = Circle(radius=0.28, color=OR, stroke_width=4).move_to(ax.c2p(0.34, 1))
        note = Text("crosses 1 at δ ≈ 0.34", font="Carlito", font_size=28, color=OR).next_to(cross, DOWN+RIGHT, buff=0.1)
        self.play(Create(cross), FadeIn(note)); self.wait(1)
        self.play(t.animate.set_value(1.5), run_time=5, rate_func=linear)
        self.wait(2)


class Shortcut(Scene):
    PART = 1
    def construct(self):
        BLUE = "#58C4DD"
        f = lambda v: abs(1.153 - 0.3*v*v) / (1 + v*v)
        t = ValueTracker(0.0)
        # ---- left: the shortcut diagram
        Y = np.array([-1.6, 0.3, 0])
        ynode = Circle(radius=0.42, color=TXT, stroke_width=3).move_to(Y)
        ylab = MathTex("y", color=TXT, font_size=48).move_to(Y)
        xs = [np.array([-6.6, 2.1 - 1.2*i, 0]) for i in range(4)]
        tnodes = VGroup(*[Dot(p, radius=0.17, color=OR) for p in xs])
        tlabs = VGroup(*[MathTex(f"x_{i+1}", color=OR, font_size=38).next_to(p, LEFT, buff=0.2) for i, p in enumerate(xs)])
        tarrows = VGroup(*[Arrow(p, Y, buff=0.48, color=OR, stroke_width=4, max_tip_length_to_length_ratio=0.06) for p in xs])
        I = np.array([-4.6, -3.0, 0])
        inode = Dot(I, radius=0.19, color=BLUE)
        ilab = Text("impostor", font="Carlito", font_size=30, color=BLUE).next_to(I, DOWN, buff=0.18)
        mixes = VGroup(*[DashedLine(p, I, color=BLUE, stroke_width=2, dash_length=0.1) for p in xs])
        iarrow = Arrow(I, Y, buff=0.45, color=BLUE, stroke_width=7, max_tip_length_to_length_ratio=0.08)
        if self.PART == 1:
            self.play(FadeIn(ynode), FadeIn(ylab), LaggedStart(*[AnimationGroup(FadeIn(n), FadeIn(l), GrowArrow(a)) for n, l, a in zip(tnodes, tlabs, tarrows)], lag_ratio=0.25), run_time=2.2)
            self.wait(1.2)
            self.play(LaggedStart(*[Create(m) for m in mixes], lag_ratio=0.15), FadeIn(inode), FadeIn(ilab), run_time=1.8)
            self.play(GrowArrow(iarrow), run_time=1.0)
            self.wait(2.2)
            return
        self.add(ynode, ylab, tnodes, tlabs, tarrows, mixes, inode, ilab, iarrow)
        # ---- right: the irrepresentable statistic
        ax = Axes(x_range=[0, 1.5, 0.5], y_range=[0, 1.3, 0.5], x_length=5.0, y_length=3.6,
                  axis_config={"color": MUT, "stroke_width": 2},
                  x_axis_config={"numbers_to_include": [0, 0.5, 1, 1.5], "font_size": 24, "decimal_number_config": {"num_decimal_places": 1, "color": MUT}},
                  y_axis_config={"numbers_to_include": [0.5, 1], "font_size": 24, "decimal_number_config": {"num_decimal_places": 1, "color": MUT}},
                  tips=False).move_to([3.85, -0.6, 0])
        one = DashedLine(ax.c2p(0, 1), ax.c2p(1.5, 1), color=TXT, stroke_width=2)
        la = Text("impostor selected", font="Carlito", font_size=22, color=BLUE).next_to(ax.c2p(1.5, 1), UP, buff=0.1).shift(LEFT*1.1)
        lb = Text("impostor rejected", font="Carlito", font_size=22, color=OR).next_to(ax.c2p(1.5, 1), DOWN, buff=0.1).shift(LEFT*1.1)
        xl = Text("noise level δ", font="Carlito", font_size=24, color=MUT).next_to(ax.x_axis, DOWN, buff=0.5)
        yl = Text("irrepresentable statistic", font="Carlito", font_size=24, color=MUT).next_to(ax, UP, buff=0.12).align_to(ax, LEFT)
        curve = always_redraw(lambda: ax.plot(f, x_range=[0, max(t.get_value(), 1e-3)], color=YEL, stroke_width=5))
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), f(t.get_value())), radius=0.1, color=OR if f(t.get_value()) < 1 else BLUE))
        rd = always_redraw(lambda: Text(f"δ = {t.get_value():.2f}", font="Carlito", font_size=34, color=YEL).to_corner(UR, buff=0.45))
        self.play(FadeIn(ax), FadeIn(one), FadeIn(la), FadeIn(lb), FadeIn(xl), FadeIn(yl), run_time=1.2)
        self.add(curve, dot, rd)
        def op():
            c = f(t.get_value())
            return float(np.clip((c - 0.82) / 0.18, 0.08, 1.0))
        iarrow.add_updater(lambda m: m.set_opacity(op()))
        for m in mixes: m.add_updater(lambda mm: mm.set_stroke(opacity=op()))
        inode.add_updater(lambda m: m.set_opacity(op()))
        ilab.add_updater(lambda m: m.set_opacity(op()))
        self.play(t.animate.set_value(0.34), run_time=3.2, rate_func=linear)
        cross = Circle(radius=0.22, color=OR, stroke_width=4).move_to(ax.c2p(0.34, 1))
        self.play(Create(cross), run_time=0.6)
        self.play(t.animate.set_value(1.5), run_time=5, rate_func=linear)
        self.wait(2)


class ShortcutB(Shortcut):
    PART = 2


class LassoFormula(Scene):
    def construct(self):
        tp = TexTemplate(); tp.add_to_preamble(r"\usepackage{amsmath,amssymb}")
        f = MathTex(r"\min_{\beta}\;", r"\tfrac{1}{2n}\,\|y - X\beta\|_2^2", r"\;+\;", r"\lambda\|\beta\|_1", tex_template=tp, font_size=64)
        f[3].set_color(YEL)
        b1 = Brace(f[1], DOWN, color=TXT); t1 = Text("how well we fit the data", font="Carlito", font_size=30, color=TXT).next_to(b1, DOWN, buff=0.12)
        b2 = Brace(f[3], DOWN, color=YEL); t2 = Text("a price for every coefficient", font="Carlito", font_size=30, color=YEL).next_to(b2, DOWN, buff=0.12)
        t1.next_to(b1, DOWN, buff=0.12)
        t2.next_to(b2, DOWN, buff=0.12)
        if t2.get_left()[0] < t1.get_right()[0] + 0.5:
            t2.shift(RIGHT * (t1.get_right()[0] + 0.5 - t2.get_left()[0]))
        g = VGroup(f, b1, t1, b2, t2)
        g.scale_to_fit_width(config.frame_width * 0.96).move_to(ORIGIN)
        self.add(g)
