
import pygame
import numpy as np
import math
import random
import sys
import pickle
import os
from collections import defaultdict, deque

Q_SAVE_PATH = "q_table.pkl"

def save_q_table(Q, path=Q_SAVE_PATH):
    data = {k: v.tolist() for k, v in Q.items()}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVE] Q-table kaydedildi: {path}  ({len(data)} state)")

def load_q_table(path=Q_SAVE_PATH):
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    if not os.path.exists(path):
        print(f"[LOAD] Kayıt dosyası bulunamadı: {path}")
        return Q
    with open(path, "rb") as f:
        data = pickle.load(f)
    for k, v in data.items():
        Q[tuple(k)] = np.array(v)
    print(f"[LOAD] Q-table yüklendi: {path}  ({len(data)} state)")
    return Q
SW, SH      = 1400, 900
FPS         = 60
ROAD_W      = 100
NUM_AGENTS  = 8
SPLINE_RES  = 18

MAX_SPD  = 3.2
ACCEL    = 0.13
BRAKE_F  = 0.22
FRICTION = 0.035
STEER    = 3.8

ALPHA     = 0.10
GAMMA     = 0.95
EPS_INIT  = 1.0
EPS_MIN   = 0.05
EPS_DECAY = 0.99999

NUM_ACTIONS = 5

TL_RED_DUR = 160
TL_GRN_DUR = 130

BG_COL      = (45,  90,  35)
ROAD_COL    = (68,  68,  68)
EDGE_COL    = (210, 210, 210)
DASH_COL    = (245, 215,  50)
POLE_COL    = (80,  80,  80)
TL_BOX_COL  = (30,  30,  30)
POT_COL     = (42,  28,  10)
ACC_COL     = (185, 68,  12)
BEST_COL    = (255, 215,   0)
TXT_COL     = (235, 235, 235)
PNL_COL     = (15,  15,  25)
START_COL   = (20,  200,  90)
FINISH_COL  = (200,  40,  40)

AGENT_COLORS = [
    ( 60, 130, 255),
    (255, 100,  60),
    ( 60, 200,  60),
    (200,  60, 200),
    ( 60, 200, 200),
    (200, 200,  60),
    (200,  60,  60),
    (180, 100, 255),
]

CTRL_PTS = np.array([
    [  70, 840], [155, 812], [252, 782], [342, 722],
    [392, 642], [376, 560], [306, 498], [256, 420],
    [270, 340], [340, 274], [436, 250], [540, 270],
    [620, 334], [666, 424], [696, 514], [776, 570],
    [876, 560], [970, 510], [1040, 434], [1076, 344],
    [1090, 254],[1146, 178],[1246, 150],[1330, 190],
    [1376, 280],[1386, 378],
], dtype=float)


TL_FRACS  = [0.22, 0.50, 0.74]
OBS_FRACS = [0.7]
OBS_TYPES = ["pothole", "accident", "pothole", "accident"]

def _cr(p0, p1, p2, p3, t):
    """Single Catmull-Rom interpolation step."""
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t**2
        + (-p0 + 3*p1 - 3*p2 + p3) * t**3
    )


class Road:
    def __init__(self):
        pts = []
        n = len(CTRL_PTS)
        for i in range(1, n - 2):
            p0 = CTRL_PTS[max(i - 1, 0)]
            p1 = CTRL_PTS[i]
            p2 = CTRL_PTS[i + 1]
            p3 = CTRL_PTS[min(i + 2, n - 1)]
            for j in range(SPLINE_RES):
                pts.append(_cr(p0, p1, p2, p3, j / SPLINE_RES))
        pts.append(CTRL_PTS[-2])
        self.path = np.array(pts)
        N = len(self.path)

        
        tang = np.zeros((N, 2))
        tang[:-1] = self.path[1:] - self.path[:-1]
        tang[-1]  = tang[-2]
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        self.tang = tang

        
        self.perp = np.column_stack([-tang[:, 1], tang[:, 0]])

        seg_lens  = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.cum  = np.concatenate([[0], np.cumsum(seg_lens)])
        self.total = float(self.cum[-1])

        
        h = ROAD_W / 2
        self.le = self.path + self.perp * h
        self.re = self.path - self.perp * h
        
        
        le_ipts = [(int(x), int(y)) for x, y in self.le]
        re_ipts = [(int(x), int(y)) for x, y in self.re]
        self.poly    = le_ipts + re_ipts[::-1]
        self.le_poly = le_ipts
        self.re_poly = re_ipts
        self.ctr_pts = [(int(x), int(y)) for x, y in self.path]

    
    def nearest(self, pos):
        d2 = np.sum((self.path - np.asarray(pos)) ** 2, axis=1)
        return int(np.argmin(d2))

    def lat_off(self, pos, idx):
        return float(np.dot(np.asarray(pos) - self.path[idx], self.perp[idx]))

    def head_err(self, ang_deg, idx):
        road_ang = math.degrees(math.atan2(self.tang[idx, 1], self.tang[idx, 0]))
        err = ang_deg - road_ang
        while err >  180: err -= 360
        while err < -180: err += 360
        return err

    def progress(self, idx):
        return self.cum[idx] / self.total

    def pt_at_frac(self, frac):
        target = float(np.clip(frac, 0, 1)) * self.total
        idx = int(np.clip(np.searchsorted(self.cum, target), 0, len(self.path) - 1))
        return self.path[idx]

    def draw(self, surf):
        pygame.draw.polygon(surf, ROAD_COL, self.poly)
        pygame.draw.lines(surf, EDGE_COL, False, self.le_poly, 2)
        pygame.draw.lines(surf, EDGE_COL, False, self.re_poly, 2)

        cpts = self.ctr_pts
        for i in range(0, len(cpts) - 2, 4):
            pygame.draw.line(surf, DASH_COL, cpts[i], cpts[i + 1], 2)

        self._banner(surf, self.path[4],   self.perp[4],   START_COL,  "A  START")
        self._banner(surf, self.path[-4],  self.perp[-4],  FINISH_COL, "B  FINISH")

    def _banner(self, surf, pt, perp, col, label):
        p  = np.asarray(pt)
        h  = ROAD_W / 2 + 10
        p1 = (int(p[0] + perp[0] * h), int(p[1] + perp[1] * h))
        p2 = (int(p[0] - perp[0] * h), int(p[1] - perp[1] * h))
        pygame.draw.line(surf, col, p1, p2, 6)
        fnt = pygame.font.SysFont("arial", 18, bold=True)
        txt = fnt.render(label, True, col)
        surf.blit(txt, (int(p[0]) - 14, int(p[1]) - 34))


class TrafficLight:
    def __init__(self, road: Road, frac: float, phase_offset: int = 0):
        n   = len(road.path)
        self.idx  = int(np.clip(frac * (n - 1), 0, n - 1))
        self.pos  = road.path[self.idx].copy()
        self.perp = road.perp[self.idx].copy()
        self.t    = phase_offset
        self._refresh()

    def _refresh(self):
        self.red = (self.t % (TL_RED_DUR + TL_GRN_DUR)) < TL_RED_DUR

    def update(self):
        self.t += 1
        self._refresh()

    def draw(self, surf):
        base = self.pos + self.perp * (ROAD_W / 2 + 16)
        bx, by = int(base[0]), int(base[1])

        pygame.draw.line(surf, POLE_COL, (bx, by), (bx, by - 48), 5)

        box = pygame.Rect(bx - 14, by - 76, 28, 48)
        pygame.draw.rect(surf, TL_BOX_COL, box, border_radius=5)

        r_col = (255,  30,  30) if self.red      else ( 80, 10, 10)
        g_col = ( 30, 225,  30) if not self.red  else ( 10, 60, 10)
        pygame.draw.circle(surf, r_col, (bx, by - 66), 10)
        pygame.draw.circle(surf, g_col, (bx, by - 47), 10)

        if self.red:
            pygame.draw.circle(surf, (255, 80, 80), (bx, by - 66), 13, 2)
        else:
            pygame.draw.circle(surf, (80, 255, 80), (bx, by - 47), 13, 2)


class Obstacle:
    def __init__(self, road: Road, frac: float, kind: str):
        n = len(road.path)
        self.idx  = int(np.clip(frac * (n - 1), 0, n - 1))
        self.pos  = road.path[self.idx].copy()
        self.kind = kind
        self.r    = 18 if kind == "pothole" else 24

    def draw(self, surf):
        x, y = int(self.pos[0]), int(self.pos[1])
        r = self.r
        if self.kind == "pothole":
            pygame.draw.ellipse(surf, POT_COL,
                                (x - r, y - int(r * 0.60), r * 2, int(r * 1.2)))
            pygame.draw.ellipse(surf, (28, 18, 5),
                                (x - r + 2, y - int(r * 0.50), r * 2 - 4, int(r)), 2)
            crack_ends = [(6, 3), (-7, 2), (3, -6), (-4, 5)]
            for dx, dy in crack_ends:
                pygame.draw.line(surf, (25, 15, 5),
                                 (x, y), (x + dx * 2, y + dy * 2), 2)
        else:
            pygame.draw.circle(surf, ACC_COL, (x, y), r)
            for angle in (40, 130):
                rad = math.radians(angle)
                x1 = int(x + math.cos(rad) * (r - 4))
                y1 = int(y + math.sin(rad) * (r - 4))
                x2 = int(x - math.cos(rad) * (r - 4))
                y2 = int(y - math.sin(rad) * (r - 4))
                pygame.draw.line(surf, (230, 200, 0), (x1, y1), (x2, y2), 4)
            fnt = pygame.font.SysFont("arial", 16, bold=True)
            surf.blit(fnt.render("!", True, (255, 255, 255)), (x - 4, y - 10))


P_BINS = 30
L_BINS = 5
A_BINS = 5
S_BINS = 4

def discretise(prog, lat, head, spd, tl_code, ob_code):
    pb = int(np.clip(prog * P_BINS,                        0, P_BINS - 1))
    lb = int(np.clip((lat / (ROAD_W / 2) + 1) / 2 * L_BINS, 0, L_BINS - 1))
    ab = int(np.clip((head / 90 + 1) / 2 * A_BINS,          0, A_BINS - 1))
    sb = int(np.clip(spd / MAX_SPD * S_BINS,                 0, S_BINS - 1))
    return (pb, lb, ab, sb, tl_code, ob_code)


class Car:
    W, H = 20, 11

    def __init__(self, cid, road, tls, obstacles, q_table, eps_ref):
        self.id        = cid
        self.road      = road
        self.tls       = tls
        self.obstacles = obstacles
        self.Q         = q_table
        self.eps       = eps_ref
        self.color     = AGENT_COLORS[cid % len(AGENT_COLORS)]
        self.is_best   = False
        self.trail     = deque(maxlen=35)

        
        self.ep_reward   = 0.0
        self.best_rew    = -1e9
        self.total_steps = 0
        self.episodes    = 0

        self.reset()

    def reset(self):
        si         = 5
        self.idx   = si
        self.pos   = self.road.path[si].copy().astype(float)
        tang_ang   = math.degrees(math.atan2(
            self.road.tang[si, 1], self.road.tang[si, 0]))
        self.angle = tang_ang
        self.speed = 0.4 + random.uniform(0.0, 0.3)

        
        self.pos  += self.road.perp[si] * random.uniform(-16, 16)
        self.ep_reward = 0.0
        self.steps_ep  = 0
        self.trail.clear()
        self.episodes += 1

    
    def _observe(self):
        idx  = self.road.nearest(self.pos)
        self.idx = idx
        prog = self.road.progress(idx)
        lat  = self.road.lat_off(self.pos, idx)
        head = self.road.head_err(self.angle, idx)
        spd  = self.speed

        tl_code = 0
        for tl in self.tls:
            ahead = self.road.cum[tl.idx] - self.road.cum[idx]
            if 0 < ahead < 130:
                tl_code = 2 if tl.red else 1
                break

        ob_code = 0
        for ob in self.obstacles:
            d = float(np.linalg.norm(self.pos - ob.pos))
            if d < 85:
                ob_code = 2 if d < 38 else 1
                break

        return discretise(prog, lat, head, spd, tl_code, ob_code)

    def _apply(self, action):
        if   action == 0: self.angle -= STEER
        elif action == 1: self.angle += STEER
        elif action == 2: self.speed  = min(self.speed + ACCEL,  MAX_SPD)
        elif action == 3: self.speed  = max(self.speed - BRAKE_F, 0.0)
    
        self.speed = max(self.speed - FRICTION, 0.30)

        rad = math.radians(self.angle)
        self.pos[0] += math.cos(rad) * self.speed
        self.pos[1] += math.sin(rad) * self.speed

    def _reward(self, old_idx, new_idx, s2):
        r    = 0.0
        done = False

        r += (self.road.cum[new_idx] - self.road.cum[old_idx]) * 0.09

        r -= 0.04

        lat = self.road.lat_off(self.pos, new_idx)
        if abs(lat) > ROAD_W / 2 * 0.88:
            return -12.0, True
        r -= abs(lat) / ROAD_W * 0.25

        if s2[4] == 2 and self.speed > 0.65:
            r -= 2.0

        for ob in self.obstacles:
            d = float(np.linalg.norm(self.pos - ob.pos))
            if d < 28:
                return -8.0, True
            if d < 55:
                r -= 0.5

        if self.road.progress(new_idx) > 0.97:
            return r + 100.0, True

        return r, False

    def step(self):
        s       = self._observe()
        old_idx = self.idx
        eps     = self.eps[0]

        action = (random.randrange(NUM_ACTIONS)
                  if random.random() < eps
                  else int(np.argmax(self.Q[s])))

        self._apply(action)
        self.trail.append((int(self.pos[0]), int(self.pos[1])))

        new_idx = self.road.nearest(self.pos)
        self.idx = new_idx
        s2 = self._observe()

        reward, done = self._reward(old_idx, new_idx, s2)

        future = 0.0 if done else GAMMA * float(np.max(self.Q[s2]))
        self.Q[s][action] += ALPHA * (reward + future - self.Q[s][action])

        self.ep_reward   += reward
        self.steps_ep    += 1
        self.total_steps += 1

        if done or self.steps_ep > 2500:
            if self.ep_reward > self.best_rew:
                self.best_rew = self.ep_reward
            self.reset()

    def draw(self, surf):
        col = BEST_COL if self.is_best else self.color

        # Trail (best agent only, fading gold)
        if self.is_best:
            trail = list(self.trail)
            n = len(trail)
            for i in range(n - 1):
                frac = (i + 1) / max(n, 1)
                dim  = int(frac * 200)
                c    = (min(255, BEST_COL[0]),
                        min(215, int(BEST_COL[1] * frac * 0.8)),
                        0)
                try:
                    pygame.draw.line(surf, c, trail[i], trail[i + 1], 2)
                except Exception:
                    pass

        rad = math.radians(self.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        hw, hh = self.W / 2, self.H / 2
        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        poly = [
            (int(self.pos[0] + cx * ca - cy * sa),
             int(self.pos[1] + cx * sa + cy * ca))
            for cx, cy in corners_local
        ]
        pygame.draw.polygon(surf, col, poly)

        outline = (220, 175, 0) if self.is_best else (180, 180, 180)
        pygame.draw.polygon(surf, outline, poly, 2)

        hx = int(self.pos[0] + ca * hw)
        hy = int(self.pos[1] + sa * hw)
        pygame.draw.circle(surf, (255, 255, 180), (hx, hy), 3)

        if self.is_best:
            self._draw_crown(surf)

    def _draw_crown(self, surf):
        cx = int(self.pos[0])
        cy = int(self.pos[1]) - 32
        star_pts = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r     = 11 if i % 2 == 0 else 5
            star_pts.append((cx + r * math.cos(angle),
                             cy + r * math.sin(angle)))
        pygame.draw.polygon(surf, BEST_COL, star_pts)
        pygame.draw.polygon(surf, (180, 140, 0), star_pts, 1)


def draw_hud(surf, agents, eps, frame, sim_speed,
             font_sm, font_md, font_lg):
    pw, ph = 310, 310
    panel  = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((*PNL_COL, 215))
    surf.blit(panel, (6, 6))

    y = 12
    surf.blit(font_lg.render("RL Car Simulation", True, TXT_COL), (14, y))
    y += 30
    pygame.draw.line(surf, (80, 80, 110), (10, y), (pw + 4, y), 1)
    y += 7

    stats = [
        f"Frame      : {frame:>9,}",
        f"Epsilon    : {eps:>9.5f}",
        f"Q-States   : {len(agents[0].Q):>9,}",
        f"Sim Speed  : {sim_speed:>8}x",
    ]
    for line in stats:
        surf.blit(font_sm.render(line, True, TXT_COL), (14, y))
        y += 17

    y += 5
    pygame.draw.line(surf, (80, 80, 110), (10, y), (pw + 4, y), 1)
    y += 6
    surf.blit(font_md.render(" #  Ep   Best Rew    Steps", True, (200, 200, 90)),
              (14, y))
    y += 18

    for a in sorted(agents, key=lambda x: x.best_rew, reverse=True):
        col    = BEST_COL if a.is_best else AGENT_COLORS[a.id % len(AGENT_COLORS)]
        prefix = "* " if a.is_best else "  "
        line   = f"{prefix}#{a.id} {a.episodes:>4d}  {a.best_rew:>9.1f}  {a.total_steps:>6d}"
        surf.blit(font_sm.render(line, True, col), (14, y))
        y += 16

    best = next((a for a in agents if a.is_best), None)
    if best:
        bx, by = 6, SH - 58
        bp = pygame.Surface((210, 50), pygame.SRCALPHA)
        bp.fill((*PNL_COL, 205))
        surf.blit(bp, (bx - 2, by - 2))
        surf.blit(font_sm.render("Best Agent Speed", True, TXT_COL), (bx, by))
        by += 17
        bw = int(best.speed / MAX_SPD * 196)
        pygame.draw.rect(surf, (55, 55, 55), (bx, by, 196, 18))
        r = best.speed / MAX_SPD
        bar_col = ((50, 220, 50) if r < 0.55
                   else (220, 180, 30) if r < 0.82
                   else (220, 55, 55))
        pygame.draw.rect(surf, bar_col, (bx, by, bw, 18))
        lbl = font_sm.render(f"{best.speed:.2f} / {MAX_SPD:.1f} px/f",
                             True, TXT_COL)
        surf.blit(lbl, (bx + 200, by))

    legends = [
        (BEST_COL,        "* Best Agent"),
        (ACC_COL,         "  Accident"),
        (POT_COL,         "  Pothole"),
        ((220,  30,  30), "  Red Light"),
        (( 30, 220,  30), "  Green Light"),
        (EDGE_COL,        "  Road Edge"),
    ]
    lx, ly = SW - 185, SH - len(legends) * 18 - 14
    lp = pygame.Surface((180, len(legends) * 18 + 10), pygame.SRCALPHA)
    lp.fill((*PNL_COL, 205))
    surf.blit(lp, (lx - 5, ly - 5))
    for col, lbl in legends:
        pygame.draw.rect(surf, col, (lx, ly + 3, 12, 12))
        surf.blit(font_sm.render(lbl, True, TXT_COL), (lx, ly))
        ly += 18

    hint = font_sm.render("SPACE: reset all    +/-: sim speed    S: save    ESC: quit",
                          True, (160, 160, 160))
    surf.blit(hint, (SW // 2 - hint.get_width() // 2, SH - 22))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption(
        "Multi-Agent RL Car Simulation  |  SPACE=reset  +/-=speed  ESC=quit")
    clock = pygame.time.Clock()

    font_sm = pygame.font.SysFont("consolas", 13)
    font_md = pygame.font.SysFont("consolas", 15, bold=True)
    font_lg = pygame.font.SysFont("consolas", 20, bold=True)

    road = Road()
    tls  = [TrafficLight(road, f, i * 58)
            for i, f in enumerate(TL_FRACS)]
    obs  = [Obstacle(road, f, t)
            for f, t in zip(OBS_FRACS, OBS_TYPES)]

    Q_table = load_q_table()
    epsilon = [EPS_INIT]

    agents = [Car(i, road, tls, obs, Q_table, epsilon)
              for i in range(NUM_AGENTS)]

    bg = pygame.Surface((SW, SH))
    bg.fill(BG_COL)
    road.draw(bg)
    for ob in obs:
        ob.draw(bg)

    frame     = 0
    sim_speed = 1

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    for a in agents:
                        a.reset()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim_speed = min(sim_speed * 2, 8)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim_speed = max(sim_speed // 2, 1)
                elif ev.key == pygame.K_s:
                    save_q_table(Q_table)

        for _ in range(sim_speed):
            for tl in tls:
                tl.update()
            for a in agents:
                a.step()
            epsilon[0] = max(EPS_MIN, epsilon[0] * EPS_DECAY)
            frame += 1

        best_val = max(a.best_rew for a in agents)
        for a in agents:
            a.is_best = (a.best_rew >= best_val)
        found = False
        for a in agents:
            if a.is_best:
                if found:
                    a.is_best = False
                else:
                    found = True

        screen.blit(bg, (0, 0))

        for tl in tls:
            tl.draw(screen)

        for a in sorted(agents, key=lambda x: int(x.is_best)):
            a.draw(screen)

        draw_hud(screen, agents, epsilon[0], frame, sim_speed,
                 font_sm, font_md, font_lg)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
