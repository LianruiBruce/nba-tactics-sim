"""Render a stretch of a replay to an animated GIF (for the README).

  python scripts/make_gif.py --start 176.5 --end 192.5                 # replays/sample.json -> docs/demo.gif
  python scripts/make_gif.py --replay my.json --start 0 --end 30 --speed 2 --out clip.gif

Times are replay seconds (the `t` of each frame). Chinese captions need a CJK
font (WenQuanYi, Noto CJK, PingFang, YaHei...); without one the GIF uses English.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbasim import court  # noqa: E402

S = 9.0            # pixels per foot
HEAD = 58          # scoreboard height
FOOT = 34          # caption bar height
W, H = int(court.LENGTH * S), int(court.WIDTH * S) + HEAD + FOOT

FLOOR = (236, 224, 202)
PAINT = (226, 206, 172)
LINE = (120, 96, 70)
INK = (28, 39, 51)
MUTED = (93, 107, 120)
PANEL = (250, 250, 247)
TEAM = [(31, 95, 191), (196, 55, 44)]
BALL = (224, 122, 31)

CJK_FONTS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
LATIN_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
               "C:/Windows/Fonts/arial.ttf"]

PLAYS = {
    "high_pnr": ("高位挡拆", "High pick-and-roll"),
    "horns": ("牛角", "Horns"),
    "floppy": ("弹簧", "Floppy"),
    "spain_pnr": ("西班牙挡拆", "Spain pick-and-roll"),
    "post_up": ("低位单打", "Post-up"),
    "iso": ("拉开单打", "Isolation"),
}
ZONES = {"rim": ("篮下", "at the rim"), "short_mid": ("近距离", "short mid-range"),
         "long_mid": ("中距离", "long two"), "corner3": ("底角三分", "corner three"),
         "above3": ("三分", "three")}


def load_font(size: int, cjk: bool):
    for path in (CJK_FONTS if cjk else LATIN_FONTS):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return None


def pt(x: float, y: float) -> tuple[float, float]:
    return x * S, HEAD + y * S


def draw_court(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, HEAD, W, HEAD + court.WIDTH * S], fill=FLOOR)
    lw = 2
    for direction in (1, -1):
        bx, by = court.basket(direction)
        base = 0.0 if direction < 0 else court.LENGTH
        sgn = 1 if direction < 0 else -1  # toward half court
        fx = base + sgn * court.PAINT_LENGTH
        x0, x1 = sorted([base, fx])
        d.rectangle([*pt(x0, 25 - court.PAINT_HALF_WIDTH), *pt(x1, 25 + court.PAINT_HALF_WIDTH)],
                    fill=PAINT, outline=LINE, width=lw)
        cx, cy = pt(fx, 25)
        d.ellipse([cx - 6 * S, cy - 6 * S, cx + 6 * S, cy + 6 * S], outline=LINE, width=lw)
        # three-point line: corner segments plus the arc between them
        r = court.THREE_ARC_RADIUS
        break_x = base + sgn * court.THREE_BREAK_LY
        for yy in (25 - court.THREE_CORNER_DX, 25 + court.THREE_CORNER_DX):
            d.line([pt(base, yy), pt(break_x, yy)], fill=LINE, width=lw)
        half = math.degrees(math.asin(court.THREE_CORNER_DX / r))
        mid = 0 if direction < 0 else 180
        px, py = pt(bx, by)
        d.arc([px - r * S, py - r * S, px + r * S, py + r * S], mid - half, mid + half, fill=LINE, width=lw)
        ra = 4 * S
        d.arc([px - ra, py - ra, px + ra, py + ra], mid - 90, mid + 90, fill=LINE, width=lw)
        # backboard and rim
        bb = base + sgn * 4.0
        d.line([pt(bb, 22), pt(bb, 28)], fill=INK, width=3)
        d.ellipse([px - 0.75 * S, py - 0.75 * S, px + 0.75 * S, py + 0.75 * S], outline=BALL, width=2)
    d.line([pt(court.HALF, 0), pt(court.HALF, court.WIDTH)], fill=LINE, width=lw)
    cx, cy = pt(court.HALF, 25)
    d.ellipse([cx - 6 * S, cy - 6 * S, cx + 6 * S, cy + 6 * S], outline=LINE, width=lw)
    d.rectangle([0, HEAD, W - 1, HEAD + court.WIDTH * S], outline=LINE, width=lw)


def caption(ev: dict, names: list[str], zh: bool) -> str | None:
    k = 0 if zh else 1
    t = ev["type"]
    by = names[ev["by"]] if "by" in ev and ev["by"] is not None else ""
    if t == "play_call" and ev.get("play") in PLAYS:
        return ("战术：" if zh else "Play: ") + PLAYS[ev["play"]][k]
    if t == "shot":
        z = ZONES.get(ev.get("zone"), (ev.get("zone"), ev.get("zone")))[k]
        if ev.get("blocked"):
            return f"{by} {z}出手被盖" if zh else f"{by} {z} attempt blocked"
        return f"{by} {z}出手" if zh else f"{by} shoots ({z})"
    if t == "score" and ev.get("pts", 0) >= 2:
        return f"{by} 得分 +{ev['pts']}" if zh else f"{by} scores +{ev['pts']}"
    if t == "miss":
        return "没进" if zh else "No good"
    if t == "pass":
        to = names[ev["to"]]
        return f"{by} → {to}"
    if t == "drive":
        if ev.get("beat"):
            return f"{by} 突破过掉 {names[ev['on']]}" if zh else f"{by} blows by {names[ev['on']]}"
        return f"{by} 突破" if zh else f"{by} drives"
    if t == "steal":
        return f"{by} 抢断" if zh else f"{by} steal"
    if t == "block":
        return f"{by} 盖帽！" if zh else f"{by} block!"
    if t == "rebound":
        kind = ("进攻篮板" if ev.get("offensive") else "防守篮板") if zh else \
            ("offensive rebound" if ev.get("offensive") else "defensive rebound")
        return f"{by} {kind}"
    if t == "foul":
        return f"{by} 犯规" if zh else f"Foul on {by}"
    if t == "free_throw":
        return f"{by} 罚球{'命中' if ev.get('made') else '不中'}" if zh else \
            f"{by} free throw {'good' if ev.get('made') else 'missed'}"
    return None


def clock(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60)}:{int(sec % 60):02d}" if sec >= 60 else f"{sec:.1f}"


def render(rep: dict, start: float, end: float, speed: float, out: Path) -> int:
    meta = rep["meta"]
    names = [p["name"].split()[-1] for p in meta["players"]]
    pos = [p["pos"] for p in meta["players"]]
    teams = meta["teams"]
    frames = [f for f in rep["frames"] if start <= f["t"] <= end]
    if not frames:
        raise SystemExit(f"no frames between {start} and {end}")
    events = [e for e in rep["events"] if e["t"] <= end]

    zh_font = load_font(20, cjk=True)
    zh = zh_font is not None
    big = load_font(26, cjk=zh) or ImageFont.load_default()
    mid = load_font(18, cjk=zh) or ImageFont.load_default()
    small = load_font(11, cjk=False) or ImageFont.load_default()

    base = Image.new("RGB", (W, H), PANEL)
    draw_court(ImageDraw.Draw(base))

    step = max(1, round(speed))  # replay runs at 10 Hz; keep every `step`-th frame
    dt_ms = int(100 * step / speed)
    images = []
    for f in frames[::step]:
        im = base.copy()
        d = ImageDraw.Draw(im)
        # scoreboard
        d.rectangle([0, 0, W, HEAD], fill=PANEL)
        x = 14
        for k in (0, 1):
            d.text((x, 14), f"{teams[k]} {f['s'][k]}", font=big, fill=TEAM[k])
            x += d.textlength(f"{teams[k]} {f['s'][k]}", font=big) + 28
        q = f"Q{f['q']}" if f["q"] <= 4 else f"OT{f['q'] - 4}"
        info = f"{q}  {clock(f['gc'])}   {'进攻时间' if zh else 'Shot clock'} {math.ceil(f['sc'])}"
        d.text((W - 14 - d.textlength(info, font=big), 14), info, font=big, fill=INK)
        lab = f["lab"][f["off"]] if f.get("lab") else ""
        if lab:
            d.text((x, 20), "· " + lab if zh else "", font=mid, fill=MUTED)
        # players
        for i, (px, py) in enumerate(f["p"]):
            cx, cy = pt(px, py)
            r = court.PLAYER_RADIUS * S
            col = TEAM[0 if i < 5 else 1]
            if f["h"] == i:
                d.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], outline=BALL, width=3)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(255, 255, 255), width=2)
            d.text((cx, cy), pos[i], font=small, fill=(255, 255, 255), anchor="mm")
            d.text((cx, cy + r + 8), names[i], font=small, fill=INK, anchor="mm")
        # ball (bigger when in the air)
        bx, by, bz = f["ball"]
        cx, cy = pt(bx, by)
        if f["h"] is not None:  # dribbling at the hip, not over the number
            cx, cy = cx + 0.9 * S, cy + 0.6 * S
        br = 4 + min(bz, 16.0) * 0.35
        d.ellipse([cx - br, cy - br, cx + br, cy + br], fill=BALL, outline=(120, 60, 10))
        # caption: latest interesting event in the last 2.5 s
        cap = None
        for e in reversed(events):
            if e["t"] > f["t"]:
                continue
            if f["t"] - e["t"] > 2.5:
                break
            cap = caption(e, names, zh)
            if cap:
                break
        d.rectangle([0, H - FOOT, W, H], fill=PANEL)
        if cap:
            d.text((14, H - FOOT + 6), cap, font=mid, fill=INK)
        images.append(im.quantize(colors=64, method=Image.Quantize.MEDIANCUT))

    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=dt_ms, loop=0, optimize=True)
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(ROOT / "replays" / "sample.json"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice real time)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    args = ap.parse_args()
    rep = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    n = render(rep, args.start, args.end, args.speed, Path(args.out))
    print(f"wrote {args.out} ({n} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
