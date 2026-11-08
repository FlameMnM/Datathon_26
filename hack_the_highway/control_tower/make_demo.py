"""Assemble the storyboard screenshots into a captioned animated GIF (runs/run1/demo.gif)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FRAMES = [
    ("01_run_console", "1/12  Events arrive from GET /events; each is understood (Gemini 3.5 Flash-Lite) and handled live"),
    ("02_E-01_event", "2/12  E-01 compound digest: LLM splits it into a plant outage + a supplier delay; code grounds names & dates"),
    ("03_E-01_graph_paths", "3/12  One graph path per incident: Plant->PO->CustomerOrder->Dealer/Customer and Shipment->Part->Model->PO->..."),
    ("04_E-01_trace", "4/12  Agents acting: SupplyChainPlanner moves orders to the earliest free slot; CustomerDealerAgent notifies + assigns loaners"),
    ("05_E-01_actions", "5/12  Actions taken: reschedule, dealer.notify (one per dealer), assign_loaner - every call with its reason, 0 failures"),
    ("05b_E-01_what_changed", "6/12  What changed - and what did not: business impact, plus every record deliberately left alone (with counts)"),
    ("06_E-11_slot_reasoning_transfers", "7/12  Capacity-aware slot reasoning: no same-plant slot left -> transfer to another plant that builds the model"),
    ("07_E-03_quality_graph_path", "8/12  Quality hold: Lot -> ProductionOrder -> Vehicle -> Dealer (unsold: stop sale) / Customer (sold: contact owner)"),
    ("08_E-05_noise_no_action", "9/12  Noise: judged not operational -> no tool calls at all"),
    ("09_E-13_rumor_escalated", "10/12  Unconfirmed rumor: escalate to a human, no operational change"),
    ("10_audit_trail", "11/12  Audit trail: every tool call of the run (reads + writes), agent, arguments, reason, result"),
    ("11_ontology", "12/12  Ontology / knowledge graph: 12 classes, 19 relations, 17,040 nodes"),
]


def main(run):
    run = Path(run)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    except OSError:
        font = ImageFont.load_default(size=19)
    import textwrap
    W = 1200
    frames = []
    for name, cap in FRAMES:
        im = Image.open(run / "screenshots" / f"{name}.png").convert("RGB")
        im = im.resize((W, int(im.height * W / im.width)))
        im = im.crop((0, 0, W, min(im.height, 860)))
        bar = 70
        canvas = Image.new("RGB", (W, 860 + bar), (17, 24, 39))
        canvas.paste(im, (0, bar))
        ImageDraw.Draw(canvas).multiline_text((16, 10), "\n".join(textwrap.wrap(cap, 96)[:2]), fill=(255, 255, 255),
                                              font=font, spacing=6)
        frames.append(canvas)
    write_mp4(run, frames, W, 860 + 70)
    # per-frame adaptive palette, full frames (no delta encoding) -> exact colours, no cross-frame artifacts
    frames = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    out = run / "demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[4500] * len(frames), loop=0,
                   optimize=False, disposal=1)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames, ~{len(frames) * 4.5:.0f} s)")


def card(W, H, lines):
    """Plain text card (title / closing) for the video."""
    im = Image.new("RGB", (W, H), (17, 24, 39))
    d = ImageDraw.Draw(im)
    y = H // 2 - 40 * len(lines) // 2
    for text, size, colour in lines:
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            f = ImageFont.load_default(size=size)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, fill=colour, font=f)
        y += int(size * 1.9)
    return im


def write_mp4(run, frames, W, H, seconds=5.0):
    """demo.mp4: title card + the 12 captioned frames + results card (needs: pip install imageio-ffmpeg)."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("imageio-ffmpeg not installed - skipping demo.mp4 (the GIF is still written)")
        return
    import json
    import subprocess
    import tempfile
    meta = json.loads((run / "run_meta.json").read_text())
    white, grey, red = (255, 255, 255), (170, 180, 190), (230, 70, 80)
    title = card(W, H, [("Hack the Highway - Wacky Wheels Control Tower", 40, white),
                        ("Team penberthy-pushers  |  end-to-end demo of Run 1", 26, grey),
                        (f"Coordinator + 3 specialist agents  |  model: {meta.get('model')}", 24, grey),
                        ("15 events arrive  ->  understood  ->  graph traversal  ->  actions  ->  audit trail", 22, red)])
    calls = sum(1 for _ in open(run / "action_log.jsonl", encoding="utf-8"))
    end = card(W, H, [("Run 1 result", 40, white),
                      (f"15/15 events handled  |  {calls} tool calls  |  0 failed", 28, white),
                      ("Independent replay (verify.py): all policy checks pass on every event", 22, grey),
                      ("3 noise events: 0 actions   |   2 rumors: escalated, 0 writes", 22, grey),
                      ("Explore it yourself: runs/run1/control_tower.html", 24, red)])
    seq = [title] + frames + [end]
    with tempfile.TemporaryDirectory() as tmp:
        for i, im in enumerate(seq):
            im.save(Path(tmp) / f"f{i:02d}.png")
        out = run / "demo.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", f"1/{seconds}",
                        "-i", str(Path(tmp) / "f%02d.png"), "-vf", "fps=25,format=yuv420p", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(seq)} slides, ~{len(seq) * seconds:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "runs" / "run1")
