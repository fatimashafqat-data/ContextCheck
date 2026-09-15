"""Render portfolio figures from recorded results, without fitting or querying models."""
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import get_data_path

ROOT = Path(__file__).resolve().parents[1]
FONT_ROOT = Path(get_data_path()) / "fonts/ttf"
NAVY, MUTED, WHITE = "#101a30", "#a6b5cd", "#f4f7fc"
TEAL, LILAC, AMBER = "#83e0cf", "#b7a4fc", "#ffc58f"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def text(draw, xy, value, size=24, color=WHITE, bold=False):
    draw.text(xy, str(value), fill=color, font=font(size, bold))


def wrapped(draw, xy, value, width, size=24, color=WHITE, bold=False, spacing=9):
    words = value.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if current and draw.textlength(candidate, font=font(size, bold)) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for line in lines:
        text(draw, xy, line, size, color, bold)
        xy = (xy[0], xy[1] + size + spacing)
    return xy[1]


def render_cover():
    image = Image.new("RGB", (1600, 700), NAVY)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 8), fill=TEAL)
    draw.rounded_rectangle((62, 48, 112, 98), radius=12, fill=TEAL)
    text(draw, (74, 50), "C", 34, NAVY, True)
    text(draw, (131, 55), "CONTEXTCHECK", 27, WHITE, True)
    text(draw, (65, 140), "A STUDY OF LANGUAGE, LABELS & MODEL ERRORS", 20, MUTED)
    text(draw, (58, 210), "When toxicity", 76, WHITE, True)
    text(draw, (58, 300), "models disagree.", 76, WHITE, True)
    wrapped(draw, (65, 425), "Classic machine learning. LLM assessments. Evidence at every step.", 740, 30, MUTED, spacing=15)

    draw.rounded_rectangle((910, 140, 1538, 534), radius=28, fill="#192640", outline="#31415b", width=2)
    text(draw, (951, 174), "CLASSIC MODELS / TEST MACRO F1", 22, MUTED, True)
    scores = {row["model"]: float(row["macro_f1"]) for row in csv.DictReader((ROOT / "results/test_model_comparison.csv").open())}
    for y, (name, color) in zip([244, 335, 426], [("Logistic Regression", TEAL), ("Random Forest", LILAC), ("Majority baseline", "#93a4bd")]):
        text(draw, (951, y), name, 24, WHITE)
        text(draw, (1390, y-3), f"{scores[name]:.3f}", 30, color, True)
        draw.rounded_rectangle((951, y+41, 1479, y+52), radius=5, fill="#30405a")
        draw.rounded_rectangle((951, y+41, 951+int(528*scores[name]), y+52), radius=5, fill=color)

    draw.line((65, 590, 1535, 590), fill="#30405a", width=2)
    text(draw, (65, 622), "24,544 modeled tweets", 25, WHITE, True)
    text(draw, (607, 622), "3,682 held-out test examples", 25, WHITE, True)
    text(draw, (1290, 622), "CPU / COLAB", 25, TEAL, True)
    image.save(ROOT / "images/contextcheck_cover.png", optimize=True)


def render_case():
    r = ROOT / "results"
    row = next(row for row in csv.DictReader((r / "llm_comparison_rows.csv").open()) if row["source_id"] == "18879")
    labels = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
    image = Image.new("RGB", (1600, 860), "#f0f4fa")
    draw = ImageDraw.Draw(image)
    text(draw, (65, 42), "RECORDED ASSESSMENT / SOURCE #18879", 21, "#52617b", True)
    text(draw, (61, 86), "A joke or an insult?", 53, NAVY, True)
    text(draw, (65, 160), "Same input. Different readings of the word \"trash\".", 27, "#52617b")
    draw.rounded_rectangle((60, 215, 1540, 360), radius=20, fill="white", outline="#dae1ee", width=2)
    excerpt = '"In 2014 you tweeted that Migos were trash. Care to explain?"'
    text(draw, (93, 246), excerpt, 31, NAVY)
    text(draw, (93, 300), "2064 Supreme Court Confirmation Hearing", 27, "#52617b")
    for x, key, title, accent in [(60, "gemini", "Gemini", "#096958"), (815, "openrouter", "Nemotron via OpenRouter", "#7e470c")]:
        draw.rounded_rectangle((x, 405, x+725, 694), radius=22, fill="white", outline="#dae1ee", width=2)
        text(draw, (x+33, 435), title, 28, NAVY, True)
        value = labels[int(float(row[key+"_label"]))]
        text(draw, (x+33, 485), value, 34, accent, True)
        wrapped(draw, (x+33, 551), row[key+"_explanation"], 655, 23, "#52617b", spacing=10)
    lr = labels[int(row["logistic_regression"])]
    rf = labels[int(row["random_forest"])]
    truth = labels[int(row["label"])]
    text(draw, (65, 735), f"Dataset label: {truth}     |     Logistic Regression: {lr}     |     Random Forest: {rf}", 24, NAVY, True)
    text(draw, (65, 798), "Saved benchmark outputs. Tweet excerpt omits the account mention. Try your own text in the Colab demo.", 21, "#52617b")
    image.save(ROOT / "images/recorded_disagreement.png", optimize=True)


if __name__ == "__main__":
    render_cover()
    render_case()
    print("Rendered portfolio figures from the saved evaluation and response table.")
