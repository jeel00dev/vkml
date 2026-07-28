"""Interactive MNIST viewer: browse the test set, or draw a digit yourself.

Loads weights saved by train.py and shows what the model predicts, with the
probability it assigns to every class -- so a confident right answer and a
lucky one look different, which a bare prediction hides.

Usage:
    python examples/mnist/gui.py                    # the MLP, GPU if present
    python examples/mnist/gui.py --model cnn        # the convolutional model
    python examples/mnist/gui.py --device cpu       # force the reference backend

The window title names the device, and everything the model does -- including
each digit you draw -- runs there.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# Prefer an installed vkml; fall back to the in-tree package when there is not
# one, so this runs both from a clone and after `pip install .`.
import sys
import importlib.util
if importlib.util.find_spec("vkml") is None:
    sys.path.insert(0, str(HERE.parent.parent / "python"))

import vkml as V  # noqa: E402

import data as mnist_data  # noqa: E402
import train as trainer  # noqa: E402

CANVAS = 280           # on-screen drawing area, 10x the model's input
MNIST_SIDE = 28
DIGIT_BOX = 20         # MNIST normalises the glyph into a 20x20 box...
BRUSH = 12             # ...so a stroke this wide scales to roughly 1.2 pixels

BAR_WIDTH = 220
BAR_HEIGHT = 18


def to_mnist_frame(drawing: np.ndarray) -> np.ndarray:
    """Convert a raw drawing into MNIST's normalisation, or the model will not
    recognise it.

    MNIST is not "a 28x28 picture of a digit". Every glyph was scaled to fit a
    20x20 box and then placed so its CENTRE OF MASS sits at the centre of a
    28x28 frame. A drawing that skips this is a different distribution from the
    one the model was trained on, and predictions look broken for a reason that
    has nothing to do with the model.
    """
    ink = np.argwhere(drawing > 0)
    if len(ink) == 0:
        return np.zeros((MNIST_SIDE, MNIST_SIDE), dtype=np.float32)

    (top, left), (bottom, right) = ink.min(0), ink.max(0) + 1
    glyph = Image.fromarray(drawing[top:bottom, left:right])

    # Scale the longer side to 20, keeping the aspect ratio: a stretched '1'
    # would look like nothing in the training set.
    height, width = bottom - top, right - left
    scale = DIGIT_BOX / max(height, width)
    glyph = glyph.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                         Image.LANCZOS)
    small = np.asarray(glyph, dtype=np.float32)

    # Place by centre of mass rather than by bounding box: MNIST centres the
    # ink, and for a glyph like '7' the two differ by several pixels.
    frame = np.zeros((MNIST_SIDE, MNIST_SIDE), dtype=np.float32)
    rows, cols = small.shape
    total = small.sum()
    if total > 0:
        centre_row = (small.sum(axis=1) @ np.arange(rows)) / total
        centre_col = (small.sum(axis=0) @ np.arange(cols)) / total
    else:
        centre_row, centre_col = rows / 2, cols / 2

    top_offset = int(round(MNIST_SIDE / 2 - centre_row))
    left_offset = int(round(MNIST_SIDE / 2 - centre_col))
    top_offset = max(0, min(MNIST_SIDE - rows, top_offset))
    left_offset = max(0, min(MNIST_SIDE - cols, left_offset))

    frame[top_offset:top_offset + rows, left_offset:left_offset + cols] = small
    return frame / 255.0


class Viewer:
    def __init__(self, root: tk.Tk, model: V.nn.Module, device, dataset: dict,
                 model_name: str):
        self.root = root
        self.model = model
        self.device = device
        self.test_x = dataset["test_x"]
        self.test_y = dataset["test_y"]
        self.index = 0
        self.drawing_mode = False
        self.strokes = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        self.last_point = None

        root.title(f"vkml MNIST — {model_name} on {device}")
        root.configure(bg="#1e1e1e")

        self._build_image_pane()
        self._build_probability_pane()
        self._build_controls()
        self.show_test_image(0)

    # -- layout -------------------------------------------------------------

    def _build_image_pane(self) -> None:
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.grid(row=0, column=0, padx=14, pady=14)

        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black",
                                highlightthickness=1, highlightbackground="#555")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.caption = tk.Label(left, text="", font=("TkDefaultFont", 11),
                                bg="#1e1e1e", fg="#dddddd")
        self.caption.pack(pady=(8, 0))

        self.verdict = tk.Label(left, text="", font=("TkDefaultFont", 22, "bold"),
                                bg="#1e1e1e")
        self.verdict.pack(pady=(4, 0))

    def _build_probability_pane(self) -> None:
        right = tk.Frame(self.root, bg="#1e1e1e")
        right.grid(row=0, column=1, padx=(0, 14), pady=14, sticky="n")

        tk.Label(right, text="confidence", font=("TkDefaultFont", 11, "bold"),
                 bg="#1e1e1e", fg="#dddddd").grid(row=0, column=0, columnspan=3,
                                                  pady=(0, 6))
        self.bars = []
        self.percents = []
        for digit in range(10):
            tk.Label(right, text=str(digit), width=2, font=("TkFixedFont", 12),
                     bg="#1e1e1e", fg="#dddddd").grid(row=digit + 1, column=0)
            bar = tk.Canvas(right, width=BAR_WIDTH, height=BAR_HEIGHT, bg="#2a2a2a",
                            highlightthickness=0)
            bar.grid(row=digit + 1, column=1, pady=1)
            percent = tk.Label(right, text="0.0%", width=7, anchor="e",
                               font=("TkFixedFont", 10), bg="#1e1e1e", fg="#dddddd")
            percent.grid(row=digit + 1, column=2, padx=(6, 0))
            self.bars.append(bar)
            self.percents.append(percent)

    def _build_controls(self) -> None:
        bar = tk.Frame(self.root, bg="#1e1e1e")
        bar.grid(row=1, column=0, columnspan=2, pady=(0, 14))

        def button(text, command, width=10):
            return tk.Button(bar, text=text, command=command, width=width)

        button("< prev", lambda: self.step(-1)).pack(side=tk.LEFT, padx=3)
        button("next >", lambda: self.step(1)).pack(side=tk.LEFT, padx=3)
        button("random", self.random_image).pack(side=tk.LEFT, padx=3)

        tk.Label(bar, text="  index:", bg="#1e1e1e", fg="#dddddd").pack(side=tk.LEFT)
        self.index_entry = tk.Entry(bar, width=7)
        self.index_entry.pack(side=tk.LEFT, padx=(2, 3))
        self.index_entry.bind("<Return>", lambda _event: self.jump())
        button("go", self.jump, width=4).pack(side=tk.LEFT, padx=(0, 12))

        button("next mistake", self.next_mistake, width=13).pack(side=tk.LEFT, padx=3)
        self.draw_button = button("draw mode", self.toggle_draw, width=11)
        self.draw_button.pack(side=tk.LEFT, padx=(12, 3))
        button("clear", self.clear_drawing).pack(side=tk.LEFT, padx=3)

    # -- prediction ---------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Class probabilities for one 28x28 image."""
        batch = image.reshape(1, 1, MNIST_SIDE, MNIST_SIDE).astype(np.float32)
        with V.no_grad():
            logits = self.model(V.tensor(batch, device=self.device))
            return V.softmax(logits, -1).numpy()[0]

    def _render_probabilities(self, probabilities: np.ndarray, truth: int | None) -> None:
        best = int(np.argmax(probabilities))
        for digit, (bar, label) in enumerate(zip(self.bars, self.percents)):
            share = float(probabilities[digit])
            bar.delete("all")
            if share > 0.001:
                colour = "#4caf50" if digit == best else "#3a6ea5"
                if truth is not None and digit == best and best != truth:
                    colour = "#c0392b"
                bar.create_rectangle(0, 0, max(2, share * BAR_WIDTH), BAR_HEIGHT,
                                     fill=colour, width=0)
            label.config(text=f"{share * 100:5.1f}%",
                         fg="#ffffff" if digit == best else "#888888")
        return best

    def _show(self, image: np.ndarray, truth: int | None, caption: str) -> None:
        self._paint(image)
        probabilities = self.predict(image)
        best = self._render_probabilities(probabilities, truth)
        confidence = float(probabilities[best]) * 100

        self.caption.config(text=caption)
        if truth is None:
            self.verdict.config(text=f"guess: {best}   ({confidence:.1f}%)", fg="#dddddd")
        elif best == truth:
            self.verdict.config(text=f"✓ {best}   ({confidence:.1f}%)", fg="#4caf50")
        else:
            self.verdict.config(text=f"✗ {best}, not {truth}   ({confidence:.1f}%)",
                                fg="#e74c3c")

    def _paint(self, image: np.ndarray) -> None:
        """Draw a 28x28 image as a block-scaled bitmap.

        PhotoImage.put with a whole-row string per row: setting 784 pixels
        individually is visibly slow in tkinter, and this is called on every
        navigation.
        """
        scale = CANVAS // MNIST_SIDE
        photo = tk.PhotoImage(width=CANVAS, height=CANVAS)
        levels = np.clip(image * 255, 0, 255).astype(np.uint8)
        for row in range(MNIST_SIDE):
            colours = " ".join(f"#{v:02x}{v:02x}{v:02x}" for v in levels[row]
                               for _ in range(scale))
            photo.put("{" + colours + "}", to=(0, row * scale, CANVAS, (row + 1) * scale))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo   # keep a reference or tkinter frees it

    # -- test-set navigation ------------------------------------------------

    def show_test_image(self, index: int) -> None:
        self.drawing_mode = False
        self.draw_button.config(text="draw mode", relief=tk.RAISED)
        self.index = index % len(self.test_x)
        truth = int(self.test_y[self.index])
        self._show(self.test_x[self.index, 0], truth,
                   f"test image {self.index}   ·   label {truth}")

    def step(self, delta: int) -> None:
        self.show_test_image(self.index + delta)

    def random_image(self) -> None:
        self.show_test_image(int(np.random.randint(len(self.test_x))))

    def jump(self) -> None:
        try:
            self.show_test_image(int(self.index_entry.get()))
        except ValueError:
            self.caption.config(text="index must be a whole number")

    def next_mistake(self) -> None:
        """Find the next image the model gets wrong -- the interesting ones."""
        for offset in range(1, len(self.test_x) + 1):
            candidate = (self.index + offset) % len(self.test_x)
            probabilities = self.predict(self.test_x[candidate, 0])
            if int(np.argmax(probabilities)) != int(self.test_y[candidate]):
                self.show_test_image(candidate)
                return
        self.caption.config(text="no mistakes in the test set")

    # -- drawing ------------------------------------------------------------

    def toggle_draw(self) -> None:
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.draw_button.config(text="drawing", relief=tk.SUNKEN)
            self.clear_drawing()
        else:
            self.show_test_image(self.index)

    def clear_drawing(self) -> None:
        self.strokes[:] = 0
        self.last_point = None
        if self.drawing_mode:
            self.canvas.delete("all")
            self.caption.config(text="draw a digit with the mouse")
            self.verdict.config(text="", fg="#dddddd")
            for bar, label in zip(self.bars, self.percents):
                bar.delete("all")
                label.config(text="  0.0%", fg="#888888")

    def _on_drag(self, event) -> None:
        if not self.drawing_mode:
            return
        x, y = event.x, event.y
        if self.last_point is not None:
            # Interpolate: a fast drag fires motion events far apart, and
            # stamping only at those points leaves a dotted line.
            x0, y0 = self.last_point
            steps = max(abs(x - x0), abs(y - y0), 1)
            for i in range(steps + 1):
                self._stamp(round(x0 + (x - x0) * i / steps),
                            round(y0 + (y - y0) * i / steps))
        else:
            self._stamp(x, y)
        self.last_point = (x, y)

        radius = BRUSH // 2
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill="white", outline="white")

    def _stamp(self, x: int, y: int) -> None:
        radius = BRUSH // 2
        top, bottom = max(0, y - radius), min(CANVAS, y + radius)
        left, right = max(0, x - radius), min(CANVAS, x + radius)
        self.strokes[top:bottom, left:right] = 255

    def _on_release(self, _event) -> None:
        if not self.drawing_mode:
            return
        self.last_point = None
        self._show(to_mnist_frame(self.strokes), None,
                   "your drawing, normalised to MNIST's 20x20-in-28x28 framing")


def load_model(name: str, device) -> V.nn.Module:
    weights = HERE / f"{name}.vkml"
    if not weights.exists():
        raise SystemExit(
            f"no weights at {weights}\n"
            f"train first:  python examples/mnist/train.py --model {name}"
        )
    # load, then check, then install -- in that order. load_module would do all
    # three at once, and the key-set mismatch would fire before the metadata
    # could be consulted, which is the confusing error this check exists to
    # replace.
    checkpoint = V.load(weights)

    saved_name = checkpoint.metadata.get("model")
    if saved_name is not None and saved_name != name:
        raise SystemExit(f"{weights} holds a '{saved_name}' model, not '{name}'")

    model = trainer.MODELS[name][0]()
    model.load_state_dict(checkpoint.tensors)

    accuracy = checkpoint.metadata.get("test_accuracy")
    if accuracy is not None:
        print(f"loaded {weights.name}: {saved_name}, "
              f"{accuracy * 100:.2f}% test accuracy when saved")

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(trainer.MODELS), default="mlp")
    parser.add_argument("--device", default="auto",
                        help="auto (GPU if present), cpu, or vulkan:N")
    args = parser.parse_args()

    # Same resolution as training, from one place: the window title shows the
    # device, and it disagreeing with what train.py chose is exactly the
    # confusion this shares a helper to avoid.
    device = trainer.resolve_device(args.device)

    model = load_model(args.model, device)
    dataset = mnist_data.load()

    root = tk.Tk()
    Viewer(root, model, device, dataset, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
