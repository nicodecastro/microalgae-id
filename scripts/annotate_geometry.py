"""Draw reference silhouettes on original resized images, without candidate masks."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import cv2
import numpy as np

from scripts._geometry_references import ReferenceSession, load_canvas, rasterize


class AnnotationWindow:
    """Small polygon editor; only explicit Save/Uncertain actions persist changes."""

    def __init__(self, session):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.session = session
        self.index = next((i for i, row in enumerate(session.data["images"])
                           if row["status"] == "pending"), 0)
        self.figure, self.axes = plt.subplots(figsize=(11, 9))
        self.figure.subplots_adjust(bottom=0.23, top=0.88)
        self.figure.text(0.05, 0.96, "Trace filled cell bodies, not texture. Orange = padding (exclude).", fontsize=11)
        self.figure.text(0.05, 0.925, "Click vertices; click the first vertex to close. Add all intended cells.\n"
                         "Only Save/Uncertain persists edits. Quit/Previous discards unsaved edits.", fontsize=10)
        self.message = self.figure.text(0.05, 0.025, "")
        self.buttons = []
        actions = [("Add polygon", lambda _: self.start_polygon("add")),
                   ("Subtract", lambda _: self.start_polygon("subtract")),
                   ("Undo", self.undo), ("Reset", self.reset),
                   ("Save + next", lambda _: self.save("approved")),
                   ("Uncertain + next", lambda _: self.save("uncertain")),
                   ("Previous", self.previous), ("Quit", lambda _: plt.close(self.figure))]
        for index, (label, callback) in enumerate(actions):
            axis = self.figure.add_axes([0.05 + (index % 4) * 0.23, 0.13 - (index // 4) * 0.065, 0.21, 0.05])
            button = Button(axis, label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.selector = None
        self.load()

    def stop_selector(self):
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector.set_visible(False)
            self.selector.disconnect_events()
            self.selector = None

    def load(self):
        self.stop_selector()
        row = self.session.data["images"][self.index]
        self.canvas, self.content = load_canvas(self.session.settings, row)
        self.polygons = deepcopy(row["polygons"])
        self.message.set_text("")
        self.draw()

    def draw(self):
        self.axes.clear()
        self.axes.imshow(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
        padding = np.zeros((*self.content.shape, 4))
        padding[~self.content] = [1, 0.5, 0, 0.55]
        self.axes.imshow(padding)
        mask = rasterize(self.polygons, self.content.shape)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 1, 0.7, 0.3]
        self.axes.imshow(overlay)
        row = self.session.data["images"][self.index]
        self.axes.set_title(f"{self.index + 1}/{len(self.session.data['images'])} — {row['image_id']}\nSaved status: {row['status']}", fontsize=10)
        self.axes.set_xlim(-0.5, self.content.shape[1] - 0.5)
        self.axes.set_ylim(self.content.shape[0] - 0.5, -0.5)
        self.figure.canvas.draw_idle()

    def start_polygon(self, mode):
        from matplotlib.widgets import PolygonSelector

        self.stop_selector()

        def finished(vertices):
            points = np.asarray(vertices, dtype=float)
            if len(points) < 3 or not np.isfinite(points).all():
                self.message.set_text("A polygon needs at least three finite vertices.")
                return
            points[:, 0] = np.clip(points[:, 0], 0, self.content.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, self.content.shape[0] - 1)
            self.polygons.append({"mode": mode, "vertices": points.tolist()})
            self.stop_selector()
            self.message.set_text("Polygon added; save the image or draw another polygon.")
            self.draw()

        self.selector = PolygonSelector(self.axes, finished, useblit=False)
        self.message.set_text(f"Drawing {mode} polygon. Close it by clicking its first vertex.")
        self.figure.canvas.draw_idle()

    def undo(self, _=None):
        self.stop_selector()
        if self.polygons:
            self.polygons.pop()
        self.draw()

    def reset(self, _=None):
        self.stop_selector()
        self.polygons = []
        self.draw()

    def save(self, status):
        self.stop_selector()
        try:
            self.session.save(self.index, self.polygons, status)
        except (ValueError, OSError) as error:
            self.message.set_text(str(error))
            self.figure.canvas.draw_idle()
            return
        if self.index + 1 < len(self.session.data["images"]):
            self.index += 1
            self.load()
        else:
            self.draw()
            self.message.set_text("Last image saved. Review previous images or Quit; pending images remain excluded from scores.")
            self.figure.canvas.draw_idle()

    def previous(self, _=None):
        self.index = max(0, self.index - 1)
        self.load()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--select-all", action="store_true", help="Select every comparison image in a new session")
    parser.add_argument("--import-references", type=Path, help="Import a compatible reference session into a new session")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare or validate the session without opening the annotation window")
    args = parser.parse_args(argv)
    try:
        session = ReferenceSession(args.comparison_dir, args.output_dir, resume=args.resume,
                                   select_all=args.select_all, import_references=args.import_references)
        if args.prepare_only:
            from collections import Counter
            counts = Counter(row["status"] for row in session.data["images"])
            print(f"Reference session validated: {len(session.data['images'])} images; "
                  f"{counts['approved']} approved, {counts['pending']} pending, {counts['uncertain']} uncertain. "
                  "No annotation window opened.")
            return 0
        import matplotlib
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        window = AnnotationWindow(session)
        plt.show()
        del window
    except Exception as error:
        print(f"Geometry annotation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
