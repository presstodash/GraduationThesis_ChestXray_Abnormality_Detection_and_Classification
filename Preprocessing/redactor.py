from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import shutil

SOURCE_DIR = Path(r"source/path/png_input")
OUTPUT_DIR = Path(r"output/path/png_redacted")
EXTENSIONS = {".png"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted([
    p for p in SOURCE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONS
])

if not files:
    raise SystemExit("Nema slika u SOURCE_DIR.")

index = 0
rectangles = []
start_x = start_y = None
current_preview_rect = None

root = tk.Tk()
root.title("PNG redaktor: drag = bounding box, Enter/N = spremi i dalje, U = undo, S = skip, Q = izlaz")

canvas = tk.Canvas(root, bg="gray20", cursor="crosshair")
canvas.pack(fill=tk.BOTH, expand=True)

status = tk.Label(root, text="", anchor="w")
status.pack(fill=tk.X)

original_img = None
display_img = None
tk_img = None
scale = 1.0
offset_x = 0
offset_y = 0


def load_image():
    global original_img, display_img, tk_img, rectangles, scale, offset_x, offset_y

    rectangles = []
    img_path = files[index]
    original_img = Image.open(img_path).convert("RGBA")

    screen_w = root.winfo_screenwidth() - 120
    screen_h = root.winfo_screenheight() - 160

    w, h = original_img.size
    scale = min(screen_w / w, screen_h / h, 1.0)

    display_w = int(w * scale)
    display_h = int(h * scale)

    display_img = original_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(display_img)

    canvas.delete("all")
    canvas.config(width=display_w, height=display_h)

    offset_x = 0
    offset_y = 0

    canvas.create_image(offset_x, offset_y, anchor="nw", image=tk_img)
    update_status()


def update_status():
    img_path = files[index]
    status.config(
        text=f"{index + 1}/{len(files)} | {img_path.name} | "
             f"Boxeva: {len(rectangles)} | Enter/N=spremi+next | U=undo | S=skip | Q=izlaz"
    )


def canvas_to_image_coords(x, y):
    ix = int((x - offset_x) / scale)
    iy = int((y - offset_y) / scale)

    ix = max(0, min(ix, original_img.width))
    iy = max(0, min(iy, original_img.height))

    return ix, iy


def redraw():
    canvas.delete("all")
    canvas.create_image(offset_x, offset_y, anchor="nw", image=tk_img)

    for x1, y1, x2, y2 in rectangles:
        canvas.create_rectangle(
            x1 * scale, y1 * scale,
            x2 * scale, y2 * scale,
            fill="black",
            outline="black"
        )

    update_status()


def on_mouse_down(event):
    global start_x, start_y, current_preview_rect
    start_x, start_y = event.x, event.y
    current_preview_rect = canvas.create_rectangle(
        start_x, start_y, start_x, start_y,
        outline="red",
        width=2
    )


def on_mouse_drag(event):
    if current_preview_rect is not None:
        canvas.coords(current_preview_rect, start_x, start_y, event.x, event.y)


def on_mouse_up(event):
    global start_x, start_y, current_preview_rect

    if start_x is None or start_y is None:
        return

    x1, y1 = canvas_to_image_coords(start_x, start_y)
    x2, y2 = canvas_to_image_coords(event.x, event.y)

    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])

    if right - left > 2 and bottom - top > 2:
        rectangles.append((left, top, right, bottom))

    current_preview_rect = None
    start_x = start_y = None
    redraw()


def save_and_next(event=None):
    global index

    src = files[index]
    dst = OUTPUT_DIR / src.name

    img = original_img.copy()
    draw = ImageDraw.Draw(img)

    for rect in rectangles:
        draw.rectangle(rect, fill=(0, 0, 0, 255))

    img.save(dst)

    index += 1
    if index >= len(files):
        status.config(text=f"Gotovo. Spremljeno u: {OUTPUT_DIR}")
        canvas.delete("all")
        return

    load_image()


def skip_image(event=None):
    global index

    # src = files[index]
    # shutil.copy2(src, OUTPUT_DIR / src.name)

    index += 1
    if index >= len(files):
        status.config(text=f"Gotovo. Spremljeno u: {OUTPUT_DIR}")
        canvas.delete("all")
        return

    load_image()


def undo(event=None):
    if rectangles:
        rectangles.pop()
        redraw()


def quit_app(event=None):
    root.destroy()


canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

root.bind("<Return>", save_and_next)
root.bind("n", save_and_next)
root.bind("N", save_and_next)
root.bind("u", undo)
root.bind("U", undo)
root.bind("s", skip_image)
root.bind("S", skip_image)
root.bind("q", quit_app)
root.bind("Q", quit_app)
root.bind("<Escape>", quit_app)

load_image()
root.mainloop()