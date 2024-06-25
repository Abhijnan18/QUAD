from PIL import Image, ImageDraw
from collections import Counter
import heapq
import sys
import os

MODE_RECTANGLE = 1
MODE_ELLIPSE = 2
MODE_ROUNDED_RECTANGLE = 3

# MODE = MODE_ELLIPSE  # Change the default mode to ellipse For Cricle
MODE = MODE_RECTANGLE
ITERATIONS = 1024
LEAF_SIZE = 4
PADDING = 1
FILL_COLOR = (0, 0, 0)
SAVE_FRAMES = False
ERROR_RATE = 0.5
AREA_POWER = 0.25
OUTPUT_SCALE = 1


def weighted_average(hist):
    total = sum(hist)
    value = sum(i * x for i, x in enumerate(hist)) / total
    error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
    error = error ** 0.5
    return value, error


def color_from_histogram(hist):
    r, re = weighted_average(hist[:256])
    g, ge = weighted_average(hist[256:512])
    b, be = weighted_average(hist[512:768])
    e = re * 0.2989 + ge * 0.5870 + be * 0.1140
    return (int(r), int(g), int(b)), e


def rounded_rectangle(draw, box, radius, color):
    l, t, r, b = box
    d = radius * 2
    draw.ellipse((l, t, l + d, t + d), fill=color)
    draw.ellipse((r - d, t, r, t + d), fill=color)
    draw.ellipse((l, b - d, l + d, b), fill=color)
    draw.ellipse((r - d, b - d, r, b), fill=color)
    d = radius
    draw.rectangle((l, t + d, r, b - d), fill=color)
    draw.rectangle((l + d, t, r - d, b), fill=color)


class Quad:
    id_counter = 0

    def __init__(self, model, box, depth):
        self.model = model
        self.box = box
        self.depth = depth
        self.id = Quad.id_counter
        Quad.id_counter += 1
        hist = self.model.im.crop(self.box).histogram()
        self.color, self.error = color_from_histogram(hist)
        self.leaf = self.is_leaf()
        self.area = self.compute_area()
        self.children = []

    def is_leaf(self):
        l, t, r, b = self.box
        return int(r - l <= LEAF_SIZE or b - t <= LEAF_SIZE)

    def compute_area(self):
        l, t, r, b = self.box
        return (r - l) * (b - t)

    def split(self):
        l, t, r, b = self.box
        lr = l + (r - l) / 2
        tb = t + (b - t) / 2
        depth = self.depth + 1
        tl = Quad(self.model, (l, t, lr, tb), depth)
        tr = Quad(self.model, (lr, t, r, tb), depth)
        bl = Quad(self.model, (l, tb, lr, b), depth)
        br = Quad(self.model, (lr, tb, r, b), depth)
        self.children = [tl, tr, bl, br]
        return self.children

    def get_leaf_nodes(self, max_depth=None):
        if not self.children:
            return [self]
        if max_depth is not None and self.depth >= max_depth:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_leaf_nodes(max_depth))
        return result


class Model:
    def __init__(self, path):
        self.im = Image.open(path).convert('RGB')
        self.width, self.height = self.im.size
        self.heap = []
        self.root = Quad(self, (0, 0, self.width, self.height), 0)
        self.error_sum = self.root.error * self.root.area
        self.push(self.root)

    @property
    def quads(self):
        return [x[-1] for x in self.heap]

    def average_error(self):
        return self.error_sum / (self.width * self.height)

    def push(self, quad):
        score = -quad.error * (quad.area ** AREA_POWER)
        heapq.heappush(self.heap, (quad.leaf, score, quad.id, quad))

    def pop(self):
        return heapq.heappop(self.heap)[-1]

    def split(self):
        quad = self.pop()
        self.error_sum -= quad.error * quad.area
        children = quad.split()
        for child in children:
            self.push(child)
            self.error_sum += child.error * child.area

    def render(self, path, max_depth=None):
        m = OUTPUT_SCALE
        dx, dy = (PADDING, PADDING)
        im = Image.new('RGB', (self.width * m + dx, self.height * m + dy))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, self.width * m, self.height * m), FILL_COLOR)
        for quad in self.root.get_leaf_nodes(max_depth):
            l, t, r, b = quad.box
            box = (l * m + dx, t * m + dy, r * m - 1, b * m - 1)
            if MODE == MODE_ELLIPSE:
                draw.ellipse(box, quad.color)
            elif MODE == MODE_ROUNDED_RECTANGLE:
                radius = m * min((r - l), (b - t)) / 4
                rounded_rectangle(draw, box, radius, quad.color)
            else:
                draw.rectangle(box, quad.color)
        del draw
        im.save(path, 'PNG')


def process_image(input_path, output_image_folder, output_gif_folder):
    model = Model(input_path)
    frames_dir = 'frames'
    os.makedirs(frames_dir, exist_ok=True)

    previous = None
    frames = []

    for i in range(ITERATIONS):
        error = model.average_error()
        if previous is None or previous - error > ERROR_RATE:
            print(f'Iteration: {i}, Error: {error}')
            if SAVE_FRAMES:
                frame_path = os.path.join(frames_dir, f'{i:06d}.png')
                model.render(frame_path)
                frames.append(frame_path)
            previous = error
        model.split()

    # Save final output image
    os.makedirs(output_image_folder, exist_ok=True)
    output_image_file = os.path.join(output_image_folder, 'output.png')
    model.render(output_image_file)

    # Create GIF
    if frames:
        os.makedirs(output_gif_folder, exist_ok=True)
        gif_file = os.path.join(output_gif_folder, 'output.gif')
        delay = [20] * len(frames)
        delay[-1] = 200  # Last frame delay
        delay_str = ' '.join(
            f'-delay {delay[i]} {frames[i]}' for i in range(len(frames)))
        os.system(
            f'magick convert -loop 0 {delay_str} {output_image_file} {gif_file}')
        print(f'GIF saved: {gif_file}')


def main():
    input_folder = 'img'
    output_image_folder = 'output_images_square'
    output_gif_folder = 'output_gif_square'

    os.makedirs(output_image_folder, exist_ok=True)
    # os.makedirs(output_gif_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_image_folder, filename)
            output_gif_path = os.path.join(output_gif_folder, filename.split('.')[
                                           0])  # Remove extension for GIF folder name
            os.makedirs(output_gif_path, exist_ok=True)

            process_image(input_path, output_image_path, output_gif_path)


if __name__ == '__main__':
    main()
