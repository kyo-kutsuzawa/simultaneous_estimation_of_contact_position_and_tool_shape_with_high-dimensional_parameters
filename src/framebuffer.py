import mmap
import os
import numpy as np
from matplotlib.backend_bases import FigureCanvasBase


class FrameBuffer:
    def __init__(self, fb_name: str = "fb0") -> None:
        self.fb_name = fb_name

        # Setup frame buffer-related paths
        dev_path = f"/dev/{fb_name}"
        bits_per_pixel_path = f"/sys/class/graphics/{self.fb_name}/bits_per_pixel"
        size_path = f"/sys/class/graphics/{self.fb_name}/virtual_size"

        # Check the frame buffer exists
        assert os.path.exists(dev_path)
        assert os.path.exists(bits_per_pixel_path)
        assert os.path.exists(size_path)

        # Get the color bit
        with open(bits_per_pixel_path, "r") as f:
            self.color_bit = int(f.read())

        # Get the screen size
        with open(size_path, "r") as f:
            self.screen_width, self.screen_height = map(int, f.read().split(","))

        # Load a frame buffer
        self.fb_dev = os.open(dev_path, os.O_RDWR)
        self.fb = mmap.mmap(
            self.fb_dev,
            self.screen_width * self.screen_height * self.color_bit // 8,
            mmap.MAP_SHARED,
            mmap.PROT_WRITE | mmap.PROT_READ,
            offset=0,
        )

        # Initialize ndarray to draw
        self.screen = np.zeros(
            (self.screen_height, self.screen_width, 4), dtype=np.uint8
        )

    def show(self, canvas: FigureCanvasBase, margin: int = 100) -> None:
        # Get ndarray data from the canvas
        data = canvas.buffer_rgba()
        data = np.frombuffer(data, dtype=np.uint8)
        data_width, data_height = canvas.get_width_height()
        data = data.reshape(data_height, data_width, 4)

        self._show_data(data, margin)

    def _show_data(self, data: np.ndarray, margin: int = 100) -> None:
        assert data.shape[2] == 4
        assert margin + data.shape[0] < self.screen_height
        assert margin + data.shape[1] < self.screen_width

        y = slice(margin, margin + data.shape[0])
        x = slice(margin, margin + data.shape[1])

        # Set values
        self.screen[y, x, 0] = data[:, :, 2]  # Blue
        self.screen[y, x, 1] = data[:, :, 1]  # Green
        self.screen[y, x, 2] = data[:, :, 0]  # Red
        self.screen[y, x, 3] = data[:, :, 3]  # Alpha

        # Write to the frame buffer
        self.fb.seek(0)
        self.fb.write(self.screen.tobytes())
