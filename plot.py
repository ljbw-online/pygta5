import numpy as np
import cv2
from matplotlib import pyplot

pyplot.style.use('dark_background')


class Plot:
    def __init__(self):
        figure, axes = pyplot.subplots()
        self.figure = figure
        self.axes = axes
        self.add_line = axes.plot
        self.clear = axes.clear
        self.title = ''
        self.top = None
        self.bottom = None
        self.width = None
        self.height = None
        self.xlabel = None
        self.ylabel = None

    def to_array(self):
        self.axes.set_title(self.title)

        if self.top is not None:
            self.axes.set_ylim(top=self.top)

        if self.bottom is not None:
            self.axes.set_ylim(bottom=self.bottom)

        if self.width is not None:
            self.figure.set_figwidth(self.width)

        if self.height is not None:
            self.figure.set_figheight(self.height)

        if self.xlabel is not None:
            self.axes.set_xlabel(self.xlabel)

        if self.ylabel is not None:
            self.axes.set_ylabel(self.ylabel)

        self.axes.legend()
        self.figure.canvas.draw()  # necessary for tostring_rgb
        width, height = self.figure.canvas.get_width_height(physical=True)
        # flat_array = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
        flat_array = np.frombuffer(self.figure.canvas.buffer_rgba(), dtype=np.uint8)
        # bgr_array = flat_array.reshape((height, width, 3))
        bgr_array = flat_array.reshape((height, width, 4))
        # return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(bgr_array, cv2.COLOR_BGRA2RGB)


if __name__ == '__main__':
    rng = np.random.default_rng()

    plot = Plot()
    plot.title = 'Test plot'
    plot.xlabel = 'Step count'
    plot.top = 1.2
    plot.bottom = -0.3
    plot.width = 13

    x = range(10)

    while True:
        plot.clear()
        y1 = rng.random(size=10)
        y2 = rng.random(size=10)

        plot.add_line(x, y1, 'r', label='should be red')
        plot.add_line(x, y2, 'b', label='should be blue')

        plot_array = plot.to_array()
        cv2.imshow('plot.py', plot_array)
        if cv2.waitKey(1000) == ord('q'):
            cv2.destroyWindow('plot.py')
            break

    # input('input')
