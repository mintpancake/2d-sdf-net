import numpy as np
import cv2
import time

CANVAS_SIZE = (800, 800)
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)


# The PolygonDrawer class is adapted from
# https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python/37235130
class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name  # Name for our window

        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print(f'Adding point #{len(self.points)} with position ({x},{y})')
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print(f"Completing polygon with {len(self.points)} points")
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            # This is our drawing loop, we just continuously draw new shape_images
            # and show them in the named window
            canvas = np.zeros(CANVAS_SIZE, np.uint8)
            if len(self.points) > 0:
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        # User finished entering the polygon points, so let's make the final drawing
        canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if len(self.points) > 0:
            cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return self.points, canvas


class DataSaver(object):
    def __init__(self, points, image):
        self.points = points
        self.image = image
        self.data_path = '../shapes/raw/'
        self.image_path = '../shapes/raw_images/'
        self.save_name = f'polygon_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_save_name(self, save_name):
        self.save_name = save_name

    def save(self):
        f = open(f'{self.data_path}{self.save_name}.txt', 'w')
        for point in self.points:
            # Normalize (x,y) to (0,1)
            x = np.double(np.double(point[0]) / np.double(CANVAS_SIZE[0]))
            y = np.double(np.double(point[1]) / np.double(CANVAS_SIZE[1]))
            f.write(f'{x} {y}\n')
        f.close()

        cv2.imwrite(f'{self.image_path}{self.save_name}.png', self.image)


if __name__ == '__main__':
    print(f'Enter save name (skip with enter key): ')
    name = input()

    drawer = PolygonDrawer('Left click: Add vertices    Right click: Complete polygon    Any key: Save data')
    points, image = drawer.run()
    print(f'Polygon = {points}')

    saver = DataSaver(points, image)
    if name != '':
        saver.set_save_name(name)
    saver.save()
    print(f'Data path = {saver.data_path}{saver.save_name}.txt')
    print(f'Image path = {saver.image_path}{saver.save_name}.png')
    print('Done!')
