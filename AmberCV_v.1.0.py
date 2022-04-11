import cv2
import numpy as np
import webcolors


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        colour_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        colour_name = closest_colour(requested_colour)
    return colour_name


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

img = cv2.imread('test1.JPG')
img = cv2.resize(img, (640, 480))


height, width, _ = np.shape(img)
# print(height, width)

data = np.reshape(img, (height * width, 3))
data = np.float32(data)

number_clusters = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
centers = centers[centers[:, 0 or 1 or 2] < 170]

font = cv2.FONT_HERSHEY_SIMPLEX
bars = []
rgb_values = []

for index, row in enumerate(centers):
    bar, rgb = create_bar(200, 200, row)
    bars.append(bar)
    rgb_values.append(rgb)
img_bar = np.hstack(bars)
print(rgb_values)
#print(webcolors.rgb_to_name(rgb_values[0]))

for index, row in enumerate(rgb_values):
    image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                        font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
thresh = cv2.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# отображаем контуры поверх изображения
cv2.drawContours(img, contours, -1, (0,255,0), 2, cv2.LINE_AA, hierarchy, 1)

x, y = [], []
for i in list(contours):
    for j in i:
        x.append(j[0][0])
        y.append(j[0][1])
max_x, max_y = max(x), max(y)
print(max_x, max_y)

cv2.putText(img, f'{get_colour_name(rgb_values[0])}', (275, max_y+20),
                        font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imshow('Image', img)
cv2.imshow('Dominant colors', img_bar)
cv2.waitKey(0)