import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore")

class GPSVis(object):
    """
        Class for GPS data visualization using pre-downloaded OSM map in image format.
    """

    def __init__(self, data_path, map_path, points):
        """
        :param data_path: Path to file containing GPS records.
        :param map_path: Path to pre-downloaded OSM map in image format.
        :param points: Upper-left, and lower-right GPS points of the map (lat1, lon1, lat2, lon2).
        """
        self.data_path = data_path
        self.points = points
        self.map_path = map_path

        self.result_image = Image
        self.x_ticks = []
        self.y_ticks = []

    def plot_map(self, output='save', save_as='resultMap.png'):
        """
        Method for plotting the map. You can choose to save it in file or to plot it.
        :param output: Type 'plot' to show the map or 'save' to save it.
        :param save_as: Name and type of the resulting image.
        :return:
        """
        self.get_ticks()
        fig, axis1 = plt.subplots(figsize=(15, 15))
        axis1.imshow(self.result_image)
        # DIANA
        x1, y1 = self.scale_to_img((51.519137, 5.857951), (self.result_image.size[0], self.result_image.size[1]))
        axis1.scatter(x1, y1, c='magenta', s=100)
        # VENUS
        x1, y1 = self.scale_to_img((51.5192716, 5.8579155), (self.result_image.size[0], self.result_image.size[1]))
        axis1.scatter(x1, y1, c='red', s=100)
        # ALVIRA
        x1, y1 = self.scale_to_img((51.52126391, 5.85862734), (self.result_image.size[0], self.result_image.size[1]))
        axis1.scatter(x1, y1, c='green', s=100)
        # ARCUS
        x1, y1 = self.scale_to_img((51.52147, 5.87056833), (self.result_image.size[0], self.result_image.size[1]))
        axis1.scatter(x1, y1, c='yellow', s=100)

        axis1.set_xlabel('Longitude')
        axis1.set_ylabel('Latitude')
        axis1.set_xticklabels([5.8426,5.8491,5.8557,5.8622,5.8687,5.8752,5.8817,5.8882,5.8947,5.9012])
        axis1.set_yticklabels(self.y_ticks)
        axis1.grid()
        if output == 'save':
            plt.savefig(save_as)
        else:
            plt.show()

    def create_image(self, color, width=1):
        """
        Create the image that contains the original map and the GPS records.
        :param latitude:
        :param color: Color of the GPS records.
        :param width: Width of the drawn GPS records.
        :return:
        """
        data = pd.read_csv(self.data_path, names=['LATITUDE', 'LONGITUDE'], sep=',')

        self.result_image = Image.open(self.map_path, 'r')
        img_points = []
        gps_data = tuple(zip(data['LATITUDE'].values, data['LONGITUDE'].values))

        '''
        x1, y1 = self.scale_to_img(gps_data[0], (self.result_image.size[0], self.result_image.size[1]))
        img_points.append((x1, y1))

        img_points = []
        '''
        for d in gps_data[1:]:
            x1, y1 = self.scale_to_img(d, (self.result_image.size[0], self.result_image.size[1]))
            img_points.append((x1, y1))
        draw = ImageDraw.Draw(self.result_image)
        draw.line(img_points, fill=color, width=width)
        draw.point(img_points, fill='red')
        #draw.polygon(img_points, fill=color)

    def scale_to_img(self, lat_lon, h_w):
        """
        Conversion from latitude and longitude to the image pixels.
        It is used for drawing the GPS records on the map image.
        :param lat_lon: GPS record to draw (lat1, lon1).
        :param h_w: Size of the map image (w, h).
        :return: Tuple containing x and y coordinates to draw on map image.
        """
        # https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445
        old = (self.points[2], self.points[0])
        new = (0, h_w[1])
        y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        old = (self.points[1], self.points[3])
        new = (0, h_w[0])
        x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        # y must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        return int(x), h_w[1] - int(y)

    def get_ticks(self):
        """
        Generates custom ticks based on the GPS coordinates of the map for the matplotlib output.
        :return:
        """
        self.x_ticks = map(
            lambda x: round(x, 4),
            np.linspace(self.points[1], self.points[3], num=7))
        y_ticks = map(
            lambda x: round(x, 4),
            np.linspace(self.points[2], self.points[0], num=8))
        # Ticks must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        self.y_ticks = sorted(y_ticks, reverse=True)

    def display_gps(self, path, scenario):
        vis = GPSVis(data_path=path,
                     map_path='map1.png',  # Path to map downloaded from the OSM.
                     points=(51.5246, 5.8361, 51.5103, 5.8752))  # Two coordinates of the map (upper left, lower right)

        vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
        vis.plot_map(output='save', save_as='Maps_scenario_'+scenario)
