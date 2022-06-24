import sys
from datetime import datetime
import pandas as pd
from Utility.GPS import GPSVis
from Utility.print_test_console import print_on_test_map
from tkinter import *
from tkinter import messagebox

# GUI
print('')
print('*************************************************')
print('*****************  Uranus  **********************')
print('*************************************************')
print('')

print(' All available scenarios ')
print('')

scenari = ['1a', '1b', '2a', '2b', '2c', '2d', '3a']
for number, letter in enumerate(scenari):
    if letter[:1] == '3':
        x = 1
        drone = 'only one drone in flight (Parrot)'
    else:
        x = int(letter[:1])
        if x == 1:
            drone = 'only one drone in flight'
        else:
            drone = 'two drones flying'
    print(number + 1, ')  ' + letter + ' ' + drone)
print('')
scenario = input("Choose test scenario: ")
print('')
print('The chosen scenario is : ' + scenario)
print('')

if scenario == '1':
    range1 = datetime.strptime('2020-09-29 13:00:09', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-29 13:09:16', '%Y-%m-%d %H:%M:%S')

elif scenario == '2':
    range1 = datetime.strptime('2020-09-29 12:29:00', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-29 12:43:06', '%Y-%m-%d %H:%M:%S')

elif scenario == '3':
    range1 = datetime.strptime('2020-09-30 12:00:40', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-30 12:19:40', '%Y-%m-%d %H:%M:%S')

elif scenario == '4':
    range1 = datetime.strptime('2020-09-30 12:24:45', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-30 12:38:03', '%Y-%m-%d %H:%M:%S')

elif scenario == '5':
    range1 = datetime.strptime('2020-09-30 12:44:16', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-30 13:02:43', '%Y-%m-%d %H:%M:%S')

elif scenario == '6':
    range1 = datetime.strptime('2020-09-30 13:04:06', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-30 13:14:03', '%Y-%m-%d %H:%M:%S')

elif scenario == '7':
    range1 = datetime.strptime('2020-09-29 14:12:20', '%Y-%m-%d %H:%M:%S')
    range2 = datetime.strptime('2020-09-29 14:25:11', '%Y-%m-%d %H:%M:%S')

print('Choose range datatime into : ' + str(range1) + ' and ' + str(range2))

first = input("Insert first range value:  ")
first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S')
if first < range1:
    print('Error : The lower datetime limit has been violated')
    quit()
second = input("Insert second range value:  ")
print('')

second = datetime.strptime(second, '%Y-%m-%d %H:%M:%S')
if second > range2:
    print('Error : The upper datetime limit has been violated')
    quit()

data = pd.read_csv('SubmissionFileTestResults/SubmissionFileScenario_' + scenari[int(scenario)-1] + '.csv')
data['TrackDateTimeUTC'] = pd.to_datetime(data['TrackDateTimeUTC'], format='%Y-%m-%d %H:%M:%S')
data = data[data['TrackDateTimeUTC'] >= first]
data = data[data['TrackDateTimeUTC'] <= second]
data = data.sort_values(by=['TrackDateTimeUTC'], ascending=True)
data = data.reset_index()
data = data.drop('index', axis=1)

import tkinter as tk
from tkinter import ttk
import pandas as pd

df = pd.DataFrame(data)
cols = list(df.columns)
root = Tk()
root.title('Uranus')
root.iconbitmap('Utility/uranus.ico')
tree = ttk.Treeview(root, height=50, selectmode="extended")
tree.pack()

tree["columns"] = cols

for i in cols:
    tree.column(i, anchor="w")
    tree.heading(i, text=i, anchor='w')
idx = df.shape[0]
for index, row in df.iterrows():
    tree.insert("", 0, text=idx, values=list(row))
    idx -= 1

root.mainloop()

# print(tabulate(data, headers='keys', tablefmt='psql', bold=True))

print_on_test_map(data, scenario)

detected_drone = data.TrackClassification.unique()
data_class = data.groupby('TrackClassification').size()
data_class = pd.DataFrame(data_class).applymap(str)
data_class = data_class.rename(columns={0:'Occurrences'})

df = pd.DataFrame(data_class)
cols = list(df.columns)

root = Tk()
root.title('Uranus')
root.iconbitmap('Utility/uranus.ico')
tree = ttk.Treeview(root, height=10, selectmode="extended")
tree.pack()

tree["columns"] = cols

for i in cols:
    tree.column(i, anchor="w")
    tree.heading(i, text=i, anchor='w')

for index, row in df.iterrows():
    tree.insert("", 0, text=index, values=list(row))

root.mainloop()

print('***************************************************')

from tkinter import *
from PIL import Image, ImageTk

for drone in detected_drone:
    if drone == 'Parrot':

        root = Tk()
        root.title('Uranus')
        root.iconbitmap('Utility/uranus.ico')
        image = ImageTk.PhotoImage(Image.open(r'Utility/drone_images/PARROT_DISCO_[1].jpeg'))

        Label(root, text='Estimator accuracy : 0.95', image=image, compound='bottom').pack()

        root.mainloop()
    if drone == 'Mavic Pro':

        root = Tk()
        root.title('Uranus')
        root.iconbitmap('Utility/uranus.ico')
        image = ImageTk.PhotoImage(Image.open(r"Utility/drone_images/DJI_MAVIC_PRO_[1].png"))

        Label(root, text='Estimator accuracy : 0.95', image=image, compound='bottom').pack()

        root.mainloop()
    if drone == 'Mavic 2':

        root = Tk()
        root.title('Uranus')
        root.iconbitmap('Utility/uranus.ico')

        image = ImageTk.PhotoImage(Image.open(r'Utility/drone_images/DJI_MAVIC_2_[1].png'))

        Label(root, text='Estimator accuracy : 0.95', image=image, compound='bottom').pack()

        root.mainloop()
    if drone == 'Professional V2':

        root = Tk()
        root.title('Uranus')
        root.iconbitmap('Utility/uranus.ico')

        image = ImageTk.PhotoImage(Image.open(r'Utility/drone_images/DJI_PHANTOM_[1].png'))

        Label(root, text='Estimator accuracy : 0.95', image=image, compound='bottom').pack()

        root.mainloop()

