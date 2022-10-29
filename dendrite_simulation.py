# 実行用Githubリンク: https://github.com/gapul/dendrite_python.git

import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
from math import *
import time
from mpl_toolkits.mplot3d.axes3d import Axes3D

graph_type = 2 # 0:円形 1:四角形, 2:球体
a = 20 # 金属板の大きさ
N = 1000 # 試行するイオン数
t = time.time() # シミュレーション経過時間

# イオンの開始位置を定義(正方形)
def position_2d_sq():
  x_list = []
  y_list = []
  while len(x_list) != N:
    x = 2*a*(random()-0.5) # イオンの開始位置を定義
    y = 2*a*(random()-0.5)
    if abs(x+y)+abs(x-y) >= a/2+0.5 and x**2+y**2 <= a**2:
      x_list.append(x)
      y_list.append(y)
  return x_list, y_list

# イオンの開始位置を定義(円形)
def position_2d_cir():
  x_list = []
  y_list = []
  while len(x_list) != N:
    x = 2*a*(random()-0.5) # イオンの開始位置を定義
    y = 2*a*(random()-0.5)
    if  (a/4)**2 <= x**2+y**2 <= a**2:
      x_list.append(x)
      y_list.append(y)
  return x_list, y_list

# イオンの開始位置を定義(球体)
def position_3d_ball():
  x_list = []
  y_list = []
  z_list = []
  while len(x_list) != N:
    x = 2*a*(random()-0.5) # イオンの開始位置を定義
    y = 2*a*(random()-0.5)
    z = 2*a*(random()-0.5)
    if (a/4)**2 <= x**2+y**2+z**2 <= a**2:
      x_list.append(x)
      y_list.append(y)
      z_list.append(z)
  return x_list, y_list, z_list

# イオンの移動(正方形)
def moving_2d_sq():
  x_list1, y_list1 = position_2d_sq()
  x_list2 = []
  y_list2 = []
  plate = 0 # 金属板への析出量
  ion = 0 # イオンへの析出量
  for num in range(0, len(x_list1)):
    m = 0
    while m == 0:
      theta = 2.0*pi*random() # [0,2π)の乱数を発生
      x_backup = x_list1[num]
      y_backup = y_list1[num]
      x_list1[num] += cos(theta) # x方向への移動
      y_list1[num] += sin(theta) # y方向への移動
      if abs(x_list1[num]+y_list1[num])+abs(x_list1[num]-y_list1[num])<=a/2+0.5: # 金属板に触れたイオンを析出
        x_list2.append(x_list1[num])
        y_list2.append(y_list1[num])
        plate += 1
        m = 1
      elif (x_list1[num])**2+(y_list1[num])**2 >=(a*2)**2: # 離れていったイオンを移動前に戻す
        x_list1[num] = x_backup
        y_list1[num] = y_backup
        m = 0
      elif len(x_list2)<=0 :
        m = 0
      else:
        if min(x_list2)-1<=x_list1[num] and x_list1[num]<=max(x_list2)+1 and min(y_list2)-1<=y_list1[num] and y_list1[num]<=max(y_list2)+1: # 析出済みイオンに近づいたか否かを判別(計算量削減)
          for cx in x_list2:
            if (x_list1[num]-cx)**2+(y_list1[num]-y_list2[x_list2.index(cx)])**2 <=1: # 析出済みイオンに触れた別のイオンを析出
              x_list2.append(x_list1[num])
              y_list2.append(y_list1[num])
              ion += 1
              m = 1
              break
  return x_list2, y_list2, plate, ion

def moving_2d_cir():
  x_list1, y_list1 = position_2d_cir()
  x_list2 = []
  y_list2 = []
  plate = 0 # 金属板への析出量
  ion = 0 # イオンへの析出量
  for num in range(0, len(x_list1)):
    m = 0
    while m == 0:
      theta = 2.0*pi*random() # [0,2π)の乱数を発生
      x_backup = x_list1[num]
      y_backup = y_list1[num]
      x_list1[num] += cos(theta) # x方向への移動
      y_list1[num] += sin(theta) # y方向への移動
      if (x_list1[num])**2+(y_list1[num])**2 <=(a/4)**2: # 金属板に触れたイオンを析出
        x_list2.append(x_list1[num])
        y_list2.append(y_list1[num])
        plate += 1
        m = 1
      elif (x_list1[num])**2+(y_list1[num])**2 >=(a*2)**2: # 離れていったイオンを移動前に戻す
        x_list1[num] = x_backup
        y_list1[num] = y_backup
        m = 0
      elif len(x_list2)<=0 :
        m = 0
      else:
        if min(x_list2)-1<=x_list1[num] and x_list1[num]<=max(x_list2)+1 and min(y_list2)-1<=y_list1[num] and y_list1[num]<=max(y_list2)+1: # 析出済みイオンに近づいたか否かを判別(計算量削減)
          for cx in x_list2:
            if (x_list1[num]-cx)**2+(y_list1[num]-y_list2[x_list2.index(cx)])**2 <=1: # 析出済みイオンに触れた別のイオンを析出
              x_list2.append(x_list1[num])
              y_list2.append(y_list1[num])
              ion += 1
              m = 1
              break
  return x_list2, y_list2, plate, ion

def moving_3d():
  x_list1, y_list1, z_list1 = position_3d_ball()
  x_list2 = []
  y_list2 = []
  z_list2 = []
  plate = 0 # 金属板への析出量
  ion = 0 # イオンへの析出量
  for num in range(0, len(x_list1)):
    m = 0
    while m == 0:
      theta1 = 2.0*pi*random() # [0,2π)の乱数を発生
      theta2 = 2.0*pi*random()
      x_backup = x_list1[num]
      y_backup = y_list1[num]
      z_backup = z_list1[num]
      x_list1[num] += cos(theta1)*cos(theta2) # x方向への移動
      y_list1[num] += sin(theta1)*cos(theta2) # y方向への移動
      z_list1[num] += sin(theta2) # z方向への移動
      if (x_list1[num])**2+(y_list1[num])**2+(z_list1[num])**2<=(a/4)**2: # 金属板に触れたイオンを析出
        x_list2.append(x_list1[num])
        y_list2.append(y_list1[num])
        z_list2.append(z_list1[num])
        plate += 1
        m = 1
      elif (x_list1[num])**2+(y_list1[num])**2+(z_list1[num])**2>=(a*2)**2: # 離れていったイオンを移動前に戻す
        x_list1[num] = x_backup
        y_list1[num] = y_backup
        z_list1[num] = z_backup
        m = 0
      elif len(x_list2)<=0 :
        m = 0
      else:
        if min(x_list2)-1<=x_list1[num] and x_list1[num]<=max(x_list2)+1 and min(y_list2)-1<=y_list1[num] and y_list1[num]<=max(y_list2)+1 and min(z_list2)-1<=z_list1[num] and z_list1[num]<=max(z_list2)+1: # 析出済みイオンに近づいたか否かを判別(計算量削減)
          for cx in x_list2:
            if (x_list1[num]-cx)**2+(y_list1[num]-y_list2[x_list2.index(cx)])**2+(z_list1[num]-z_list2[x_list2.index(cx)])**2<=1: # 析出済みイオンに触れた別のイオンを析出
              x_list2.append(x_list1[num])
              y_list2.append(y_list1[num])
              z_list2.append(z_list1[num])
              ion += 1
              m = 1
              break
  return x_list2, y_list2, z_list2, plate, ion

# 数値データを出力(二次元)
def info_2d(x_list, y_list, plate, ion):
  print(x_list)
  print(y_list)
  print("------------------------------------------------")
  print("time:"+str(time.time()-t))
  print("plate:"+str(plate))
  print("ion:"+str(ion))

# 数値データを出力(三次元)
def info_3d(x_list, y_list, z_list, plate, ion):
  print(x_list)
  print(y_list)
  print(z_list)
  print("------------------------------------------------")
  print("time:"+str(time.time()-t))
  print("plate:"+str(plate))
  print("ion:"+str(ion))

# 図形データを出力(正方形)
def graph_2d_sq(x_list, y_list):
  fig = plt.figure(figsize=(5,5)) # 出力サイズ
  ax = fig.add_subplot(1,1,1) # グラフのサイズ
  plt.scatter(x_list,y_list) # (x,y)の散布図
  rec=plt.Rectangle(xy=(-a/4,-a/4), width=a/2, height=a/2,fill=False) # 四角形の定義
  ax.add_patch(rec) # 四角形の描写の追加
  plt.xlabel('X') # x軸のラベル
  plt.ylabel('Y') # y軸のラベル
  plt.xlim([-a,a]) # x軸の範囲
  plt.ylim([-a,a]) # y軸の範囲
  plt.show()

# 図形データを出力(円形)
def graph_2d_cir(x_list, y_list):
  fig = plt.figure(figsize=(5,5)) # 出力サイズ
  ax = fig.add_subplot(1,1,1) # グラフのサイズ
  plt.scatter(x_list,y_list) # (x,y)の散布図
  rec=plt.Circle((0,0),a/4,fill=False) # 円形の定義
  ax.add_patch(rec) # 円形の描写の追加
  plt.xlabel('X') # x軸のラベル
  plt.ylabel('Y') # y軸のラベル
  plt.xlim([-a,a]) # x軸の範囲
  plt.ylim([-a,a]) # y軸の範囲
  plt.show()

# 図形データを出力(球体)
def graph_3d_ball(x_list, y_list, z_list):
  fig = plt.figure(figsize=(5,5))#出力サイズ
  ax = Axes3D(fig)#グラフのサイズ
  ax.scatter(x_list,y_list,z_list)
  u = np.linspace(0, 2 * np.pi, 100) # 球体の定義
  v = np.linspace(0, np.pi, 100)
  x = (a/4) * np.outer(np.cos(u), np.sin(v))
  y = (a/4)  * np.outer(np.sin(u), np.sin(v))
  z = (a/4)  * np.outer(np.ones(np.size(u)), np.cos(v))
  ax.plot_surface(x, y, z,color="lightgreen",rcount=100, ccount=100, antialiased=False) # 球体の描写の追加
  for num in range(0, len(x_list)): # 析出したイオンを追加
    x=[x_list[num]]
    y=[y_list[num]]
    z=[z_list[num]]
    for px in range(0, len(x_list)):
      if (x_list[num]-x_list[px])**2+(y_list[num]-y_list[px])**2+(z_list[num]-z_list[px])**2<=1 and (x_list[num]-x_list[px])**2+(y_list[num]-y_list[px])**2+(z_list[num]-z_list[px])**2 != 0:
        x.append(x_list[px])
        y.append(y_list[px])
        z.append(z_list[px])
        ax.plot(x, y, z, marker="o", color="#00aa00", ms=4, mew=0.5)
  plt.show()

# 実行
if graph_type == 0:
  x_list, y_list, plate, ion = moving_2d_cir()
  info_2d(x_list, y_list, plate, ion)
  graph_2d_cir(x_list, y_list)
elif graph_type == 1:
  x_list, y_list, plate, ion = moving_2d_sq()
  info_2d(x_list, y_list, plate, ion)
  graph_2d_sq(x_list, y_list)
elif graph_type == 2:
  x_list, y_list, z_list, plate, ion = moving_3d()
  info_3d(x_list, y_list, z_list, plate, ion)
  graph_3d_ball(x_list, y_list, z_list)
else:
  print("graph_type error")