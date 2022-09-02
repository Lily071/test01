#from turtle import color
import pandas as pd
import matplotlib.pyplot as plt

## 追加1
# グラフの描画先の準備
# plt（最初にインポートしたmatplotlib.pyplotというモジュール）にある
# figure()メソッドを、figに代入しておきます。
fig = plt.figure()
## 追加1

def graph(x, x_rl, y_ud):
    plt.title('matplotlib graph')
    plt.ylabel('Diff', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.grid()
    #print(x, diffx, diffy)
    plt.plot(x, x_rl)
    plt.plot(x, y_ud) 
    #plt.plot(x, diffy, color='b', linewidth='5', label='UD') 
    plt.legend(fontsize=15)
    #plt.show()
    fig.savefig("./static/imgfile/graph.png")

## 追加2
# ファイルに保存
# fig（準備した描写先であるFigure(432x288)のようなデータが準備されています）に
# img.pngという画像を保存しています。
#fig.savefig("img.png")
## 追加2
