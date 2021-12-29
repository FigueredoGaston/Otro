# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:04:23 2021

@author: Monoramiro
"""

from warnings import showwarning
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta



def animate(i):
    '''
    Toma los datos desde el CSV creado en PDC, para armar el gráfico en tiempo real
    '''
    global j
    
    data = pd.read_csv('../data/datasetPMU4.csv')
    x= data['TimeStamp']
    #x_time=datetime.time(x)
    #x=PDC.timestamp_to_date(unix_time)

    # Amplitud
    y1 = data['V1']
    y2 = data['V2']
    y3 = data['V3']
    
    plt.subplot(2, 1,1) 
    plt.cla()
    plt.style.use('fivethirtyeight')
    plt.plot(x, y1, label='Channel 1')
    plt.plot(x, y2, label='Channel 2')
    plt.plot(x, y3, label='Channel 3')
    #plt.plot(x, y2, label='Channel 2')
    plt.xticks(range(0,j,5))
    plt.xlabel("TimeStamp")
    plt.ylabel("Tensión(V) ")
    plt.title("Sincrofasores - Amplitud")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Fase
    p1 = data['Phase1']
    p2 = data['Phase2']
    p3 = data['Phase3']
    
    plt.subplot(2, 1,2) 
    plt.cla()
    plt.style.use('fivethirtyeight')
    plt.plot(x, p1, label='Channel 1')
    plt.plot(x, p2, label='Channel 2')
    plt.plot(x, p3, label='Channel 3')
    #plt.plot(x, y2, label='Channel 2')
    plt.xticks(range(0,j,5))
    plt.xlabel("TimeStamp")
    plt.ylabel("Fase(rad) ")
    plt.title("Sincrofasores - Fase")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
    j+=1
    return

j=0
#Realiza el graficado en tiempo real
ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.show()
    