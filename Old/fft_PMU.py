
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:28:46 2021

@author: ldominguez@inti.gob.ar
"""

from scipy import signal # para procesar señales
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calcular_fft(y, freq_sampleo):
    '''y debe ser un vector con números reales representando datos de una 
    serie temporal.
    freq_sampleo es un número entero.
    La función devuelve dos vectores, uno de frecuencias y otro con la 
    transformada propiamente. La transformada contiene los valores complejos
    que se corresponden con respectivas frecuencias.'''
    
    N = len(y)
    freq = np.fft.fftfreq(N, d = 1/freq_sampleo)[:N//2]
    tran = (np.fft.fft(y)/N)[:N//2]
    return freq, tran

def graficar(x1,y1,x2,y2):
    
    '''
    Graficamos la señal original y la transformada de Fourier
    '''
    
    #Señal original (dominio del tiempo)
    plt.figure(1)
    plt.subplot(2, 1,1) 
    plt.plot(vect_tiempo,senoidal)
    plt.title("Señal digitalizada")
    plt.xlabel("tiempo (s)")
    plt.ylabel("Tensión (V)")

    #Transformada de Fourier
    plt.subplot(2,1,2) 
    plt.stem(x2,y2)
    plt.title("FFT de señal digitalizada")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Tensión eficaz (V)")
    plt.xlim(0,N/10)

    #Ajustes del gráfico
    top=0.88
    bottom=0.11
    left=0.11
    right=0.9
    hspace=0.41
    wspace=0.2
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    plt.show()
    

#%%

if __name__ == '__main__':
    
    # Creamos el dataFrame
    df = pd.read_csv('Data/TimeStamp_+_ADC_values.csv',sep=';',parse_dates = True)
    
    # Configuramos los parámetros de la señal digitalizada
    inicio = 0
    fin = 5000
    N=fin-inicio
    freq_sampleo=50000
    t_sampleo=1/freq_sampleo

    #Dominio del tiempo (señal original)
    vect_tiempo=(np.arange(N))*t_sampleo
    # Pasamos la serie como vector de numpy, y selleccionamos la primer columna
    senoidal = df[inicio:fin]['11/11/2021 19:35:05.999'].to_numpy()

    #Dominio de la frecuencia
    # Calculamos amplitud de la fft y su correspondencia con cada freq
    # (vector de frecuencias)
    vect_freqs, fft_senoidal = calcular_fft(senoidal,freq_sampleo)

    #Graficamos
    graficar(vect_tiempo,senoidal,vect_freqs, np.abs(fft_senoidal))



