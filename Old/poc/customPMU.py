#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File Name: customPMU.py
# Created By: Hernan Mogliasso
# Created Date: 23/10/2021
# Description: Ejercicio 11.14: precio_alquiler ~ superficie
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as md

def timestamp_to_date(timestamp, prefijo = None):
    """
    Transforma un timestamp en un datetime.
    Un timestamp es un entero que establece la cantidad de 
    segundos desde el 01/01/1970.
    timestamp puede estar en mili, micro o nano segundos. 
    prefijo debe concordar con el prefijo del timestamp a convertir.
    """
    prefijos = {'mili': 10**3, 'micro': 10**6, 'nano': 10**9,}
    if prefijo and prefijo in prefijos:
        segundos = timestamp/prefijos[prefijo]
    else:
        segundos = timestamp
    return datetime(1970, 1, 1) + timedelta(seconds=segundos)

class SineWave:
    """
    Representa una señal senoidal monofasica.
    """
    def __init__(self, frequency, amplitude, phase_shift, duration, start=0) -> None:
        self.frequency = frequency  
        self.amplitude = amplitude  
        self.phase_shift = phase_shift  
        self.duration = duration 
        self.w = 2*np.pi*frequency # angular frequency
        self.fs = 50*frequency     # sampling frequency        
        
        # Errores (en %)
        self.amplitude_error, self.frequency_error, self.phase_shift_error = 5, 0.2, 2
        amplitude_min, amplitude_max = self.amplitude*(1-self.amplitude_error/100), self.amplitude*(1+self.amplitude_error/100) 
        phase_shift_min, phase_shift_max = self.phase_shift*(1-self.phase_shift_error/100), self.phase_shift*(1+self.phase_shift_error/100) 
        freq_min, freq_max = self.frequency*(1-self.frequency_error/100), self.frequency*(1+self.frequency_error/100)
        # array de tiempo    
        self.t = np.arange(start, start+self.duration, 1/self.fs) # array de tiempo
        self.t = np.around(self.t, 4)
        # array de las fases de los fasores
        self.phasors_phase = np.random.uniform(low=phase_shift_min, high=phase_shift_max, size=(len(self.t),))
        # array de las amplitudes de los fasores
        self.phasors_v = np.random.uniform(low=amplitude_min, high=amplitude_max, size=(len(self.t),))        
        # array de las frecuencias de los fasores
        self.phasors_f = np.random.uniform(low=freq_min, high=freq_max, size=(len(self.t),))

    def generate_wave(self):
        """Transforma los fasores en la señal en el dominio del tiempo
           Devuelve un array con los valores de la señal en el tiempo.
        """
        return self.phasors_v*np.cos(self.w*self.t + self.phasors_phase*np.pi/180)
    
    def fft(self):
        """Transforma los fasores en la señal en el dominio de la frecuencia
           Devuelve un array de amplitudes y un array de frecuencias.
        """
        time_domain_signal = self.generate_wave()
        N = len(time_domain_signal)
        freq = np.fft.fftfreq(N, d = 1/self.fs)[:N//2]
        tran = (np.fft.fft(time_domain_signal)/N)[:N//2]
        return tran, freq
    
class ThreePhaseSineWave():
    """
    Representa una señal senoidal trifasica.
    """
    def __init__(self, frequency: float, amplitude: tuple, phase_shift: tuple, duration:float, start:float=0) -> None:
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.duration = duration
        self.start = start
        self.phase1 = SineWave(frequency=self.frequency, amplitude=self.amplitude[0], phase_shift=self.phase_shift[0], duration=self.duration, start=self.start)
        self.phase2 = SineWave(frequency=self.frequency, amplitude=self.amplitude[1], phase_shift=self.phase_shift[1], duration=self.duration, start=self.start)
        self.phase3 = SineWave(frequency=self.frequency, amplitude=self.amplitude[2], phase_shift=self.phase_shift[2], duration=self.duration, start=self.start)
        self.dataset_pmu = None
        self.generate_dataset()

    def generate_dataset(self):
        """Genera un dataframe segun la norma IEEE c37.118.2"""
        headers = ['TimeStamp', 'Trash', 'Phase1', 'V1', 'Phase2', 'V2', 'Phase3', 'V3', 'Freq', 'dFreq/dt', 'SOC']
        self.dataset_pmu = pd.DataFrame(columns=headers)
        #self.dataset_pmu['TimeStamp'] = [datetime.utcfromtimestamp(a).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for a in self.phase1.t]
        self.dataset_pmu['TimeStamp'] = [datetime.utcfromtimestamp(a) for a in self.phase1.t]
        #self.dataset_pmu['TimeStamp'] = self.phase1.t
        self.dataset_pmu['Trash'] = np.zeros(len(self.phase1.t))
        self.dataset_pmu['Phase1'] = self.phase1.phasors_phase
        self.dataset_pmu['V1'] = self.phase1.phasors_v 
        self.dataset_pmu['Phase2'] = self.phase2.phasors_phase
        self.dataset_pmu['V2'] = self.phase2.phasors_v
        self.dataset_pmu['Phase3'] = self.phase3.phasors_phase
        self.dataset_pmu['V3'] = self.phase3.phasors_v
        self.dataset_pmu['Freq'] = np.mean(np.array([self.phase1.phasors_f, self.phase2.phasors_f, self.phase3.phasors_f]), axis=0)
        self.dataset_pmu['Trash'] = np.zeros(len(self.phase1.t))
        self.dataset_pmu['dFreq/dt'] = np.zeros(len(self.phase1.t))
        self.dataset_pmu['SOC'] = np.zeros(len(self.phase1.t))        
        self.dataset_pmu.set_index('TimeStamp', drop=False, inplace=True)

    def create_csv(self, file_name):
        """Crea un archivo csv con la informacion del dataframe generado.
           Lo guarda con el nombre file_name
        """
        self.dataset_pmu.to_csv(file_name, sep=';', index=False, header=False)

    def return_dataset(self, i):
        return self.dataset_pmu.iloc[i]

    def plot_time_domain_signal(self):
        """Grafica la señal en el dominio del tiempo"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle('Señal trifásica aleatoria', fontsize=14)
        # ax1.title.set_text('Amplitud')
        ax1.title.set_text('Señal real')
        ax1.plot(self.dataset_pmu['TimeStamp'], self.phase1.generate_wave(), label='Fase 1', color='red')
        ax1.plot(self.dataset_pmu['TimeStamp'], self.phase2.generate_wave(), label='Fase 2', color='blue')
        ax1.plot(self.dataset_pmu['TimeStamp'], self.phase3.generate_wave(), label='Fase 3', color='green')
        ax1.set_ylabel('Tensión[V]')
        
        ax2.title.set_text('Representación fasorial')
        ax2.plot(self.dataset_pmu['TimeStamp'], self.phase1.phasors_v, label='Fase 1', color='red')
        ax2.plot(self.dataset_pmu['TimeStamp'], self.phase2.phasors_v, label='Fase 2', color='blue')
        ax2.plot(self.dataset_pmu['TimeStamp'], self.phase3.phasors_v, label='Fase 3', color='green')
        ax2.set_ylabel('Amplitud[V]')
        
        ax3.plot(self.dataset_pmu['TimeStamp'], self.phase1.phasors_phase, label='Fase 1', color='red')
        ax3.plot(self.dataset_pmu['TimeStamp'], self.phase2.phasors_phase, label='Fase 2', color='blue')
        ax3.plot(self.dataset_pmu['TimeStamp'], self.phase3.phasors_phase, label='Fase 3', color='green')
        ax3.set_ylabel('Fase[°]')

        ax3.set_xlabel('Tiempo[S]')      
        
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S.%f')
        ax1.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_formatter(xfmt) 
        plt.xticks( rotation=12 )
        plt.show()
    
    def plot_frequency_domain_signal(self):
        """Grafica la señal en el dominio de la frecuencia"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle('Señal trifásica aleatoria', fontsize=14)
        ph1, w1 = self.phase1.fft()
        ph2, w2 = self.phase2.fft()
        ph3, w3 = self.phase3.fft()
        ax1.plot(w1, np.abs(ph1), label='Fase 1', color='red')
        ax1.legend(loc="upper right")
        ax2.plot(w2, np.abs(ph2), label='Fase 2', color='blue')
        ax2.set_ylabel('Tensión eficaz [V]')
        ax2.legend(loc="upper right")
        ax3.plot(w3, np.abs(ph3), label='Fase 3', color='green')
        ax3.set_xlabel('Frecuencia [Hz]')
        ax3.legend(loc="upper right")
        plt.show()   


if __name__ == '__main__':
    ts = datetime.now().timestamp()
    signal_0 = ThreePhaseSineWave(frequency=10, amplitude=(60, 60, 60), phase_shift=(0,120,240), duration=0.3, start=ts)
    #signal_1 = ThreePhaseSineWave(frequency=10, amplitude=(0, 0, 0), phase_shift=(0,120,240), duration=0.7, start=ts+0.3)
    #signal_2 = ThreePhaseSineWave(frequency=10, amplitude=(60, 60, 60), phase_shift=(0,120,240), duration=1, start=ts+1)
    #signal_3 = ThreePhaseSineWave(frequency=10, amplitude=(60, 60, 60), phase_shift=(0,180,240), duration=1, start=ts+2)

    #simulated_signal =  pd.concat([signal_0.dataset_pmu, signal_1.dataset_pmu, signal_2.dataset_pmu, signal_3.dataset_pmu])
    #simulated_signal.to_csv('simulated_signal.csv', sep=';', index=False, header=False)

    signal_0.plot_time_domain_signal()
    