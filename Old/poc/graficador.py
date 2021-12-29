#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta


def instant_value(x):
    """Calcula el valor instantaneo de la señal representada por fasores.
    """
    w = 2*np.pi*x['Freq']*x['t']  # Frecuencia angular
    argumento = w + x['phase_rad']
    return x['V'] * np.cos(argumento)


def timestamp_to_date(timestamp, prefijo=None):
    """
    Transforma un timestamp en un datetime.
    Un timestamp es un entero que establece la cantidad de
    segundos desde el 01/01/1970.
    timestamp puede estar en mili, micro o nano segundos.
    prefijo debe concordar con el prefijo del timestamp a convertir.
    """
    prefijos = {'mili': 10**3, 'micro': 10**6, 'nano': 10**9, }
    if prefijo and prefijo in prefijos:
        segundos = timestamp/prefijos[prefijo]
    else:
        segundos = timestamp
    return datetime(1970, 1, 1) + timedelta(seconds=segundos)


def grados_a_radianes(grados):
    "Convierte grados en radianes"
    return grados*np.pi/180

def generate_file(path):
    '''Abre un archivo y lo devuelve como un dataframe.
    Pre: se debe pasar su nombre de ruta de acceso.
    Pos: se devuelve un dataframe.
    '''

    headers = ['TimeStamp', 'Trash', 'Phase1', 'V1', 'Phase2', 'V2', 'Phase3', 'V3', 'Freq', 'dFreq/dt', 'SOC']
    types = [str, bool, float, float, float, float, float, float, float, float, np.int64]
    dtypes = {k: t for k, t in zip(headers, types)}
    pmu_df = pd.read_csv(path, engine='python', names=headers, sep=';', dtype=dtypes, comment='T').dropna(how='all')
    # Uso el parámetro comment='T' y el dropna(how='all') para que ignore la línea que tiene texto,
    # deduzco que es el encabezado así que siempre empieza con T.
    pmu_df['SOC_date'] = pmu_df.apply(lambda x: timestamp_to_date(x['SOC'], prefijo='nano'), axis=1)
    return pmu_df

def generate_phase(pmu_df, phase):
    '''Genera una fase de un PMU.

    Pre: se debe pasar un dataframe y una fase.
    Pos: se devuelve un dataframe con la fase.
    '''

    header_phase = 'Phase{}'.format(phase)
    header_voltage = 'V{}'.format(phase)
    phase_df = pmu_df[['TimeStamp', header_phase, header_voltage, 'Freq']].copy()
    phase_df.rename(columns={header_phase: 'Phase', header_voltage: 'V'}, inplace=True)
    phase_df.loc[:, 'phase_rad'] = phase_df['Phase'].apply(lambda x: grados_a_radianes(x)).copy()
    m = phase_df.shape[0] * 0.001 + 0.001
    phase_df.loc[:, 't'] = np.arange(0.001, m, 0.001)
    phase_df.loc[:, f'phase_{phase}_signal'] = phase_df.apply(lambda x: instant_value(x), axis=1)
    return phase_df



plt.style.use('seaborn')
fig = plt.figure()
ax = fig.add_subplot(2,1,1, xlim=(0,10), ylim=(-1,1))
ax2 = fig.add_subplot(2,1,2, xlim=(0,10), ylim=(-1,1))
#data = generate_file('../data/datasetPMU1.csv')
data = generate_file('../data/PMU_casero.csv')
phase_1 = generate_phase(data, 1)
phase_2 = generate_phase(data, 2)
phase_3 = generate_phase(data, 3)

def anima(i):
    x = []
    y = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    
    x = phase_1[0:i]['t']
    y = phase_1[0:i]['V']
    
    x2 = phase_2[0:i]['t']
    y2 = phase_2[0:i]['V']
    
    x3 = phase_3[0:i]['t']
    y3 = phase_3[0:i]['V']

    ax.clear()
    ax.plot(x, y)
    ax.plot(x2, y2)
    ax.plot(x3, y3)
    ax.title.set_text('Tiempo vs Señal')
    ax.set_xlabel('TimeStamp')
    ax.set_ylabel('Tensión[V]')

    x = phase_1[0:i]['TimeStamp']
    y = phase_1[0:i]['phase_1_signal']
    x2 = phase_2[0:i]['TimeStamp']
    y2 = phase_2[0:i]['phase_2_signal']
    x3 = phase_3[0:i]['TimeStamp']
    y3 = phase_3[0:i]['phase_3_signal']
    ax2.clear()
    ax2.plot(x, y)
    ax2.plot(x2, y2)
    ax2.plot(x3, y3)
    ax2.title.set_text('Tiempo vs Señal')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Tensión[V]')
    #ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


anim = animation.FuncAnimation(fig, func=anima, interval=50)
fig.tight_layout()
plt.show()