import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def instant_value(x):
    w = 2*np.pi*x['Freq']*x['t'] # Frecuencia angular
    argumento = w + x['phase_rad']
    return x['V'] * np.cos(argumento) 

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


def grados_a_radianes(grados):
    return grados*np.pi/180


data_path = os.path.join('..', 'data', 'simulated_signal.csv')
headers = ['TimeStamp', 'Trash', 'Phase1', 'V1', 'Phase2', 'V2', 'Phase3', 'V3', 'Freq', 'dFreq/dt', 'SOC']
types = [str, int, float, float, float, float, float, float, float, float, np.int64]
dtypes = {k: t for k, t in zip(headers, types)}

sim_pmu_df = pd.read_csv(data_path, names=headers, sep=';', dtype=dtypes)

time_0 = datetime.strptime(sim_pmu_df.iloc[0]['TimeStamp'], '%Y-%m-%d %H:%M:%S.%f')
time_1 = datetime.strptime(sim_pmu_df.iloc[1]['TimeStamp'], '%Y-%m-%d %H:%M:%S.%f')
step = (time_1 - time_0).total_seconds()

# calculos de la fase 1
phase1 = sim_pmu_df[['TimeStamp', 'Phase1', 'V1', 'Freq']]
phase1.rename(columns={"Phase1": "Phase", "V1": "V"}, inplace=True)
phase1.loc[:,'t'] = np.arange(start=0, stop=phase1.shape[0]*step, step=step ) 
phase1.loc[:,'phase_rad'] = phase1['Phase'].apply(lambda x: grados_a_radianes(x))
phase1.loc[:, 'phase_1'] = phase1.apply(lambda x: instant_value(x), axis=1)
ax = phase1.plot.line('TimeStamp', 'phase_1', c = 'red')
ax.title.set_text('Señal reconstruída')
ax.set_ylabel('Tensión[V]')
ax.set_xlabel('Fecha y Hora')

# calculos de la fase 2

phase2 = sim_pmu_df[['TimeStamp', 'Phase2', 'V2', 'Freq']]
phase2.rename(columns={"Phase2": "Phase", "V2": "V"}, inplace=True)
phase2.loc[:,'t'] = np.arange(start=0, stop=phase1.shape[0]*step, step=step ) 
phase2.loc[:,'phase_rad'] = phase2['Phase'].apply(lambda x: grados_a_radianes(x))
phase2.loc[:, 'phase_2'] = phase2.apply(lambda x: instant_value(x), axis=1)
phase2.plot.line('TimeStamp', 'phase_2', c = 'green', ax=ax)

# calculos de la fase 3

phase3 = sim_pmu_df[['TimeStamp', 'Phase3', 'V3', 'Freq']]
phase3.rename(columns={"Phase3": "Phase", "V3": "V"}, inplace=True)
phase3.loc[:,'t'] = np.arange(start=0, stop=phase1.shape[0]*step, step=step ) 
phase3.loc[:,'phase_rad'] = phase3['Phase'].apply(lambda x: grados_a_radianes(x))
phase3.loc[:, 'phase_3'] = phase3.apply(lambda x: instant_value(x), axis=1)
phase3.plot.line('TimeStamp', 'phase_3', c = 'blue', ax=ax)

plt.xticks( rotation=12 )
plt.show()