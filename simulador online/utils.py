#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
    Herramientas para el proyecto de simulación de PMU.
 '''
import pandas as pd
import numpy as np
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