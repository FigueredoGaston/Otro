import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


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


def generate_file_2(path):
    '''Abre un archivo y lo devuelve como un dataframe.
    Pre: se debe pasar su nombre de ruta de acceso.
    Pos: se devuelve un dataframe.
    '''

    headers = ['TimeStamp', 'Trash', 'V1', 'Phase1', 'V2', 'Phase2', 'V3', 'Phase3', 'Freq', 'dFreq/dt', 'SOC']
    types = [str, str, float, float, float, float, float, float, float, str, str]
    dtypes = {k: t for k, t in zip(headers, types)}
    pmu_df = pd.read_csv(path, engine='python', names=headers, sep=',', dtype=dtypes, comment='T').dropna(how='all')
    # Uso el parámetro comment='T' y el dropna(how='all') para que ignore la línea que tiene texto,
    # deduzco que es el encabezado así que siempre empieza con T.
    #pmu_df['SOC_date'] = pmu_df.apply(lambda x: timestamp_to_date(x['SOC'], prefijo='nano'), axis=1)
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


def plot_PMU_2(pmu_df1, pmu_df2):
    '''Genera un grafico de un PMU.

    Pre: se debe pasar un dataframe.
    Pos: se devuelve un grafico.
    '''

    plt.subplot(1, 2, 1)
    plt.title('CSV generado con simulador')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Tensión [V]')
    phase_1 = generate_phase(pmu_df1, 1)
    phase_2 = generate_phase(pmu_df1, 2)
    phase_3 = generate_phase(pmu_df1, 3)
    plt.plot(phase_1['t'], phase_1['phase_1_signal'], label='Fase 1', color='red')
    plt.plot(phase_2['t'], phase_2['phase_2_signal'], label='Fase 2', color='green')
    plt.plot(phase_3['t'], phase_3['phase_3_signal'], label='Fase 3', color='blue')
    plt.subplot(1, 2, 2)
    plt.title('CSV que pasó Luciano')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Tensión [V]')
    phase_1 = generate_phase(pmu_df2, 1)
    phase_2 = generate_phase(pmu_df2, 2)
    phase_3 = generate_phase(pmu_df2, 3)
    plt.plot(phase_1['t'], phase_1['phase_1_signal'], label='Fase 1', color='red')
    plt.plot(phase_2['t'], phase_2['phase_2_signal'], label='Fase 2', color='green')
    plt.plot(phase_3['t'], phase_3['phase_3_signal'], label='Fase 3', color='blue')
    # plt.savefig(f'trifasica_completa.png')
    plt.show()


def plot_PMU(pmu_df):
    '''Genera un grafico de un PMU.

    Pre: se debe pasar un dataframe.
    Pos: se devuelve un grafico.
    '''
    phase_1 = generate_phase(pmu_df, 1)
    phase_2 = generate_phase(pmu_df, 2)
    phase_3 = generate_phase(pmu_df, 3)
    ax = phase_1.plot.line('t', 'phase_1_signal', c='green')
    phase_2.plot.line('t', 'phase_2_signal', c='red', ax=ax)
    phase_3.plot.line('t', 'phase_3_signal', c='blue', ax=ax)
    # plt.savefig(f'trifasica_completa.png')
    plt.show()


if __name__ == '__main__':
    path_2 = '../pypmu ejemplos/random_data.csv'
    path = '../data/datasetPMU2.csv'
    ''' pmu_df = generate_file_2(path_2)
    plot_PMU(pmu_df) '''
    plot_PMU_2(generate_file_2(path_2), generate_file(path))
