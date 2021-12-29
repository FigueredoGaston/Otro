# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:42:05 2021

@author: Monoramiro
"""

import csv
from synchrophasor.pdc import Pdc
from synchrophasor.frame import DataFrame
from datetime import datetime, timedelta


"""
PDC se conecta con el PMU y recibe las mediciones 
"""

def obtener_datos(pdc, cant = 10):
    '''
    Recibe los datos que envía el PMU y los convierte en diccionario
    '''
    for _ in range(cant):
        data = pdc.get()  # Keep receiving data
        if type(data) == DataFrame:
            new_dict = dict()
            reading = data.get_measurements() 
            new_dict['TimeStamp'] = reading['time']
            new_dict['Flag'] = reading['measurements'][0]['stat']
            new_dict['V1'], new_dict['Phase1'] = reading['measurements'][0]['phasors'][0]
            new_dict['V2'], new_dict['Phase2'] = reading['measurements'][0]['phasors'][1]
            new_dict['V3'], new_dict['Phase3'] = reading['measurements'][0]['phasors'][2]
            new_dict['Freq'] =  reading['measurements'][0]['frequency']
            new_dict['dFreq/dt'] = 'UNK'
            new_dict['SOC'] = 'UNK'
            yield new_dict

 
def timestamp_to_date(timestamp, prefijo=None):
    """
    Transforma una estampa Unix time en un datetime.
    Un timestamp es un entero que establece la cantidad de
    segundos desde el 01/01/1970.
    El unix Time puede estar en mili, micro o nano segundos.
    prefijo debe concordar con el prefijo del timestamp a convertir.
    """
    prefijos = {'mili': 10**3, 'micro': 10**6, 'nano': 10**9, }
    if prefijo and prefijo in prefijos:
        segundos = timestamp/prefijos[prefijo]
    else:
        segundos = timestamp
    return datetime(1970, 1, 1) + timedelta(seconds=segundos)


def generate_csv(file):
    '''
    Crea el archivo CSV del que el graficador en tiempo real va a tomar los datos
    '''
    headers = ['TimeStamp', 'Flag', 'Phase1', 'V1', 'Phase2', 'V2', 'Phase3', 
               'V3', 'Freq', 'dFreq/dt', 'SOC']
      
    with open(file, 'w',newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
        csv_writer.writeheader()
    return file


def charge_csv(file,data):
    '''
    Va cargando en el CSV los datos que envía el PMU
    '''

    headers = ['TimeStamp', 'Flag', 'Phase1', 'V1', 'Phase2', 'V2', 'Phase3', 'V3', 'Freq', 'dFreq/dt', 'SOC']
    with open(file, 'a', newline='') as csv_file:
        
        wr = csv.DictWriter(csv_file, fieldnames=headers)
        
        #recibe el tiepo Unix y lo convierte a TimeStamp
        unix_time= data['TimeStamp']
        data['TimeStamp']=timestamp_to_date(unix_time)
        
        wr.writerow(data)

if __name__ == "__main__":

    #Configuración para recibir los datos del PMU
    pdc = Pdc(pdc_id=7, pmu_ip="127.0.0.1", pmu_port=1410)
    pdc.logger.setLevel("DEBUG")

    pdc.run()  # Connect to PMU

    header = pdc.get_header()  # Obtiene el header message desde el PMU
    config = pdc.get_config()  # Obtiene configuración desde el PMU

    pdc.start()  # Pide al PMU que comience a enviar las mediciones

    generate_csv('../data/data.csv')
    
    nsincrof=40    #Configuramos la cantidad de sincrofasores a enviar
    for a in obtener_datos(pdc, nsincrof):
        charge_csv('../data/data.csv',a)
                      
    pdc.quit()  # Cierra la conexión
