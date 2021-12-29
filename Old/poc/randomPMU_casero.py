#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
sim_mercado.py toma datos de "fruta", precio, volumen 
del archivo mcentral.csv
Genera un log de movimientos en mercadolog.csv
'''
import time
import csv
from customPMU import ThreePhaseSineWave

def escribir_log(filename):
    '''
    Escribe items de prod al azar con valores cambiados
    en filename 1 vez por segundo
    '''
    f= open(filename,'w')
    count = 0
    try:
        while True:
            signal = ThreePhaseSineWave(frequency=50, amplitude=(60, 60, 60), phase_shift=(0, 180, 240), duration=0.2, start=count)
            for i, s in signal.dataset_pmu.iterrows(): 
                #line = signal.return_dataset(count)
                wr = csv.writer(f, delimiter=';', lineterminator='\n')
                wr.writerow(s)
                f.flush()
                #print(f'new row = {s}')
                time.sleep(0.1)
            count += 1
            continue
    except KeyboardInterrupt:
        f.close()

escribir_log('../data/PMU_casero.csv')

