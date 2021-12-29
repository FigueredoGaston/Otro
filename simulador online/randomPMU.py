#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
from sine import ThreePhaseSineWave

def escribir_log(filename):
    '''
    Genera una señal de 3 fases con una frecuencia y una amplitud
    y la escribe en un archivo csv.
    '''
    f= open(filename,'w')
    count = 0
    try:
        while True:
            signal = ThreePhaseSineWave(frequency=50, amplitude=(60, 60, 60),
                                        phase_shift=(0, 120, 240), duration=0.2, start=count)
            for i, s in signal.dataset_pmu.iterrows():
                wr = csv.writer(f, delimiter=';', lineterminator='\n')
                wr.writerow(s)
                f.flush()
                time.sleep(0.5)
            count += 1
            continue
    except KeyboardInterrupt:
        f.close()

if __name__ == '__main__':
    escribir_log('../data/PMU_live.csv')

