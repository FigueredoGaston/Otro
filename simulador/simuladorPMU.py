#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File Name: simuladorPMU.py
# Created By: Hernan Mogliasso
# Created Date: 23/11/2021
# Description: PMU customizado a las necesidades del proyecto
# =============================================================================

from datetime import datetime
from sine import ThreePhaseSineWave
import pandas as pd

if __name__ == '__main__':
    ts = datetime.now().timestamp()
    signal_0 = ThreePhaseSineWave(frequency=50, amplitude=(60, 60, 60), phase_shift=(0,120,240), duration=0.1, start=ts)
    signal_1 = ThreePhaseSineWave(frequency=50, amplitude=(60, 60, 60), phase_shift=(0,120,240), duration=0.2, start=ts+0.1)
    signal_2 = ThreePhaseSineWave(frequency=50, amplitude=(60, 60, 60), phase_shift=(0,120,240), duration=0.7, start=ts+0.3)
    signal_3 = ThreePhaseSineWave(frequency=50, amplitude=(60, 0, 60), phase_shift=(0,120,240), duration=1, start=ts+1)
    signal_4 = ThreePhaseSineWave(frequency=50, amplitude=(60, 60, 60), phase_shift=(0,180,240), duration=1, start=ts+2)

    simulated_signal = pd.concat([signal_0.dataset_pmu, signal_1.dataset_pmu, signal_2.dataset_pmu, signal_3.dataset_pmu])
    simulated_signal.to_csv('../data/simulated_signal.csv', sep=';', index=False, header=False)
    signal_0.plot_time_domain_signal()
    signal_1.plot_time_domain_signal()
    signal_2.plot_time_domain_signal()
    signal_3.plot_time_domain_signal()
    signal_4.plot_time_domain_signal()