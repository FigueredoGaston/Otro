import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

class phasor:
    def __init__(self, amp, phase, freq, t) -> None:
        self.amplitude = amp
        self.phase = phase
        self.frequency = freq
        self.t = t

    def w(self): 
        "Angular frequency"
        return 2*math.pi*self.frequency
    
    def phase_rad(self):
        "Phase in radians form"
        return self.phase*math.pi/180
    
    def instant_value(self):
        "Recover time domain function"
        return self.amplitude * math.cos(self.w()*self.t + self.phase_rad()) 


class ThreePhasePhasor:
    def __init__(self, p1, p2, p3) -> None:
        self.phasor1 = p1
        self.phasor2 = p2
        self.phasor3 = p3
        self.ideal_dif = 120
        self.warning_phase = 5 # 5%

    def phase_difference_1(self):
        return abs(self.phasor1.phase - self.phasor3.phase)

    def phase_difference_2(self):
        return abs(self.phasor2.phase - self.phasor3.phase)

    def phase_alarm(self):
        min_dif = self.ideal_dif * (1 - self.warning_phase/100)
        max_dif = self.ideal_dif * (1 + self.warning_phase/100)
        if (self.phase_difference_1() < min_dif) or (self.phase_difference_1() > max_dif):
            return True
        if (self.phase_difference_2() < min_dif) or (self.phase_difference_2() > max_dif):
            return True
        return False

class Signal:
    """Clase Signal Iterable
       Es un contenedor de objetos tipo ThreePhasePhasor
    """
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return self.samples.__iter__()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index): 
        return self.samples[index]

    def time_domain_signal(self):
        """
        Devuelve el costo total del camion.
        """
        time_signal_1 = np.array([s.phasor1.instant_value() for i, s in enumerate(self.samples)])
        time_signal_2 = np.array([s.phasor2.instant_value() for i, s in enumerate(self.samples)])
        time_signal_3 = np.array([s.phasor3.instant_value() for i, s in enumerate(self.samples)])
        time = np.array([s.phasor1.t for s in self.samples])
        return time_signal_1, time_signal_2, time_signal_3, time
    
    def plot_time_domain_signal(self):
        time_signal_1, time_signal_2, time_signal_3, t = self.time_domain_signal()
        plt.plot(t, time_signal_1, label='Fase 1', color='red')
        plt.plot(t, time_signal_2, label='Fase 2', color='green')
        plt.plot(t, time_signal_3, label='Fase 3', color='blue')
        plt.show()


if __name__ == '__main__':
    data_path = os.path.join('custom_dataset_sin_desfasaje.csv')
    headers = ['TimeStamp', 'Trash', 'Phase1', 'V1', 'Phase2', 'V2', 'Phase3', 'V3', 'Freq', 'dFreq/dt', 'SOC']
    types = [float, bool, float, float, float, float, float, float, float, float, np.int64]
    dtypes = {k: t for k, t in zip(headers, types)}
    data_df = pd.read_csv(data_path, names=headers, sep=';', dtype=dtypes, comment='T').dropna(how='all')
    phasors_list = []

    for i, r in data_df.iterrows():
        p1 = phasor(amp=r['V1'], phase=r['Phase1'], freq=r['Freq'], t=r['TimeStamp'])
        p2 = phasor(amp=r['V2'], phase=r['Phase2'], freq=r['Freq'], t=r['TimeStamp'])
        p3 = phasor(amp=r['V3'], phase=r['Phase3'], freq=r['Freq'], t=r['TimeStamp'])
        phasors_list.append(ThreePhasePhasor(p1, p2, p3))

    signal = Signal(phasors_list)
    signal.plot_time_domain_signal()


