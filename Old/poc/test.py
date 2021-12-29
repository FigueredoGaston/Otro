from customPMU import ThreePhaseSineWave


s = ThreePhaseSineWave(frequency=100, amplitude=(60, 60, 60), phase_shift=(0,120,240), duration=0.2, start=0)

s.plot_time_domain_signal()
s.plot_frequency_domain_signal()