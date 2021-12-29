from synchrophasor.pdc import Pdc
from synchrophasor.frame import DataFrame
import pandas as pd

"""
tinyPDC will connect to pmu_ip:pmu_port and send request
for header message, configuration and eventually
to start sending measurements.
"""

def obtener_datos(pdc, cant = 10):
    
    for _ in range(cant):
        data = pdc.get()  # Keep receiving data
        if type(data) == DataFrame:
            new_dict = dict()
            reading = data.get_measurements() 
            new_dict['TimeStamp'] = reading['time']
            new_dict['Trash'] = reading['measurements'][0]['stat']
            new_dict['V1'], new_dict['Phase1'] = reading['measurements'][0]['phasors'][0]
            new_dict['V2'], new_dict['Phase2'] = reading['measurements'][0]['phasors'][1]
            new_dict['V3'], new_dict['Phase3'] = reading['measurements'][0]['phasors'][2]
            new_dict['Freq'] =  reading['measurements'][0]['frequency']
            new_dict['dFreq/dt'] = 'UNK'
            new_dict['SOC'] = 'UNK'
            yield new_dict


if __name__ == "__main__":

    pdc = Pdc(pdc_id=7, pmu_ip="127.0.0.1", pmu_port=1410)
    pdc.logger.setLevel("DEBUG")

    pdc.run()  # Connect to PMU

    header = pdc.get_header()  # Get header message from PMU
    config = pdc.get_config()  # Get configuration from PMU

    pdc.start()  # Request to start sending measurements

    saved_data = []

    for a in obtener_datos(pdc, 300):
        saved_data.append(a)
    pdc.quit()  # Close connection

    df = pd.DataFrame(saved_data)
    df.to_csv('random_data.csv')
    
