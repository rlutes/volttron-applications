import sys
import logging
from datetime import timedelta as td, date
from copy import deepcopy
import pkgutil
import os
import itertools
import csv
import inspect
import numpy as np
from scipy.signal import butter, filtfilt
from dateutil.parser import parse

from volttron.platform.messaging import topics
from volttron.platform.agent import utils
from volttron.platform.agent.math_utils import mean, stdev
from volttron.platform.vip.agent import Agent, Core
plot_loader = pkgutil.find_loader('matplotlib')
plotter_found = plot_loader is not None
if plotter_found:
    import matplotlib.pyplot as plt

cutoff = 300
fs = 3000

utils.setup_logging()
_log = logging.getLogger(__name__)
__version__ = '1.0.0'

__author1__ = 'Woohyun Kim <woohyun.kim@pnnl.gov>'
__author2__ = 'Robert Lutes <robert.lutes@pnnl.gov>'
__copyright__ = 'Copyright (c) 2016, Battelle Memorial Institute'
__license__ = 'FreeBSD'
DATE_FORMAT = '%m-%d-%y %H:%M'


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def find_intersections(m1, m2, std1, std2):
    a = 1./(2.*std1**2) - 1./(2.*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a, b, c])

def locate_min_max(*args):
    filtered_timeseries = butter_lowpass_filtfilt(args[0], cutoff, fs)
    
    maximums = detect_peaks(filtered_timeseries, args[1], mpd=10, valley=False)
    minimums = detect_peaks(filtered_timeseries, args[1], mpd=10, valley=True)
    _log.debug("WOBAH0: {}   ------  {} ".format(minimums, maximums))
    return minimums, maximums, filtered_timeseries


def align_pv(zone_temperature_array, peak_ind, val_ind, dtime):
    '''align_pv takes the indices of peaks (peak_ind) and indices of

    valleys (val_ind) and ensures that there is only one valley
    in-between two consecutive peaks and only one peak between two
    consecutive valleys.  If there are two or more peaks between
    valleys the largest value is kept.  If there are two or more
    valleys between two peaks then the smallest value is kept.
    '''
    try:
        reckon = 0
        aligned = False
        find_peak = True if peak_ind[0] < val_ind[0] else False
        begin = 0
        while not aligned:
            if find_peak:
                while peak_ind[reckon+1] < val_ind[reckon+begin]:
                    if zone_temperature_array[peak_ind[reckon]] > zone_temperature_array[peak_ind[reckon+1]]:
                        peak_ind = np.delete(peak_ind, reckon+1)
                    else:
                        peak_ind = np.delete(peak_ind, reckon)
                if (dtime[val_ind[reckon+begin]] - dtime[peak_ind[reckon]]) <= td(minutes=5):
                    val_ind = np.delete(val_ind, reckon+begin)
                    peak_ind = np.delete(peak_ind, reckon+1)
                else:
                    find_peak = False
                    begin += 1
                    if begin > 1:
                        begin = 0
                        reckon += 1
            else:
                while val_ind[reckon + 1] < peak_ind[reckon+begin]:
                    if zone_temperature_array[val_ind[reckon]] > zone_temperature_array[val_ind[reckon+1]]:
                        val_ind = np.delete(val_ind, reckon)
                    else:
                        val_ind = np.delete(val_ind, reckon+1)
                if (dtime[peak_ind[reckon+begin]] - dtime[val_ind[reckon]]) <= td(minutes=5):
                    val_ind = np.delete(val_ind, reckon+1)
                    peak_ind = np.delete(peak_ind, reckon+begin)
                else:
                    find_peak = True
                    begin += 1
                    if begin > 1:
                        begin = 0
                        reckon += 1
            if (reckon+1) == min(val_ind.size, peak_ind.size):
                aligned = True
        if peak_ind.size > val_ind.size:
            peak_ind = np.resize(peak_ind, val_ind.size)
        elif val_ind.size > peak_ind.size:
            val_ind = np.resize(val_ind, peak_ind.size)
        return peak_ind, val_ind
    except:
        return np.empty(0), np.empty(0)



def detect_peaks(data, mph=None, threshold=0.2, mpd=1, edge='rising',
                 kpsh=False, valley=False, ax=None):
    '''
    Detect peaks in data based on their amplitude and other features.
    Original source for detect_peaks function can be obtained at:
    https://github.com/demotu/BMC/blob/master/functions/detect_peaks.py

    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"

    Copyright (c) 2013 Marcos Duarte
    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
    '''
    data = np.array(data)
    if data.size < 3:
        return np.array([], dtype=int)
    if mph is not None:
        mph = np.array(mph)
    if valley:
        data = -data
        mph = -mph if mph is not None else None
    
    # find indices of all peaks
    dx = data[1:] - data[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(data))[0]
    if indnan.size:
        data[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)

    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))

    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of data cannot be peaks

    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == data.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None :
        ind = ind[data[ind] > mph[ind]]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([data[ind]-data[ind-1], data[ind]-data[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(data[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (data[ind[i]] > data[ind] if kpsh else True)
                idel[i] = 0  # Keep currentent peak
        # remove the small peaks and sort back
        # the indices by their occurrentence
        ind = np.sort(ind[~idel])
    return ind


class SccCyclingDiagnostic(Agent):
    '''VOLTTRON Compressor Cycling diagnostic agent.'''
    def __init__(self, config_path, **kwargs):
        super(SccCyclingDiagnostic, self).__init__(**kwargs)
        config = utils.load_config(config_path)

        self.device_topic = topics.DEVICES_VALUE(campus=config.get('campus', ''),
                                                 building=config.get('building', ''),
                                                 unit=config.get('unit', ''),
                                                 path='',
                                                 point='all')
        self.check_time = config.get('analysis-run-interval')
        self.compressor_name = config.get('CompressorStatus')
        self.fanstatus_name = config.get('FanStatus')
        self.zonetemperature_name = config.get('ZoneTemperature')
        self.zonetemperature_stpt_name = config.get('ZoneTemperatureSetPoint')
        self.minimum_data_count = config.get('minimum_data_count', 5)
        self.monitored_data = [self.compressor_name, self.fanstatus_name,
                               self.zonetemperature_name, self.zonetemperature_stpt_name]

        # Initialize data arrays
        self.zone_temperature_array = []
        self.zone_temperature_stpt_array = []
        self.compressor_status_array = []
        self.timestamp_array = []

        self.startup = True
        self.mode = None
        self.available_data_points = []
        self.inconsistent_data_flag = 0
        self.last_state = None
        self.last_time = None
        self.intervals = 1
        self.file = 1
        self.file_sp = 1

    def initialize(self):
        self.zone_temperature_array = []
        self.zone_temperature_stpt_array = []
        self.compressor_status_array = []
        self.timestamp_array = []
        self.last_state = None
        self.last_time = None
        self.startup = True
        self.mode = None

    @Core.receiver('onstart')
    def starting_base(self, sender, **kwargs):
        self.vip.pubsub.subscribe(peer='pubsub',
                                  prefix=self.device_topic,
                                  callback=self.on_new_data)

    def determine_mode(self, available):
        '''Determine diagnostic algorithm based on available data.

        Minimum data requirement is zone temperature and supply fan status.
        '''
        available_and_needed = [item for item in self.monitored_data if item in available]
        if self.compressor_name in available_and_needed and self.fanstatus_name in available_and_needed:
            self.mode = 1
        elif self.zonetemperature_name in available_and_needed and self.zonetemperature_stpt_name in available_and_needed and self.fanstatus_name in available_and_needed:
            self.mode = 2
        elif self.zonetemperature_name in available_and_needed and self.fanstatus_name in available_and_needed:
            self.mode = 3
        else:
            self.mode = 4
        self.available_data_points = available_and_needed
        return

    def on_new_data(self, peer, sender, bus, topic, headers, message):
        data = message[0]
        available_data = data.keys()
        if self.startup:
            self.determine_mode(available_data)
        if self.mode == 4:
            _log.info('Required data for diagnostic is not available or '
                      'configured names do not match published names!')
            return

        for data_point_name in self.available_data_points:
            if data_point_name not in available_data:
                self.inconsistent_data_flag += 1
                _log.info('Previously available data is missing from device publish.')
                if self.inconsistent_data_flag > 5:
                    _log.info('data fields available for device are not consistent. Reinitializing diagnostic.')
                    self.initialize()
                return

        self.inconsistent_data_flag = 0
        current_time = parse(headers['Date'])
        fan_status_data = int(data[self.fanstatus_name])
        if not fan_status_data:
            _log.debug('Supply fan is off.  Data for {} will not used'.format(current_time))
            return

        if self.mode == 1:
            compressor_data = int(data.get(self.compressor_name))
            self.operating_mode1(compressor_data, current_time)
        if self.mode == 2:
            zonetemp_data = float(data.get(self.zonetemperature_name))
            zonetemp_stpt_data = float(data.get(self.zonetemperature_stpt_name))
            self.operating_mode2(zonetemp_data, zonetemp_stpt_data, current_time)
        if self.mode == 3:
            zonetemp_data = float(data.get(self.zonetemperature_name))
            self.operating_mode3(zonetemp_data, current_time)
        return

    def operating_mode1(self, compressor_data, current_time):
        _log.debug('Running Cycling Dx Mode 1.')
        self.timestamp_array.append(current_time)
        self.compressor_status_array.append(compressor_data)
        iterate_on = len(self.compressor_status_array) - 1

        if self.timestamp_array[-1] - self.timestamp_array[0] >= td(minutes=self.check_time):
            on_indices = []
            off_indices = []
            for status in range(1, iterate_on):
                if self.compressor_status_array[status] and not self.compressor_status_array[status-1]:
                    on_indices.append(status)
                if not self.compressor_status_array[status] and self.compressor_status_array[status-1]:
                    off_indices.append(status)

            results = self.cycling_dx(on_indices, off_indices)
            self.output_cycling()
            self.shrink(self.compressor_status_array)
            _log.debug('CyclingDx results: {}'.format(results))

    def operating_mode2(self, zonetemperature_data, zonetemperature_stpt_data, current_time):
        _log.debug('Running Cycling Dx Mode 2.')
        self.timestamp_array.append(current_time)
        self.zone_temperature_array.append(zonetemperature_data)
        self.zone_temperature_stpt_array.append(zonetemperature_stpt_data)
        if self.timestamp_array[-1] - self.timestamp_array[0] >= td(minutes=self.check_time):
            minimums, maximums, filtered_timeseries = locate_min_max(self.zone_temperature_array, self.zone_temperature_stpt_array)
            self.results_handler(maximums, minimums, filtered_timeseries)

    def operating_mode3(self, zonetemperature_data, current_time):
        _log.debug('Running CyclingDx Mode 3.')
        self.timestamp_array.append(current_time)
        self.zone_temperature_array.append(zonetemperature_data)
        if self.timestamp_array[-1] - self.timestamp_array[0] >= td(minutes=self.check_time):
            valleys, peaks, filtered_timeseries = locate_min_max(self.zone_temperature_array, None)

            if np.prod(valleys.shape) < self.minimum_data_count or np.prod(peaks.shape) < self.minimum_data_count:
                _log.debug('Set point detection is inconclusive.  Not enough data.')
                self.shrink(self.zone_temperature_array)
                results = {
                    "cycles": 'INCONCLUSIVE',
                    "Avg On Cycle": "INCONCLUSIVE",
                    "Avg Off Cycle": "INCONCLUSIVE"
                }
                return
            peak_array, valley_array = align_pv(filtered_timeseries, peaks, valleys, self.timestamp_array)
            if np.prod(peak_array.shape) < self.minimum_data_count or np.prod(peak_array.shape) < self.minimum_data_count:
                _log.debug('Set point detection is inconclusive.  Not enough data.')
                self.shrink(self.zone_temperature_array)
                results = {
                    "cycles": 'INCONCLUSIVE',
                    "Avg On Cycle": "INCONCLUSIVE",
                    "Avg Off Cycle": "INCONCLUSIVE"
                }
                return
            
            self.zone_temperature_stpt_array = self.create_setpoint_array(deepcopy(peak_array), deepcopy(valley_array))
            self.output_sp()
            _log.debug("TOBAH -- SP: {}".format(self.zone_temperature_stpt_array))
            minimums, maximums, filtered_timeseries = locate_min_max(self.zone_temperature_array, self.zone_temperature_stpt_array)
            self.results_handler(maximums, minimums, filtered_timeseries)

    def shrink(self, array):
        self.timestamp_array = [item for item in self.timestamp_array if (item - self.timestamp_array[0]) >= td(minutes=self.check_time/4)]
        shrink = len(array) - len(self.timestamp_array)
        self.zone_temperature_array = self.zone_temperature_array[shrink:]

        self.zone_temperature_stpt_array = self.zone_temperature_stpt_array[shrink:]
        self.compressor_status_array = self.compressor_status_array[shrink:]

    def results_handler(self, maximums, minimums, filtered_timeseries):
        if np.prod(maximums.shape) < self.minimum_data_count or np.prod(minimums.shape) < self.minimum_data_count:
            _log.debug('Set point detection is inconclusive.  Not enough data.')
            self.shrink(self.zone_temperature_array)
            results = {
                "cycles": 'INCONCLUSIVE',
                "Avg On Cycle": "INCONCLUSIVE",
                "Avg Off Cycle": "INCONCLUSIVE"
            }
            return
        peak_array, valley_array = align_pv(filtered_timeseries, maximums, minimums, self.timestamp_array)
        _log.debug("WOBAH2 -- Valleys: {} --------- Peaks: {}".format(peak_array, valley_array))
        if np.prod(peak_array.shape) < self.minimum_data_count or np.prod(valley_array.shape) < self.minimum_data_count:
            _log.debug('Set point detection is inconclusive.  Not enough data.')
            self.shrink(self.zone_temperature_array)
            results = {
                "cycles": 'INCONCLUSIVE',
                "Avg On Cycle": "INCONCLUSIVE",
                "Avg Off Cycle": "INCONCLUSIVE"
            }
            return
        pcopy = deepcopy(peak_array)
        vcopy = deepcopy(valley_array)
        self.compressor_status_array = self.gen_status(pcopy, vcopy, self.timestamp_array)
        _log.debug("WOBAH3: {}".format(self.compressor_status_array))
        self.output_cycling()
        results = self.cycling_dx(pcopy, vcopy)
        _log.debug('Cycling diagnostic results: ' + str(results))
        self.shrink(self.zone_temperature_array)

    def create_setpoint_array(self, pcopy, vcopy):
        '''Creates setpoint array when zone temperature set point is not measured.'''
        peak_ts = zip([self.timestamp_array[ind] for ind in pcopy], [self.zone_temperature_array[ind] for ind in pcopy])
        valley_ts = zip([self.timestamp_array[ind] for ind in vcopy], [self.zone_temperature_array[ind] for ind in vcopy])

        remove_temp1 = [(x[0], x[1]) for x, y in zip(peak_ts, valley_ts) if x[1] >= y[1] + 0.25]
        remove_temp2 = [(y[0], y[1]) for x, y in zip(peak_ts, valley_ts) if x[1] >= y[1] + 0.25]
        
        peaks = [pcopy[x] for x in range(pcopy.size) if peak_ts[x][1] >= valley_ts[x][1] + 0.25]
        valleys = [vcopy[x] for x in range(vcopy.size) if peak_ts[x][1] >= valley_ts[x][1] + 0.25]

        peak_temp = [row[1] for row in remove_temp1]
        valley_temp = [row[1] for row in remove_temp2]

        setpoint_raw = [(peak_val + valley_val)/2 for peak_val, valley_val in zip(peak_temp, valley_temp)]
        peak_timestamp = [row[0] for row in remove_temp1]
        valley_timestamp = [row[0] for row in remove_temp2]

        indexer = 0
        zone_temperature_stpt = []

        current = valleys if peak_timestamp[0] < valley_timestamp[0] else peaks
        for ind in range(len(self.zone_temperature_array)):
            if ind <= current[indexer]:
                zone_temperature_stpt.append(setpoint_raw[indexer])
            elif indexer + 1 >= len(setpoint_raw):
                zone_temperature_stpt.append(setpoint_raw[-1])
                continue
            else:
                indexer += 1
                current = peaks if current == valleys else valleys
                zone_temperature_stpt.append(setpoint_raw[indexer])

        return zone_temperature_stp

    def cycling_dx(self, on_indices, off_indices):
        _log.debug('Determine if units is cycling excessively.')
        no_cycles = False
        always_on = False
        always_off = False
        on_count = self.compressor_status_array.count(1)
        if on_count == len(self.compressor_status_array):
            always_on = True
            no_cycles = True
        if sum(self.compressor_status_array) == 0:
            always_off = True
            no_cycles = True
        if no_cycles:
            if always_on: results = {"cycles": 0, "Avg On Cycle": "ALL", "Avg Off Cycle": 0}
            if always_off: results = {"cycles": 0, "Avg On Cycle": 0, "Avg Off Cycle": "ALL"}
            return results

        no_cycles = len(on_indices)
        on_check = 0
        off_check = 1
        if off_indices[0] < on_indices[0]:
            on_check = 1
            off_check = 0
        on_time = [(self.timestamp_array[off] - self.timestamp_array[on]).total_seconds()/60 - 1 for on, off in zip(on_indices, off_indices[on_check:])]
        off_time = [(self.timestamp_array[on] - self.timestamp_array[off]).total_seconds()/60 - 1 for on, off in zip(on_indices[off_check:], off_indices)]

        if self.last_state:
            from_previous = (self.timestamp_array[off_indices[0]] - self.last_time).total_seconds()/60
            on_time.insert(0, from_previous)
        if self.last_state is not None and self.last_state == 0:
            from_previous = (self.timestamp_array[on_indices[0]] - self.last_time).total_seconds()/60
            off_time.insert(0, from_previous)
        self.last_time = self.timestamp_array[0] + td(minutes=self.check_time/4)
        state_ind = self.timestamp_array.index(self.last_time)
        self.last_state = self.compressor_status_array[state_ind]

        avg_on = mean(on_time) if on_time else -99.9
        avg_off = mean(off_time) if off_time else -99.9

        results = {"cycles": no_cycles, "Avg On Cycle": avg_on, "Avg Off Cycle": avg_off}
        return results

    def gen_status(self, peak, valley, time_array):
        '''Generate cycling status array.'''
        extrema_array = [peak, valley]
        first = min(peak[0], valley[0])
        first_array = peak if peak[0] == first else valley
        first_array_index = 0 if  peak[0] == first else 1
        status_value = 1 if peak[0] == first else 0

        extrema_array.pop(first_array_index)
        second_array = extrema_array[0]
        current = first_array[0]
        _next = second_array[0]
        last_stat = self.last_state if self.last_state is not None else 0
        status_array = [last_stat for _ in range(0, current)]
        ascend = True
        index_count = 0

        while True:
            num_pts = int(_next - current)
            for _ in range(0, num_pts):
                status_array.append(status_value)
            if ascend:
                index_count += 1
                if index_count == min(len(valley), len(peak)):
                    break
                ascend = False
                _next = first_array[index_count]
                current = second_array[index_count-1]
                status_value = 0 if status_value == 1 else 1
            else:
                ascend = True
                current = first_array[index_count]
                _next = second_array[index_count]
                status_value = 0 if status_value == 1 else 1
        status_value = 0
        if len(peak) > len(valley):
            status_value = 1
            for _ in range(peak[-1] - valley[-1]):
                status_array.append(0)
        while len(status_array) < len(time_array):
            status_array.append(status_value)
        return status_array

    def output_cycling(self):
        '''output_aggregate writes the results of the data
        aggregation to file for inspection.
        '''
        file_path = inspect.getfile(inspect.currentframe())
        out_dir = os.path.dirname(os.path.realpath(file_path))
        
        now = date.today()
        my_file = "cycling-{}-{}".format(now, self.file)
        file_path = os.path.join(out_dir, my_file)
        ofile = open(file_path, 'a+')
        data_payload = [self.timestamp_array, self.compressor_status_array]
        outs = csv.writer(ofile, dialect='excel')
        writer = csv.DictWriter(ofile, fieldnames=["Timestamp",
                                                   "status"],
                                delimiter=',')
        writer.writeheader()
        for row in itertools.izip_longest(*data_payload):
            outs.writerow(row)
        ofile.close()
        self.file +=1
        
    def output_sp(self):
        '''output_aggregate writes the results of the data
        aggregation to file for inspection.
        '''
        file_path = inspect.getfile(inspect.currentframe())
        out_dir = os.path.dirname(os.path.realpath(file_path))

        now = date.today()
        my_file = "sp-{}-{}".format(now, self.file_sp)
        file_path = os.path.join(out_dir, my_file)
        ofile = open(file_path, 'a+')
        data_payload = [self.timestamp_array,self.zone_temperature_stpt_array]
        outs = csv.writer(ofile, dialect='excel')
        writer = csv.DictWriter(ofile, fieldnames=["Timestamp", "SP"],
                                delimiter=',')
        writer.writeheader()
        for row in itertools.izip_longest(*data_payload):
            outs.writerow(row)
        ofile.close()
        self.file_sp +=1

def main(argv=sys.argv):
    '''Main method called to start the agent.'''
    utils.vip_main(SccCyclingDiagnostic)


if __name__ == '__main__':
    # Entry point for script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass

