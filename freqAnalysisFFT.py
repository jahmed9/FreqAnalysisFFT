"""
    It captures audio from microphone, finds and displays the frequency ranges of interest
    in realtime. It works continuously forever unless stopped by Ctrl-C
    It takes the following argumnets as input
    Args:
        threshold: Magnitude threshold above which the frequencies will be considered.
        winSize: Number of samples in a window.
        samplingRate: Audio sampling rate in samples/sec. Default is 8000 samples/sec
                        there is no benefit in increasing sampling rate for 
                        violin sounds as the max frequency is 4000Hz
    
    Note: increasing winSize would increase the frequency resolution and vice versa
"""

import pyaudio
import numpy as np
import scipy
from scipy.fft import fft
import argparse 


freqBoundries = [190,201,213,226,240,254,269,285,302,320,339,359,380,403,427,453,\
             480,508,538,570,604,640,678,719,761,807,855,906,960,1017,1077,\
                 1141,1209,1281,1357,1438,1523,1614,1710,1812,1920,2034,2155,\
                2283,2419,2563,2715,2876,3047,3229,3421]
freqRangeNames = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def findName(freq):
    ind = np.searchsorted(freqBoundries, freq)
    ind -= 1
    return freqRangeNames[ind]

def main(threshold: int, winSize: int, samplingRate: int) -> None:
    captured_frequencies_s = scipy.fft.fftfreq(winSize, d=1.0/samplingRate)[0:int(winSize/2)]
    print("Widow size = ",1000*winSize/samplingRate," msec")
    print("Frequency precision = ",captured_frequencies_s[1])

    p = pyaudio.PyAudio() # open the port 
    # initialize the audio stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=samplingRate, 
                    input=True, frames_per_buffer=winSize)
    
    for i in range(10):# flushout junk in buffer
        data = stream.read(winSize)
    done = False #run forever
    while not done:
        data = stream.read(winSize) #capture from microphone
        data_fl = np.frombuffer(data, dtype=np.short) #convert from bytes to 16 bit integers
        dataFFT = np.abs(fft(data_fl))[0:int(winSize/2)] #find fft
            
        dataFFT /= len(dataFFT) # normalize due to variable winSize
        dataFFT[0:np.searchsorted(captured_frequencies_s,freqBoundries[0])]=dataFFT.sum()/len(dataFFT) #disregard the unaudible very low freq components 
        mag_mask_s = dataFFT > threshold # check which frequency component magnitudes are bigger than threshold 
        freq_s = captured_frequencies_s[mag_mask_s] #collect the frequencies that are above the threshold
        if (np.sum(mag_mask_s)>0):
            for i in range(np.sum(mag_mask_s)):
                tName = findName(freq_s[i]) # Find the violin note for each frequency 
                print("Frequency band Name :",tName, "|",end =" ") # display the notes
            print(" ") # for end of line only
    
    stream.stop_stream() # stop audio stream
    stream.close() # close audio stram
    p.terminate() # close the port

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and display Violin notes continuously')
    parser.add_argument('-t', '--mag_threshold', type=int, default=50)
    parser.add_argument('-w', '--win_size', type=int, default=1024,
                        help="Number of samples in a window")
    parser.add_argument('-s','--samplingRate', type=int, default=8000,
                        help="Audio sampling rate in samples/sec.")
 
    args = parser.parse_args()
    main(args.mag_threshold, args.win_size, args.samplingRate)
