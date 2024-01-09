from scipy.io import wavfile
import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt

# read the audio file
filename = "output.wav"
sample_rate, audio_array = wavfile.read(filename)

# scale the audio signal to a higher range
max_value = np.max(np.abs(audio_array))
audio_array = audio_array.astype(np.float32) / max_value

# apply the Wiener filter to the audio signal
filtered_signal = wiener(audio_array,5005,0.001)

# scale the filtered signal back to the original range
filtered_signal = (filtered_signal * max_value).astype(np.int16)

# save the filtered signal to a new audio file
wavfile.write("output_audio.wav", sample_rate, filtered_signal)


# visualization
fig, (ax_orig, ax_filt) = plt.subplots(2, 1, sharex=True)
t = np.arange(len(audio_array)) / sample_rate

ax_orig.plot(t, audio_array, 'r-', linewidth=1, label='Original')
ax_orig.legend()

ax_filt.plot(t, filtered_signal, 'b-', linewidth=1, label='Wiener filtered')
ax_filt.legend()

ax_orig.set_xlabel('Time [s]')
ax_orig.set_ylabel('Amplitude')
ax_filt.set_xlabel('Time [s]')
ax_filt.set_ylabel('Amplitude')

plt.show()

# plot the spectrograms of the original and filtered signals
fig, (ax_orig, ax_filt) = plt.subplots(2, 1, sharex=True)
ax_orig.specgram(audio_array, Fs=sample_rate, cmap='jet')
ax_orig.set_xlabel('Time [s]')
ax_orig.set_ylabel('Frequency [Hz]')
ax_orig.set_title('Original')

ax_filt.specgram(filtered_signal, Fs=sample_rate, cmap='jet')
ax_filt.set_xlabel('Time [s]')
ax_filt.set_ylabel('Frequency [Hz]')
ax_filt.set_title('Wiener filtered')

plt.show()
