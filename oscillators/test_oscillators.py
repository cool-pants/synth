import oscillators
from scipy.io import wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt


SR = 44100  # Sample rate
figsize = (25, 6.25)
colors = "#323031", "#308E91", "#34369D","#5E2A7E", "#5E2A7E", "#6F3384"


def get_val(osc, sample_rate=SR):
    return [next(osc) for _ in range(sample_rate)]


def plot_osc(Osc, name=""):
    fig = plt.figure(figsize=figsize)

    f = 8
    plt.title(f"{f}Hz {name} Wave")
    for a, p, c in zip([1.0, 0.9, 0.8, 0.7], [0, 15, 30, 45], colors):
        osc = Osc()
        osc.freq = f; osc.amp = a; osc.phase = p
        osc._f = f; osc._a = a; osc._p = p
        iter(osc)
        plt.plot(get_val(osc), color=c, label=f"amp:{a}, phase:{p:02}°")

    plt.legend(loc='lower right')
    fig.savefig(f"./{name.lower()}_all.jpg")

def get_seq(osc, notes=["C4", "null", "C4", "null", "G4", "null", "G4", "null", "A4", "null", "A4", "null", "G4", "null"], note_lens=[0.4, 0.1, 0.3, 0.1, 0.4, 0.1, 0.3, 0.1, 0.4, 0.1, 0.3, 0.1, 0.4, 0.1]):
    samples = []
    osc = iter(osc)
    for note, note_len in zip(notes, note_lens):
        osc.freq = librosa.note_to_hz(note) if note!="null" else 0
        for _ in range(int(SR * note_len)):
            samples.append(next(osc))
    return samples


to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))


def wave_to_wav(wav, wav2=None, fname="temp.wav", amp=0.1):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T

    wavfile.write(fname, SR, wav)

plot_osc(oscillators.SineOsc, "Sine")

osc = oscillators.SineOsc()
wav = get_seq(osc)
wave_to_wav(wav, fname="c4_maj_sine.wav")


# osc = oscillators.SineOsc()
