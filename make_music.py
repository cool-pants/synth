import seaborn as sns
import itertools
import librosa
import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
import matplotlib.pyplot as plt

from envelopes import ADSREnvelope
from composers import Chain, WaveAdder
from modifiers import Volume, ModulatedVolume, \
                        Panner, ModulatedPanner, \
                        Clipper, ModulatedOscillator

from oscillators import SineOsc, SquareOsc, SawtoothOsc, TriangleOsc

sns.set_theme()
SR = 44_100

figsize = (25, 6.25)
colors = "#323031", "#308E91", "#34369D","#5E2A7E", "#5E2A7E", "#6F3384"

hz = lambda note:librosa.note_to_hz(note)
getfig = lambda : plt.figure(figsize=figsize)
savefig = lambda fig, name: fig.savefig(f"./{name}.jpg")
to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))


def plot(xy, r=1,c=1,i=1,title="", xlabel="",ylabel="",yticks=None, xticks=None,**plot_kwargs):
    # plt.plot helper
    if r > 0:
        plt.subplot(r,c,i)
    plt.title(title)
    if len(xy) == 2:
        plt.plot(*xy, **plot_kwargs)
    else:
        plt.plot(xy, **plot_kwargs)
        
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


def getval(osc, count=SR, it=False):
    if it: osc=iter(osc)
    # returns 1 sec of samples of given osc.
    return [next(osc) for i in range(count)]


def getseq(osc, notes=["C4", "E4", "G4"], note_lens=[0.5, 0.5, 0.5]):
    # Returns samples of the note seq for the given osc.
    samples = []
    osc = iter(osc)
    for note, note_len in zip(notes, note_lens):
        osc.freq = librosa.note_to_hz(note)
        for _ in range(int(SR * note_len)):
            samples.append(next(osc))
    return samples


def wave_to_wav(wav, wav2=None, fname="temp", amp=0.1):
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T
    wavfile.write(f"./{fname}.wav", SR, wav)


def fplot_xy(wave, fslice=slice(0,100), sample_rate=SR):
    # Returns FFTed samples of input wave
    fd = np.fft.fft(wave)
    fd_mag = np.abs(fd)
    x = np.linspace(0, sample_rate, len(wave))
    y = fd_mag * 2 / sample_rate
    return x[fslice], y[fslice]


def getadsr(a=0.05, d=0.3, sl=0.7, r=0.2, sd=0.4, sample_rate=SR):
    adsr = ADSREnvelope(a, d, sl, r)
    down_len = int(sum([a, d, sd])*sample_rate)
    up_len = int(r*sample_rate)
    adsr = iter(adsr)
    adsr_vals = getval(adsr, down_len)
    adsr.trigger_release()
    adsr_vals.extend(getval(adsr, up_len))
    return adsr_vals, down_len, up_len


def visualizeEnvelope(XOscillator, name, a=0.02, d=0.05, sl=0.7, r=0.1, sd=0.1):
    fig = getfig()
    fig.suptitle(f"Modulating A '{name}' Wave")
    envvals, down, up = getadsr(a,d,sl,r,sd)
    envvals = np.array(envvals)
    xd = np.arange(down)
    xu = np.arange(down, down+up)

    plot((xd, envvals[:down]),1,3,1, color=colors[2], label="Envelope down")
    plot((xu, envvals[down:]),0,3,1, color=colors[1], label="Envelope released", xlabel="samples", ylabel="amplitude")
    plt.legend(loc="lower center")

    osc = XOscillator(110)
    oscvals = np.array(getval(osc, it=True))[:down+up]
    plot(oscvals, 1,3,2, label=f"110 Hz '{name}' Wave", xlabel="samples", color=colors[3])
    plt.legend(loc="lower center")

    plot((xd, oscvals[:down]*envvals[:down]),1,3,3, color=colors[2], label="Note down")
    plot((xu, oscvals[down:]*envvals[down:]),0,3,3, color=colors[1], label="Note released", xlabel="samples")
    plt.legend(loc="lower center")
    savefig(fig, f"adsr'{name}'")

visualizeEnvelope(SineOsc, "Sine")
visualizeEnvelope(SquareOsc, "Square")

def amp_mod(init_amp, env):
    return env * init_amp

def freq_mod(init_freq, env, mod_amt=0.01, sustain_level=0.7):
    return init_freq+ ((env - sustain_level) * init_freq * mod_amt)

def visualizeModulatedADSR(XOscillator, name,a=0.1, d=0.2, sl=0.7, r=0.3, sd=0.4):
    mod = ModulatedOscillator(
        XOscillator(110),
        ADSREnvelope(a,d,sl,r),
        amp_mod= amp_mod,
        fred_mod = lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sl)
    )
    vals = getadsr(a,d,sl,sd,r,mod=mod)
    fig = getfig()
    plot(vals, title=f"110Hz {name} Wave, ADSR applied to Frequency as well as Amplitude",\
            xlabel="samples", ylabel="amplitude", color=colors[2])
    savefig(fig, f"{name}_freq_amp_mod")
    pass

# visualizeModulatedADSR(SineOsc,"Sine")
# visualizeModulatedADSR(SquareOsc,"Square")

# Actually playing with waves and music here
gen = WaveAdder(
    Chain(
        TriangleOsc(hz("C4")),
        ModulatedPanner(
            SineOsc(3, wave_range=(-1,1))
        ),
    ),
    Chain(
        SineOsc(hz("E4")),
        ModulatedPanner(
            SineOsc(2, wave_range=(-1,1))
        ),
    ),
    Chain(
        TriangleOsc(hz("G4")),
        ModulatedPanner(
            SineOsc(3, phase=180, wave_range=(-1,1))
        ),
    ),
    Chain(
        SineOsc(hz("B4")),
        ModulatedPanner(
            SineOsc(2, phase=180, wave_range=(-1,1))
        ),
    ),
    stereo=True
)

wav = getval(iter(gen), int(4*SR))
wave_to_wav(wav, fname="rotate_cmajor")
ipd.Audio("./rotate_cmajor.wav")
