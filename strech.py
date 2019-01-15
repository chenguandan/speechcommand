from librosa import effects
from tqdm import tqdm
from glob import glob
import numpy as np
from scipy.io import wavfile as wf
from os.path import join as jp, basename as bn
"""
up off
Both words are rather short. You could try playing them slower and average both predictions (slow & default version of the record). This gave me a small improvement (low 87% to higher 87%). The librosa method for time stretching is slow. You can't use it online. I therefore dumped the files to disk. Like so:
"""

def main():
  tta_speed = 0.9  # slow down (i.e. &lt; 1.0)
  samples_per_sec = 16000
  test_fns = sorted(glob('data/test/audio/*.wav'))
  tta_dir = 'data/tta_test/audio'
  for fn in tqdm(test_fns):
    basename = bn(fn)
    rate, data = wf.read(fn)
    assert len(data) == samples_per_sec
    data = np.float32(data) / 32767
    data = effects.time_stretch(data, tta_speed)
    data = data[-samples_per_sec:]
    out_fn = jp(tta_dir, basename)
    wf.write(out_fn, rate, np.int16(data * 32767))


if __name__ == '__main__':
  main()