import os
import numpy as np
import mpmath as mp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

K = float(mp.khinchin(dps=16))


def save_anim(anim, fpath, fps=30):
  fname, ext = os.path.splitext(fpath)
  if ext == '.gif':
    anim.save(fpath, writer=animation.PillowWriter(fps=fps))
  elif ext == '.mp4':
    FFwriter = animation.FFMpegWriter(fps=fps)  # For colab.
    anim.save(fpath, writer=FFwriter)
  else:
    raise ValueError(f'Unknown file extension `{ext}`')


def range_change_log(OldValue, OldMin, OldMax, NewMin, NewMax):
  NewMin, NewMax = np.log(NewMin), np.log(NewMax)
  NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
  return np.exp(NewValue)

def init_plot(x_end, y_window=0.5):
  fig, ax = plt.subplots(figsize=(24,12))
  ax.set_xlim(0, x_end+1)
  ax.set_ylim(float(K)-y_window, float(K)+y_window)
  ax.axhline(K, color='r', linestyle='--', linewidth=3, zorder=4)
  return fig, ax



class Lines():
  def __init__(self, nums, line_x_points, mean_lines):
    self.nums, self.line_x_points, self.mean_lines = nums, line_x_points, mean_lines
    self.fig, self.ax = init_plot(line_x_points[-1])
    self.num_coeffs, self.num_lines = mean_lines.shape
    self.num = self.ax.text(0.2, 1.01, '0.0', transform=self.ax.transAxes, fontsize=12)
    scat = self.ax.scatter(0,0, color='b', s=72, zorder=3, label='0.0')
    self.legend = plt.legend(loc=1, fontsize=12)
    self.lines, self.scats = [], []  # Used as a global vars.

  def _update_decays(self, alpha_decay_rate):
    indices = list(range(len(self.lines)))
    filtered_indices = indices.copy()
    for i in indices:
      line, scat = self.lines[i], self.scats[i]
      a = line.get_alpha()
      if a == 1:
        line.set_alpha(0.5)
        line.set_linewidth(1)
        c = next(self.ax._get_lines.prop_cycler)['color']
        line.set_color(c)
        scat.set_alpha(0.5)
        scat.set_facecolors('none')
        scat.set_edgecolors('y')
      elif a == 0:
        filtered_indices.remove(i)
      else:
        a = max(0, a - alpha_decay_rate)  # Controls the speed of decay.
        line.set_alpha(a)
        scat.set_alpha(a)
        s = scat.get_sizes()[0] ** 0.5
        s += 250 * alpha_decay_rate
        scat.set_sizes([s**2])
    self.lines = [self.lines[i] for i in filtered_indices]
    self.scats = [self.scats[i] for i in filtered_indices]

  def update_plot(self, i):
    if i == -1:
      pass  # Could do something at the start.
    elif 0 <= i < self.num_lines:
      last_point_x = self.line_x_points[-1]
      last_point_y = self.mean_lines[i][-1]
      self._update_decays(10/self.num_lines)
      self.num.set_text(f'{mp.nstr(self.nums[i],100)}...')
      current_line = self.ax.plot(self.line_x_points, self.mean_lines[i], c='w', alpha=1, linewidth=2)[0]
      current_scat = self.ax.scatter(last_point_x, last_point_y, color='b', s=72, zorder=3)
      self.legend.get_texts()[0].set_text(f'{last_point_y:.10f}')
      self.lines.append(current_line)
      self.scats.append(current_scat)
    elif i == self.num_lines:
      self._update_decays(5/self.num_lines)
      self.num.set_text('1.0')
      self.legend.get_texts()[0].set_text('0.0')
    else:
      raise ValueError('Unexpected value of `i`:', i)
    return self.lines

def animate_line_by_line(nums, line_x_points, mean_lines):
  """The only reason to have this as a class and not a function is to be able to use a global list,
  which could be updated on every call to the function."""
  lines = Lines(nums, line_x_points, mean_lines)
  lines_range = list(range(lines.num_lines))
  # Add inital and end indicator indices.
  pad_frames_start_count = int(0.04*(lines.num_lines-1))
  pad_frames_end_count = int(0.14*(lines.num_lines-1))
  lines_range = [-1]*pad_frames_start_count + lines_range
  lines_range += [lines.num_lines] * pad_frames_end_count
  return animation.FuncAnimation(lines.fig, lines.update_plot, tqdm(lines_range, initial=1), blit=True)


def animate_all_lines(line_x_points, mean_lines, vibrant=False):
  num_coeffs, num_lines = mean_lines.shape
  fig, ax = init_plot(line_x_points[-1])
  lines = []
  for _ in range(len(mean_lines)):
    lines.append(ax.plot([],[], alpha=0.3)[0])

  colors = []
  while (color:=next(ax._get_lines.prop_cycler)['color']) not in colors:
    colors.append(color)

  def init():
    for line in lines:
      line.set_data([],[])
    return lines

  def animate(i):
    for line_index, (line, mean_line) in enumerate(zip(lines, mean_lines)):
      line.set_data(line_x_points[:i+1], mean_line[:i+1])
      if vibrant:
        shift = i//5  # Controls the speed of color shifts.
        color = colors[(line_index+shift) % len(colors)]
        line.set_color(color)
    return lines

  return animation.FuncAnimation(fig, animate, tqdm(range(num_coeffs), initial=1), init_func=init, blit=True)


def animate_average_line(line_x_points, mean_lines):
  num_coeffs, num_lines = mean_lines.shape
  line_range = 1, 1 + num_coeffs
  avg_of_mean_lines = mean_lines.mean(axis=0)
  fig, ax = init_plot(line_x_points[-1])
  diff = ax.text(0.4, 1.01, '0.0', transform=ax.transAxes, fontsize=12)
  scat = ax.scatter(0,0, color='b', s=20, zorder=3, label='0.0')
  legend = ax.legend(loc=1, fontsize=12)

  lines = []
  for mean_line in mean_lines:
    lines.append(ax.plot(line_x_points, mean_line, alpha=0.3)[0])
  avg_line = ax.plot([],[], c='k')[0]

  def init():
    avg_line.set_data([],[])
    return [avg_line]

  def range_change(OldValue, OldMin, OldMax, NewMin, NewMax):
    return (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

  def animate(i):
    avg_line.set_data(line_x_points[:i], avg_of_mean_lines[:i])
    last_point = avg_of_mean_lines[:i][-1]
    diff.set_text(f'K - mean = {(float(K)-last_point):0.10f}')
    scat.set_offsets([line_x_points[:i][-1], last_point])
    legend.get_texts()[0].set_text(f'mean = {last_point:.10f}')
    a = range_change(i, *line_range, 0.3, 0.01)
    for line in lines:
      line.set_alpha(a)  # Light to almost invisible.
    y_len = range_change_log(i, *line_range, 1, 0.02)
    ax.set_ylim(K-y_len/2, K+y_len/2)  # y axis zoom.
    if i == int(0.95*num_coeffs):  # Change color upon 95% completion.
      avg_line.set_color('w')
    return [avg_line]

  return animation.FuncAnimation(fig, animate, tqdm(range(*line_range), initial=1), init_func=init, blit=True)



def animate_labelled_lines(line_x_points, labelled_lines, y_window=0.5):
  fig, ax = init_plot(line_x_points[-1], y_window)
  lines = []
  for label in labelled_lines:
    lines.append(ax.plot([],[], label=label, alpha=0.7)[0])
  ax.legend(loc=1, fontsize=12)

  def animate(i):
    for line, (label, line_y_points) in zip(lines, labelled_lines.items()):
      line.set_data(line_x_points[:i+1], line_y_points[:i+1])
    return lines

  return animation.FuncAnimation(fig, animate, tqdm(range(len(line_x_points)), initial=1), blit=True)


def animate_line_zoom_in(line_x_points, labelled_line):
  label, line = labelled_line
  line_range = 1, 1 + len(line)
  fig, ax = init_plot(line_x_points[-1])
  diff = ax.text(0.87, 1.01, '0.0', transform=ax.transAxes, fontsize=12)
  scat = ax.scatter(0,0, color='b', s=100, zorder=3, label='0.0')
  legend = ax.legend(loc=1, fontsize=12)
  ax.set_title(label, fontsize=12)
  plot_line = ax.plot([],[])[0]

  def init():
    plot_line.set_data([],[])
    return [plot_line]

  def animate(i):
    plot_line.set_data(line_x_points[:i], line[:i])
    last_point = line[:i][-1]
    diff.set_text(f'K - mean = {(float(K)-last_point):.10f}')
    scat.set_offsets([line_x_points[:i][-1], last_point])
    legend.get_texts()[0].set_text(f'mean = {last_point:.10f}')
    y_len = range_change_log(i, *line_range, 1, 0.05)
    ax.set_ylim(K-y_len/2, K+y_len/2)  # y axis zoom.
    return [plot_line]

  return animation.FuncAnimation(fig, animate, tqdm(range(*line_range), initial=1), init_func=init, blit=True)  


def animate_line_zoom_out(line_x_points, labelled_line):
  label, line = labelled_line
  line_range = np.geomspace(1, 1+len(line), 1000).astype(int)
  fig, ax = init_plot(line_x_points[-1])
  diff = ax.text(0.87, 1.01, '0.0', transform=ax.transAxes, fontsize=12)
  scat = ax.scatter(0,0, color='b', s=100, zorder=3, label='0.0')
  legend = ax.legend(loc=1, fontsize=12)
  ax.set_title(label, fontsize=12)
  plot_line = ax.plot([],[])[0]

  def init():
    plot_line.set_data([],[])
    return [plot_line]

  def range_change_log(OldMin, OldMax, buffer=0.1):
    NewMin = OldMin - buffer*OldMin
    NewMax = OldMax + buffer*OldMax
    return NewMin, NewMax

  def animate(i):
    x, y = line_x_points[:i], line[:i]
    plot_line.set_data(x, y)
    diff.set_text(f'K - mean = {(float(K) - y[-1]):.10f}')
    scat.set_offsets([x[-1], y[-1]])
    legend.get_texts()[0].set_text(f'mean = {y[-1]:.10f}')
    ax.set_xlim(range_change_log(x.min(), x.max()))
    ax.set_ylim(range_change_log(y.min(), y.max()))
    return [plot_line]

  return animation.FuncAnimation(fig, animate, tqdm(line_range, initial=1), init_func=init, blit=True)


def animate_bars(probs, scale, steps):
  fig, ax = plt.subplots()
  ax.set_xlim(0, 11)
  ax.set_ylim(0, 1.1)
  ax.axhline(1, color='w', linestyle='--', linewidth=3, zorder=4)
  prob_bars = ax.bar(range(1,1+len(probs)), [0]*len(probs))
  probs *= scale
  bar_texts = [
      ax.text(bar_index-.1, 1.02, '0', fontsize=10) for bar_index in range(1, 1+len(prob_bars))
  ]

  def init():
    for bar in prob_bars:
      bar.set_height(0)
    return prob_bars

  def animate(i):
    for p, bar, freq in zip(probs, prob_bars, bar_texts):
      bar.set_height((i*p)%1)
      freq.set_text(int(i*p))
    return prob_bars

  return animation.FuncAnimation(fig, animate, tqdm(range(1, int(steps/scale)), initial=1), init_func=init, blit=True)


class Bars():
  def __init__(self, prob_steps):
    self.prob_steps = prob_steps
    self.fig, self.axs = plt.subplots(1, 2, figsize=(24,8))
    self.axs[0].set_ylim(0, 1.1)
    self.axs[1].set_ylim(0, 1.1)
    self.axs[0].set_xticks(range(1,11))
    self.axs[1].set_xticks(range(1,11))
    self.axs[0].axhline(1, color='w', linestyle='--', linewidth=3, zorder=4)
    nums = np.arange(1, 1+len(prob_steps))
    pmf = lambda k: -np.log2(1-(1/((k+1)**2)))
    self.axs[1].plot(
        nums, pmf(nums), c='y', linewidth=3, marker='o', mfc='r', mec='r',
        label=r'$-\log_2\left[1-\frac{1}{(k+1)^2}\right]$'
    )
    self.axs[1].legend(fontsize=12)
    self.axs[1].set_title('Normalized frequencies')

    self.probs = [0]*len(prob_steps)
    self.freqs = [0]*len(prob_steps)
    self.prob_bars = self.axs[0].bar(nums, self.probs)
    self.freq_bars = self.axs[1].bar(nums, self.freqs)
    self.bar_texts = [
        self.axs[0].text(bar_index-.1, 1.02, '0', fontsize=12) for bar_index in range(1, 1+len(prob_steps))
    ]
    self.coeffs = []
    self.prob_title = self.axs[0].text(0.35, 1.01, 'GM of coeffs: ', transform=self.axs[0].transAxes, fontsize=12)
    self.title = self.fig.suptitle(f'coeffs: {self.coeffs}', fontsize = 12)

  def update_bars(self, _):
    for i, (prob_step, prob_height, bar, freq_text) in enumerate(
        zip(self.prob_steps, self.probs, self.prob_bars, self.bar_texts)
    ):
      prob_height += prob_step
      if prob_height >= 1:
        prob_height -= 1
        self.freqs[i] += 1
        for freq, freq_bar in zip(self.freqs, self.freq_bars):
          freq_bar.set_height(freq / sum(self.freqs))
        freq_text.set_text(str(self.freqs[i]))
        self.coeffs.append(i+1)
        self.title.set_text(f'coeffs: {self.coeffs}')
        geom_mean = np.exp(np.log(self.coeffs).sum()/len(self.coeffs))
        self.prob_title.set_text(f'GM of coeffs: {geom_mean:.5f}')
      bar.set_height(prob_height)
      self.probs[i] = prob_height
    return self.prob_bars

def animate_bars(probs, scale, steps):
  bars = Bars(probs * scale)
  steps = int(np.ceil(steps/scale))
  return animation.FuncAnimation(bars.fig, bars.update_bars, tqdm(range(1, steps), initial=1), blit=True)