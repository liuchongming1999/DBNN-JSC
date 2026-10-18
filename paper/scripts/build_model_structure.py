"""Draw the DBNN computation, using the manuscript's discrete reset convention.

This explanatory figure does not load or modify any checkpoint or result.
Existing numerical figures retain their original, stable asset filenames.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    fig = plt.figure(figsize=(6.5, 3.55), facecolor='white')
    ax = fig.add_axes([.015, .025, .97, .95])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ink = '#293d4f'
    teal = '#008c80'
    blue = '#366dac'
    red = '#c94f48'
    purple = '#8065a8'

    def box(x, y, w, h, edge=ink, fill='white'):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.008,rounding_size=0.012',
            linewidth=1.15, edgecolor=edge, facecolor=fill))

    def text(x, y, value, size=10, color=ink, weight='normal', **kw):
        return ax.text(x, y, value, ha='center', va='center', fontsize=size,
                       color=color, fontweight=weight, **kw)

    def arrow(a, b, color=ink, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11,
                     linewidth=1.1, color=color, shrinkA=0, shrinkB=0, **kw))

    # The event raster is illustrative; red and blue encode the two site types.
    box(.015, .565, .225, .375)
    text(.1275, .884, 'Synaptic inputs', weight='bold')
    text(.1275, .813, r'$x_i[n]\in\{-1,0,1\}$', 10)
    for y, label, color, sites in [(.722, '+1  E', red, [.101, .147, .204]),
                                   (.634, '-1  I', blue, [.119, .180])]:
        text(.057, y, label, 9, color)
        ax.plot([.093, .221], [y-.014, y-.014], lw=.65, color='#c5cdd5')
        for x in sites:
            ax.plot([x, x], [y-.012, y+.025], lw=1.5, color=color)

    box(.293, .565, .246, .375)
    text(.416, .884, 'Temporal filters', weight='bold')
    text(.416, .799, r'$y_i[n]=(k_i*x_i)[n]$', 10)
    text(.416, .735, 'One kernel per site', 8.5)
    t = np.linspace(0, 1, 120)
    kernel = (1-np.exp(-t/.08))*np.exp(-t/.28)
    ax.plot(.330+.174*t, .600+.086*kernel/kernel.max(), lw=1.6, color=blue)
    ax.plot([.330, .509], [.600, .600], lw=.65, color='#c5cdd5')

    box(.591, .565, .394, .375, edge=teal, fill='#f6fbfa')
    text(.788, .884, 'Bilinear voltage drive', weight='bold', color=teal)
    text(.788, .774, r'$u[n]=b+\sum_i y_i[n]$', 12)
    text(.788, .674, r'$+\ y[n]^\top W\,y[n]$', 12)
    arrow((.248, .765), (.283, .765))
    arrow((.548, .765), (.581, .765))

    # The reset holds r[n] for the current voltage and updates r[n+1] only
    # after detecting s[n]. Thus the feedback is causal, not algebraic.
    box(.035, .165, .366, .282, edge=purple, fill='#faf8fc')
    text(.218, .399, 'Reset state', weight='bold', color=purple)
    text(.218, .319, r'$r[n+1]=\alpha r[n]-a_Rs[n]$', 10.5)
    text(.218, .235, r'$r[0]=0,\quad\alpha=e^{-h/\tau_R}$', 9.5)

    box(.582, .165, .403, .282, edge=teal, fill='#f6fbfa')
    text(.7835, .399, 'Voltage and event outputs', weight='bold', color=teal)
    text(.7835, .319, r'$v[n]=u[n]+r[n]$', 11.5)
    text(.7835, .235, r'$s[n]=\mathbf{1}_{\{v[n]\geq\theta\}}$', 11.5)
    arrow((.788, .557), (.788, .457), color=teal)
    text(.833, .507, r'$u[n]$', 9, teal, bbox={'facecolor':'white','edgecolor':'none','pad':1})
    arrow((.411, .313), (.572, .313), color=purple)
    text(.492, .351, r'$r[n]$', 10, purple)

    # Route the detected events below the boxes to the next reset update.
    ax.plot([.7835, .7835, .218], [.157, .068, .068], lw=1.1, color=purple)
    arrow((.218, .068), (.218, .155), color=purple)
    text(.505, .068, r'$s[n]$: update reset for the next sample', 9, purple,
         bbox={'facecolor':'white','edgecolor':'none','pad':3})
    path = root / 'figures/DBNN_model_structure'
    fig.savefig(path.with_suffix('.pdf'), metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(path.with_suffix('.png'), dpi=210)
    plt.close(fig)


if __name__ == '__main__':
    main()
