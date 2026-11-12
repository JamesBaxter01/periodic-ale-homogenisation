#!/usr/bin/env python3
"""One entry point for the publication examples and verification."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
COMMANDS={
    'figure3': ('examples/microscale/figure3_biot.py', 'Paper Figure 3 data: secant Biot coefficients'),
    'figure4': ('examples/microscale/figure4_cycles.py', 'Paper Figure 4 data: compression pulse, flow and recovery'),
    'figure4-stationary': ('examples/microscale/figure4_stationary.py', 'Paper Figure 4 stationary cell (grey line)'),
    'pore-fields': ('examples/microscale/figure4_pore_fields.py', 'Section 5.2 mechanism check: the Figure 4 pulse split by mechanism (about 50 min)'),
    'plot': ('examples/microscale/plot_paper_figures.py', 'Draw paper Figures 3–4 and Table 3 and their LaTeX from saved data'),
    'consolidation-fe2': ('examples/multiscale/consolidation_multiscale.py', 'FE² consolidation with the multiscale coupling'),
    'consolidation-resolved': ('examples/multiscale/consolidation_resolved.py', 'Fully resolved consolidation column (FE² reference)'),
    'verify': ('verification/combined/run.py', 'Verification campaign: mesh, time step, Newton tolerance, tangent, balance'),
    'verify-tables': ('verification/combined/paper_tables.py', 'LaTeX of the paper verification tables from an analysed campaign'),
    'paper': ('examples/paper_generate.py', 'Regenerate every paper figure, table and LaTeX float, and check the numbers in the text'),
}

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
        print('Usage: python run.py COMMAND [options]\n')
        for name,(_,description) in COMMANDS.items():print(f'  {name:15} {description}')
        print('\nUse python run.py COMMAND --help for options. See README.md for paper commands.')
        return 0
    command=sys.argv[1]
    if command not in COMMANDS:raise SystemExit(f'Unknown command {command!r}; use --help')
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),env.get('PYTHONPATH','')])
    env['PYTHONDONTWRITEBYTECODE']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    return subprocess.call([sys.executable,'-u',str(ROOT/COMMANDS[command][0]),*sys.argv[2:]],env=env)

if __name__=='__main__':raise SystemExit(main())
