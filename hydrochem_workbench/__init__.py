# hydrochem_workbench: hydrochemistry calculations (PHREEQC), plots and a sample map.
# The Streamlit app (app.py) is one way to use it; from a script:
#
#     import hydrochem_workbench as hw
#     result = hw.run('my_data.xlsx', unit='mg/L')
#     hw.save_results(result, 'results.xlsx')
#     hw.save_figures(result, 'figures')
#
# see examples/run_workbench.py

# keep the same as APP_VERSION in app.py
VERSION = '0.16.1'
__version__ = VERSION

from hydrochem_workbench.workflow import (  # noqa: E402,F401
    run, save_results, save_figures, process, fill_rules, style_groups, mark_charge_balance, Result,
    MARK, EXCLUDE, NO_ACTION, VALUE, INTERVAL)
