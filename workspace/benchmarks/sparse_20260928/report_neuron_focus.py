"""Retain the NEURON-first arrangement on future builds of the main report."""
import importlib.util,json
from pathlib import Path

def apply_neuron_focus(body):
    if 'id="neuron-main"' in body:return body
    parent=Path(__file__).resolve().parent.parent
    candidates=[parent/'neuron_single_20260928',parent/'dbnn_neuron_single_20260928']
    root=next((p for p in candidates if (p/'new_sections.json').exists() and (p/'analysis.json').exists()),None)
    if root is None:return body
    spec=importlib.util.spec_from_file_location('dbnn_neuron_report',root/'analyze_report.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.reorder(body,json.loads((root/'new_sections.json').read_text()),json.loads((root/'analysis.json').read_text()))
