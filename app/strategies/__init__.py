"""Signal-generation strategies that live alongside the ML pipeline.

Each strategy module exposes a service class with the same shape as
`SignalService` so the scheduler can fan signals through a uniform
interface, tagging each row with a distinct `source` for the comparison
report:

    - ML equities  → source='ml_equities'  (handled by SignalService)
    - ML FX        → source='ml_fx'        (handled by SignalService)
    - Donchian FX  → source='rule_donchian' (handled by DonchianService)
"""
