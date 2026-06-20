"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# callbacks.py
class CostAbortCallback:
    def on_step_end(self, args, state, control, **kw):
        elapsed_h = (time.time() - T_START) / 3600
        spent = RESERVE_SPENT + elapsed_h * 3.95
        if spent > 200.0:      _ckpt(); control.should_training_stop = True   # never breach ceiling
        if elapsed_h > 16.0:   _ckpt(); control.should_training_stop = True   # wall-clock
        # step-150 reframed: "flat at whatever step reached"
        if state.global_step >= 50 and _reward_flat(state, window=40):
            _ckpt(); control.should_training_stop = True
