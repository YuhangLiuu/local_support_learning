"""Explicit pre-experiment numerical gate for prepared models/batches.

No downloads, installs, or implicit invocation by training. Atol/rtol default to
zero. A nonzero tolerance requires a documented kernel-specific reason.
"""

import copy


def compare_training_step(model_a, model_b, batch, loss_fn, optimizer_factory,
                          *, atol=0., rtol=0., tolerance_reason=None):
    """A must be diagnostics-off, B basic-on, with identical initial state.

    Executes one synthetic/held-out training step on caller-owned disposable
    models. Verifies logits, loss, grads, parameters, RNG, EMA and GMM parameters.
    Full EM/evaluator equality should be tested with compare_outputs separately.
    """
    import torch
    if (atol or rtol) and not tolerance_reason:
        raise ValueError("nonzero tolerances require a reason")
    def equal(a, b):
        if not torch.equal(a, b):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    def random_state():
        import random
        import numpy as np
        return (torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                random.getstate(), copy.deepcopy(np.random.get_state()))
    def restore(state):
        import random
        import numpy as np
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2]); np.random.set_state(state[3])
    start = random_state()
    parameters_a, parameters_b = dict(model_a.named_parameters()), dict(model_b.named_parameters())
    assert parameters_a.keys() == parameters_b.keys()
    for name in parameters_a:
        equal(parameters_a[name], parameters_b[name])
    runs = []
    for model in (model_a, model_b):
        restore(start)
        optimizer = optimizer_factory(model)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = loss_fn(output, batch)
        loss.backward()
        state = random_state()
        gradients = {n: None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
        optimizer.step()
        runs.append((output.logits.detach(), loss.detach(), gradients, state))
    equal(runs[0][0], runs[1][0]); equal(runs[0][1], runs[1][1])
    for name in parameters_a:
        a, b = runs[0][2][name], runs[1][2][name]
        assert (a is None) == (b is None)
        if a is not None:
            equal(a, b)
        equal(parameters_a[name], parameters_b[name])
    assert torch.equal(runs[0][3][0], runs[1][3][0])
    import numpy as np
    assert runs[0][3][2] == runs[1][3][2]
    assert np.array_equal(runs[0][3][3][1], runs[1][3][3][1])
    assert runs[0][3][3][2:] == runs[1][3][3][2:]
    if runs[0][3][1] is not None:
        assert all(torch.equal(a,b) for a,b in zip(runs[0][3][1], runs[1][3][1]))
    modules_a, modules_b = dict(model_a.named_modules()), dict(model_b.named_modules())
    for name, a in modules_a.items():
        if not hasattr(a, "_smoothing_state"):
            continue
        b = modules_b[name]
        assert a._smoothing_state.keys() == b._smoothing_state.keys()
        for key in a._smoothing_state:
            equal(a._smoothing_state[key], b._smoothing_state[key])
        for ca, cb in zip(a.cl_gmm, b.cl_gmm):
            for which in ("pos", "neg"):
                pa, pb = ca.params_vectorized[which], cb.params_vectorized[which]
                assert (pa is None) == (pb is None)
                if pa is not None:
                    for x, y in zip(pa, pb):
                        equal(x,y)
    return dict(passed=True, atol=atol, rtol=rtol, tolerance_reason=tolerance_reason,
                scope="one prepared training step; not full EM or evaluator validation")


def compare_outputs(off_call, on_call, *, atol=0., rtol=0.):
    """Generic tree comparison for caller-prepared EM states or eval results.

    Each callable must restore the same initial RNG/model/data/EMA/KV state.
    """
    import torch
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x,y in zip(a,b):
                compare(x,y)
        else:
            assert a == b
    compare(off_call(), on_call())
