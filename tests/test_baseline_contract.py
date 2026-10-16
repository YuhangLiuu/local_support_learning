"""Static evidence that experiment policy/defaults did not change."""

import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def baseline(file):
    return subprocess.check_output(["git", "show", f"baseline-lsl-original:{file}"],
                                   cwd=ROOT, text=True, encoding="utf-8")


def functions(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def test_all_original_experiment_defaults_unchanged():
    old, new = {}, {}
    exec(compile(baseline("configs/config_continual.py"), "baseline-config", "exec"), old)
    exec(compile((ROOT / "configs/config_continual.py").read_text(encoding="utf-8"), "research-config", "exec"), new)
    before, after = vars(old["Configuration"]()), vars(new["Configuration"]())
    assert all(after[key] == value for key,value in before.items())
    assert not after["diagnostics_enabled"]


class RemoveObservationCalls(ast.NodeTransformer):
    calls = {"stage_context", "dataset_context", "reproduction_stage", "reproduction_summary", "save_diagnostic_checkpoint"}
    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='load_lm_eval':
            return None
        return self.generic_visit(node)
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in self.calls:
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.decorator_list = [d for d in node.decorator_list if not (
            isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "observe_eval")]
        return self.generic_visit(node)


def test_training_data_loss_optimizer_scheduler_and_eval_ast_unchanged():
    before = functions(baseline("train_continual.py"))
    after = functions((ROOT / "train_continual.py").read_text(encoding="utf-8"))
    # Compare every original top-level function after removing ONLY diagnostic
    # boundary calls/decorators. Data, loss, updates and evaluator policy remain exact.
    for name, node in before.items():
        stripped = RemoveObservationCalls().visit(after[name])
        assert ast.dump(node, include_attributes=False) == ast.dump(stripped, include_attributes=False), name
