import ast
import os

filepath = 'phases/phase08_sentinel_protocol.py'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

tree = ast.parse(code)

redundant_funcs = {
    'compute_per_layer_alphas',
    'load_model',
    'load_steering_vectors',
    'generate_responses_batched',
    'compute_honesty_score'
}

new_body = []

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in redundant_funcs:
        continue
        
    if isinstance(node, ast.ClassDef) and node.name == 'Config':
        node.bases = [ast.Name(id='BaseConfig', ctx=ast.Load())]
        new_class_body = []
        keep_attrs = {'SENTINEL_LAYER', 'NOISE_SCALE_FRAC', 'ANOMALY_THRESHOLD', 'N_NOISE_SAMPLES', 'GATE_THRESHOLD', 'GATE_LAYER', 'GATE_SHARPNESS', 'PROMPTS_FILE'}
        for n in node.body:
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and target.id in keep_attrs:
                        new_class_body.append(n)
        if not new_class_body:
            new_class_body.append(ast.Pass())
        node.body = new_class_body
        new_body.append(node)
        continue
        
    new_body.append(node)

tree.body = new_body

# Strip docstrings
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)

new_code = ast.unparse(tree)

imports = """import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.steering_utils import BaseConfig, init_environment, load_model, load_steering_vectors, compute_per_layer_alphas, generate_responses_batched
from utils.behonest_utils import compute_honesty_score

"""

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(imports + new_code)
