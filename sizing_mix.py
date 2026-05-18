from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

import simulation.Simulate as Simulate

# DEFINE VARIABLES FOR PATHS AND OTHER CONSTANTS
# ROOT_DIR = 
# DEFAULT_SIM_DIR =
# DEFAULT_TEMPLATE_NETLIST_PATH =
# DEFAULT_NEW_NETLIST_DIR =
# DEFAULT_TASKS_DIR = 
# DEFAULT_TASKS_JSON_PATH = 


@dataclass
class LLMPlan:
    optimizer: str
    samples: List[Dict[str, Any]]
    ranges: Dict[str, Tuple[float, float]]
    relations: List[str]
    raw_response: str

def askLLM(prompt: str, mymodel: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=mymodel,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
            include_thoughts=True
            )
        )
        )
    answer = ""
    print("Prompt sent to LLM:")
    print(prompt)
    print("LLM response:")
    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            print("Thought summary:")
            print(part.text)
            print()
        else:
            print("Answer:")
            print(part.text)
            print()
            ans += part.text
    return answer

def _ensure_text_write(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
        return
    except OSError:
        if not path.is_symlink():
            raise
    path.unlink()
    path.write_text(content, encoding="utf-8")


def compose_topology_into_sim(
    topology_id: int,
    new_netlist_dir: Path = DEFAULT_NEW_NETLIST_DIR,
    template_netlist_path: Path = DEFAULT_TEMPLATE_NETLIST_PATH,
) -> str:
    topology_path = new_netlist_dir / f"{topology_id}.txt"
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology netlist not found: {topology_path}")

    template_lines = template_netlist_path.read_text(encoding="utf-8").splitlines()
    if len(template_lines) < 3:
        raise ValueError(f"Invalid template netlist: {template_netlist_path}")

    topo_lines = topology_path.read_text(encoding="utf-8").splitlines()
    if len(topo_lines) < 3:
        raise ValueError(f"Invalid topology netlist (need at least 3 lines): {topology_path}")

    merged_lines = [template_lines[0], template_lines[1], *topo_lines[1:-1], ".ends op_amp"]
    merged_text = "\n".join(merged_lines) + "\n"
    _ensure_text_write(template_netlist_path, merged_text)
    return merged_text


def load_spec_text(spec_id: int, tasks_dir: Path = DEFAULT_TASKS_DIR) -> str:
    prompt_path = tasks_dir / f"{spec_id}.prompt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Spec prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_spec_constraints(spec_id: int, tasks_json_path: Path = DEFAULT_TASKS_JSON_PATH) -> Dict[str, Dict[str, float]]:
    if not tasks_json_path.exists():
        raise FileNotFoundError(f"tasks.json not found: {tasks_json_path}")
    with tasks_json_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    constraints = tasks.get("filters", {}).get(str(spec_id))
    if constraints is None:
        raise KeyError(f"Spec id {spec_id} not found in tasks.json filters.")
    return constraints


PARAM_QUOTED_RE = re.compile(r"'([^']+)'")
PARAM_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def extract_param_names(netlist_text: str) -> List[str]:
    names: List[str] = []
    seen = set()
    for quoted in PARAM_QUOTED_RE.findall(netlist_text):
        for token in PARAM_TOKEN_RE.findall(quoted):
            if token not in seen:
                seen.add(token)
                names.append(token)
    if not names:
        raise ValueError("No sizing parameters found in netlist.")
    return names


def is_integer_param(name: str) -> bool:
    upper = name.upper()
    return upper.endswith("_M") or "_M_" in upper


def infer_param_range(name: str) -> Tuple[float, float]:
    upper = name.upper()
    if upper.endswith("_L") or "_L_" in upper:
        return (0.15, 1.0)
    if upper.endswith("_W") or "_W_" in upper:
        return (0.5, 10.0)
    if upper.endswith("_M") or "_M_" in upper:
        return (1.0, 100.0)
    if "CAPACITOR" in upper or upper.startswith("CCOMP"):
        return (1e-12, 1e-10)
    if "RESISTOR" in upper:
        return (1.0, 1e6)
    if "CURRENT" in upper or upper.startswith("IB"):
        return (5e-7, 2e-5)
    if upper.startswith("VB") or upper.startswith("VBIAS") or upper.startswith("VREF"):
        return (0.0, 1.8)
    if upper.startswith("V"):
        return (0.0, 1.8)
    return (0.15, 10.0)


def build_bounds(param_names: List[str]) -> np.ndarray:
    return np.asarray([infer_param_range(name) for name in param_names], dtype=np.float64)


def clip_vector(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def sample_uniform(n: int, rng: np.random.Generator, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, bounds.shape[0]))


def vector_to_params(x: np.ndarray, param_names: List[str], bounds: np.ndarray) -> Dict[str, float]:
    x_clip = clip_vector(np.asarray(x, dtype=np.float64), bounds)
    out: Dict[str, float] = {}
    for i, name in enumerate(param_names):
        value = float(x_clip[i])
        if is_integer_param(name):
            value = float(int(round(value)))
            value = float(np.clip(value, bounds[i, 0], bounds[i, 1]))
        out[name] = value
    return out


def dict_to_vector(
    sample: Dict[str, Any],
    param_names: List[str],
    bounds: np.ndarray,
    rng: np.random.Generator,
    fallback: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    values: List[float] = []
    fallback = fallback or {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        raw = sample.get(name, fallback.get(name, None))
        if raw is None:
            value = float(rng.uniform(lo, hi))
        else:
            try:
                value = float(raw)
            except Exception:
                value = float(rng.uniform(lo, hi))
        value = float(np.clip(value, lo, hi))
        if is_integer_param(name):
            value = float(int(round(value)))
            value = float(np.clip(value, lo, hi))
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def params_to_vector(params: Dict[str, float], param_names: List[str], bounds: np.ndarray) -> np.ndarray:
    out = []
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        value = float(params.get(name, lo))
        value = float(np.clip(value, lo, hi))
        if is_integer_param(name):
            value = float(int(round(value)))
            value = float(np.clip(value, lo, hi))
        out.append(value)
    return np.asarray(out, dtype=np.float64)


def result_meets_constraints(result: Dict[str, Any], constraints: Dict[str, Dict[str, float]]) -> bool:
    if not isinstance(result, dict):
        return False
    for key, cond in constraints.items():
        value = result.get(key, None)
        if value is None:
            return False
        try:
            value = float(value)
        except Exception:
            return False
        if not np.isfinite(value):
            return False
        for op, target in cond.items():
            target = float(target)
            if op == ">=" and not value >= target:
                return False
            if op == "<=" and not value <= target:
                return False
    return True


def score_result(result: Dict[str, Any], constraints: Dict[str, Dict[str, float]]) -> float:
    if not isinstance(result, dict):
        return -1e9
    score = 0.0
    for key, cond in constraints.items():
        value = result.get(key, None)
        if value is None:
            score -= 10.0
            continue
        try:
            value = float(value)
        except Exception:
            score -= 10.0
            continue
        if not np.isfinite(value):
            score -= 10.0
            continue

        for op, target in cond.items():
            target = float(target)
            denom = abs(target) + 1e-12
            if op == ">=":
                # Any satisfied constraint contributes the same score (0),
                # only unsatisfied constraints get penalized.
                if value < target:
                    margin = (value - target) / denom
                    score += 4.0 * margin
            elif op == "<=":
                # Any satisfied constraint contributes the same score (0),
                # only unsatisfied constraints get penalized.
                if value > target:
                    margin = (target - value) / denom
                    score += 4.0 * margin
    return float(score)


def get_violations(result: Dict[str, Any], constraints: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    if not isinstance(result, dict):
        for key, cond in constraints.items():
            for op, target in cond.items():
                violations.append(
                    {
                        "metric": key,
                        "op": op,
                        "target": float(target),
                        "value": None,
                        "reason": "missing_result",
                    }
                )
        return violations

    for key, cond in constraints.items():
        raw_value = result.get(key, None)
        value: Optional[float]
        try:
            value = float(raw_value) if raw_value is not None else None
        except Exception:
            value = None

        for op, target in cond.items():
            target_f = float(target)
            if value is None or not np.isfinite(value):
                violations.append(
                    {
                        "metric": key,
                        "op": op,
                        "target": target_f,
                        "value": raw_value,
                        "reason": "missing_or_nonfinite",
                    }
                )
                continue

            ok = (value >= target_f) if op == ">=" else (value <= target_f) if op == "<=" else True
            if not ok:
                violations.append(
                    {
                        "metric": key,
                        "op": op,
                        "target": target_f,
                        "value": value,
                        "reason": "constraint_not_met",
                    }
                )
    return violations


def normalize_X(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = np.maximum(hi - lo, 1e-12)
    return (X - lo) / span


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 0.2, variance: float = 1.0) -> np.ndarray:
    x1_sq = np.sum(X1**2, axis=1, keepdims=True)
    x2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    dist2 = np.maximum(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), 0.0)
    return variance * np.exp(-0.5 * dist2 / (length_scale**2))


def gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    noise: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(X_train, X_train) + noise * np.eye(X_train.shape[0])
    Ks = rbf_kernel(X_train, X_test)
    Kss_diag = np.diag(rbf_kernel(X_test, X_test))

    L = np.linalg.cholesky(K + 1e-12 * np.eye(K.shape[0]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    var = np.maximum(Kss_diag - np.sum(v * v, axis=0), 1e-12)
    return mu, np.sqrt(var)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    ei = (mu - y_best) * cdf + sigma * pdf
    return np.maximum(ei, 0.0)


def propose_bo(
    history_X: List[np.ndarray],
    history_y: List[float],
    rng: np.random.Generator,
    bounds: np.ndarray,
    n_candidates: int = 1024,
) -> np.ndarray:
    if len(history_X) < 6:
        return sample_uniform(1, rng, bounds)[0]
    X = np.asarray(history_X, dtype=np.float64)
    y = np.asarray(history_y, dtype=np.float64)

    candidates = sample_uniform(n_candidates, rng, bounds)
    Xn = normalize_X(X, bounds)
    Cn = normalize_X(candidates, bounds)
    mu, sigma = gp_posterior(Xn, y, Cn)
    ei = expected_improvement(mu, sigma, float(np.max(y)))
    return candidates[int(np.argmax(ei))]


def _tournament_select(population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = min(3, len(population))
    idx = rng.choice(len(population), size=k, replace=False)
    win = idx[int(np.argmax(fitness[idx]))]
    return population[win].copy()


def propose_ga(
    history_X: List[np.ndarray],
    history_y: List[float],
    rng: np.random.Generator,
    bounds: np.ndarray,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.08,
) -> np.ndarray:
    n = len(history_X)
    if n < 4:
        return sample_uniform(1, rng, bounds)[0]

    X = np.asarray(history_X, dtype=np.float64)
    y = np.asarray(history_y, dtype=np.float64)
    top_k = min(max(8, n // 2), n)
    keep_idx = np.argsort(y)[-top_k:]
    population = X[keep_idx]
    fitness = y[keep_idx]

    p1 = _tournament_select(population, fitness, rng)
    p2 = _tournament_select(population, fitness, rng)
    alpha = rng.uniform(0.0, 1.0, size=p1.shape[0])
    child = alpha * p1 + (1.0 - alpha) * p2

    span = bounds[:, 1] - bounds[:, 0]
    mask = rng.random(child.shape[0]) < mutation_rate
    noise = rng.normal(0.0, mutation_scale * span, size=child.shape[0])
    child = child + mask * noise
    return clip_vector(child, bounds)


def propose_with_optimizer(
    optimizer: str,
    history_X: List[np.ndarray],
    history_y: List[float],
    rng: np.random.Generator,
    bounds: np.ndarray,
) -> np.ndarray:
    opt = optimizer.upper()
    if opt == "GA":
        return propose_ga(history_X, history_y, rng, bounds)
    return propose_bo(history_X, history_y, rng, bounds)


def run_single_simulation(params: Dict[str, float], sim_dir: Path) -> Dict[str, Any]:
    # Always run serially: one var.spice write + one ngspice call.
    Simulate.write_varspice(str(sim_dir), params)
    Simulate.run_ngspice(str(sim_dir))
    return Simulate.extract_specs(str(sim_dir))


def _code_blocks(text: str) -> List[str]:
    return re.findall(r"```(?:json|python|text|txt)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_like(raw: str) -> Optional[Any]:
    if not raw:
        return None
    candidates = [raw.strip()]
    obj_text = _first_balanced_object(raw)
    if obj_text:
        candidates.append(obj_text.strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    return None


RELATION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|<|>)\s*(.+?)\s*$")
RHS_NUM_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
RHS_VAR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
RHS_K_MUL_VAR_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
RHS_VAR_MUL_K_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def _eval_rhs(rhs: str, params: Dict[str, float]) -> Optional[float]:
    m = RHS_NUM_RE.match(rhs)
    if m:
        return float(m.group(1))
    m = RHS_VAR_RE.match(rhs)
    if m:
        key = m.group(1)
        if key in params:
            return float(params[key])
        return None
    m = RHS_K_MUL_VAR_RE.match(rhs)
    if m:
        k = float(m.group(1))
        key = m.group(2)
        if key in params:
            return k * float(params[key])
        return None
    m = RHS_VAR_MUL_K_RE.match(rhs)
    if m:
        key = m.group(1)
        k = float(m.group(2))
        if key in params:
            return float(params[key]) * k
        return None
    return None


def apply_relations_to_params(
    params: Dict[str, float],
    relations: List[str],
    param_names: List[str],
    bounds: np.ndarray,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not relations:
        return dict(params), []

    adjusted = dict(params)
    name_to_idx = {n: i for i, n in enumerate(param_names)}
    applied: List[Dict[str, Any]] = []
    eps = 1e-9

    # Multiple passes to propagate chained dependencies.
    for _ in range(3):
        for expr in relations:
            m = RELATION_RE.match(expr)
            if not m:
                continue
            lhs, op, rhs = m.group(1), m.group(2), m.group(3)
            if lhs not in adjusted or lhs not in name_to_idx:
                continue
            rhs_val = _eval_rhs(rhs, adjusted)
            if rhs_val is None or not np.isfinite(rhs_val):
                continue

            old = float(adjusted[lhs])
            new = old
            if op == "==":
                new = rhs_val
            elif op == ">=":
                new = max(old, rhs_val)
            elif op == ">":
                new = max(old, rhs_val + eps)
            elif op == "<=":
                new = min(old, rhs_val)
            elif op == "<":
                new = min(old, rhs_val - eps)

            idx = name_to_idx[lhs]
            lo, hi = float(bounds[idx, 0]), float(bounds[idx, 1])
            new = float(np.clip(new, lo, hi))
            if is_integer_param(lhs):
                new = float(int(round(new)))
                new = float(np.clip(new, lo, hi))

            if abs(new - old) > 1e-15:
                adjusted[lhs] = new
                applied.append({"relation": expr, "lhs": lhs, "old": old, "new": new})

    return adjusted, applied


def parse_llm_plan(response: str, default_optimizer: str = "BO") -> LLMPlan:
    parsed_obj: Optional[Dict[str, Any]] = None
    for block in _code_blocks(response) + [response]:
        obj = _parse_json_like(block)
        if isinstance(obj, dict):
            parsed_obj = obj
            break

    if parsed_obj is None:
        return LLMPlan(default_optimizer.upper(), [], {}, [], response)

    raw_optimizer = str(parsed_obj.get("optimizer", default_optimizer)).upper().strip()
    optimizer = "GA" if raw_optimizer == "GA" else "BO"

    samples_raw = parsed_obj.get("samples", [])
    samples: List[Dict[str, Any]] = []
    if isinstance(samples_raw, dict):
        samples_raw = [samples_raw]
    if isinstance(samples_raw, list):
        for item in samples_raw:
            if isinstance(item, dict):
                samples.append(item)

    ranges: Dict[str, Tuple[float, float]] = {}
    ranges_raw = parsed_obj.get("ranges", {})
    if isinstance(ranges_raw, dict):
        for k, v in ranges_raw.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    lo = float(v[0])
                    hi = float(v[1])
                    ranges[str(k)] = (lo, hi)
                except Exception:
                    continue

    relations: List[str] = []
    relations_raw = parsed_obj.get("relations", parsed_obj.get("constraints", []))
    if isinstance(relations_raw, str):
        relations_raw = [relations_raw]
    if isinstance(relations_raw, list):
        for item in relations_raw:
            if isinstance(item, str) and item.strip():
                relations.append(item.strip())

    return LLMPlan(optimizer=optimizer, samples=samples, ranges=ranges, relations=relations, raw_response=response)


def _build_llm_prompt(
    stage: str,
    n_samples: int,
    spec_text: str,
    constraints: Dict[str, Dict[str, float]],
    topology: str,
    param_names: List[str],
    bounds: np.ndarray,
    hard_bounds: np.ndarray,
    rag_context: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
    default_optimizer: str,
) -> str:
    range_lines = []
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        hlo, hhi = hard_bounds[i]
        range_lines.append(f"- {name}: current=[{lo}, {hi}], hard=[{hlo}, {hhi}]")
    history_text = json.dumps(history_tail, ensure_ascii=False, indent=2) if history_tail else "[]"
    best_text = json.dumps(best_result, ensure_ascii=False, indent=2) if best_result else "{}"
    constraints_text = json.dumps(constraints, ensure_ascii=False, indent=2)
    rag_text = rag_context.strip() if rag_context else ""
    range_infer_code = """def infer_param_range(name: str) -> Tuple[float, float]:
    upper = name.upper()
    if upper.endswith("_L") or "_L_" in upper:
        return (0.15, 1.0)
    if upper.endswith("_W") or "_W_" in upper:
        return (0.5, 10.0)
    if upper.endswith("_M") or "_M_" in upper:
        return (1.0, 100.0)
    if "CAPACITOR" in upper or upper.startswith("CCOMP"):
        return (1e-12, 1e-10)
    if "RESISTOR" in upper:
        return (1.0, 1e6)
    if "CURRENT" in upper or upper.startswith("IB"):
        return (5e-7, 2e-5)
    if upper.startswith("VB") or upper.startswith("VBIAS") or upper.startswith("VREF"):
        return (0.0, 1.8)
    if upper.startswith("V"):
        return (0.0, 1.8)
    return (0.15, 10.0)"""
    return f"""
You are optimizing analog op-amp sizing with mixed LLM + numerical optimization.
This is stage: {stage}.

Spec text:
{spec_text}

Hard JSON constraints for pass/fail:
{constraints_text}

Current topology netlist:
{topology}

Retrieved knowledge context (RAG, may be empty):
{rag_text}

Current best known result:
{best_text}

Recent simulation history:
{history_text}

Parameter names and ranges:
{chr(10).join(range_lines)}

Range inference rule used by the system (follow this exactly):
```python
{range_infer_code}
```

Task:
1) Choose optimizer for the next numerical proposal. Must be "BO" or "GA".
2) Propose exactly {n_samples} sizing samples for initialization/refresh.
3) Each sample must include every parameter exactly once with valid numeric values in range.
4) Propose a NEW range for every parameter as the next numerical-tool search range.
   Range must satisfy: hard_min <= new_min < new_max <= hard_max.
   You must provide ranges for all parameters in every response.
5) You may optionally provide sizing relations/constraints for variables, such as:
   - "P2_W == 2 * P1_W"
   - "N3_M >= N1_M"
   - "VB2 > VB1"

Return EXACTLY ONE JSON object in one code block and nothing else:
```json
{{
  "optimizer": "BO",
  "ranges": {{
    "PARAM_A": [0.0, 1.0],
    "PARAM_B": [0.0, 1.0]
  }},
  "relations": [
    "PARAM_B >= PARAM_A",
    "PARAM_C == 2 * PARAM_A"
  ],
  "samples": [
    {{"PARAM_A": 0.0, "PARAM_B": 0.0}}
  ]
}}
```
Note: You can choose either "BO" or "GA" for optimizer.
""".strip()


def request_llm_plan(
    mymodel: str,
    stage: str,
    n_samples: int,
    spec_text: str,
    constraints: Dict[str, Dict[str, float]],
    topology: str,
    param_names: List[str],
    bounds: np.ndarray,
    hard_bounds: np.ndarray,
    rag_context: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
    default_optimizer: str,
) -> LLMPlan:
    prompt = _build_llm_prompt(
        stage=stage,
        n_samples=n_samples,
        spec_text=spec_text,
        constraints=constraints,
        topology=topology,
        param_names=param_names,
        bounds=bounds,
        hard_bounds=hard_bounds,
        rag_context=rag_context,
        history_tail=history_tail,
        best_result=best_result,
        default_optimizer=default_optimizer,
    )
    try:
        response = askLLM(prompt, mymodel)
        print(response)
    except Exception as e:
        return LLMPlan(default_optimizer.upper(), [], {}, [], f"LLM_ERROR: {e}")
    return parse_llm_plan(response, default_optimizer=default_optimizer)


def sanitize_bounds_update(
    proposed_ranges: Dict[str, Tuple[float, float]],
    param_names: List[str],
    current_bounds: np.ndarray,
    hard_bounds: np.ndarray,
) -> np.ndarray:
    new_bounds = current_bounds.copy()
    for i, name in enumerate(param_names):
        if name not in proposed_ranges:
            continue
        lo_raw, hi_raw = proposed_ranges[name]
        hlo, hhi = hard_bounds[i]
        lo = float(np.clip(lo_raw, hlo, hhi))
        hi = float(np.clip(hi_raw, hlo, hhi))
        if lo > hi:
            lo, hi = hi, lo
        # guarantee non-zero span
        if abs(hi - lo) < 1e-15:
            eps = max((hhi - hlo) * 1e-6, 1e-12)
            lo = max(hlo, lo - eps)
            hi = min(hhi, hi + eps)
            if lo >= hi:
                lo, hi = hlo, hhi
        new_bounds[i, 0] = lo
        new_bounds[i, 1] = hi
    return new_bounds


def _sanitize_samples(
    samples: List[Dict[str, Any]],
    target_count: int,
    param_names: List[str],
    bounds: np.ndarray,
    rng: np.random.Generator,
    fallback_params: Optional[Dict[str, float]] = None,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for sample in samples:
        out.append(dict_to_vector(sample, param_names, bounds, rng, fallback=fallback_params))
        if len(out) >= target_count:
            return out
    while len(out) < target_count:
        out.append(sample_uniform(1, rng, bounds)[0])
    return out


def _history_tail(records: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    return records[-k:] if len(records) > k else records


def _best_designs(records: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(records, key=lambda r: float(r.get("score", -1e18)), reverse=True)
    best: List[Dict[str, Any]] = []
    for rec in ranked[: max(0, top_n)]:
        best.append(
            {
                "sim": rec.get("sim"),
                "score": rec.get("score"),
                "source": rec.get("source"),
                "optimizer": rec.get("optimizer"),
                "params": rec.get("params"),
                "result": rec.get("result"),
                "violations": rec.get("violations"),
            }
        )
    return best


def build_retrieval_query(
    mymodel: str,
    spec_text: str,
    constraints: Dict[str, Dict[str, float]],
    topology: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
) -> str:
    prompt = f"""
You are preparing one concise retrieval query for analog op-amp sizing.
Focus on parameter sizing strategy, not topology generation.

Spec text:
{spec_text}

Hard constraints:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Current topology:
{topology}

Best result so far:
{json.dumps(best_result, ensure_ascii=False, indent=2) if best_result else "{}"}

Recent history:
{json.dumps(history_tail, ensure_ascii=False, indent=2) if history_tail else "[]"}

Return exactly one query in one text code block:
```text
your retrieval query here
```
""".strip()
    response = askLLM(prompt, mymodel)
    blocks = _code_blocks(response)
    if blocks:
        q = blocks[0].strip()
        if q:
            return q
    q = response.strip()
    return q if q else "Op-amp sizing strategy to satisfy multi-objective specs in SKY130."


def get_retrieval_context(
    mymodel: str,
    spec_text: str,
    constraints: Dict[str, Dict[str, float]],
    topology: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
    rag_top_k: int = 5,
) -> Tuple[str, str]:
    try:
        import knowledge
    except Exception as e:
        return "", f"RAG skipped: import knowledge failed ({e})"

    try:
        query = build_retrieval_query(
            mymodel=mymodel,
            spec_text=spec_text,
            constraints=constraints,
            topology=topology,
            history_tail=history_tail,
            best_result=best_result,
        )
        
    except Exception as e:
        return "", f"RAG skipped: query generation failed ({e})"

    try:
        context = knowledge.getKnowledge(query, mytop_k=rag_top_k, type="parameter sizing")
        return context or "", f"RAG query: {query}"
    except Exception as e:
        return "", f"RAG retrieval failed for query [{query}] ({e})"


def sizing_mix(
    topology_id: int,
    spec_id: int,
    mymodel: str = "gemini-2.5-flash",
    max_simulations: int = 1000,
    llm_budget: int = 400,
    feedback_interval: int = 20,
    initial_llm_samples: int = 5,
    refresh_llm_samples: int = 5,
    use_retrieval: bool = True,
    rag_top_k: int = 5,
    seed: int = 42,
    sim_dir: Path = DEFAULT_SIM_DIR,
    template_netlist_path: Path = DEFAULT_TEMPLATE_NETLIST_PATH,
    new_netlist_dir: Path = DEFAULT_NEW_NETLIST_DIR,
    tasks_dir: Path = DEFAULT_TASKS_DIR,
    tasks_json_path: Path = DEFAULT_TASKS_JSON_PATH,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    merged_topology = compose_topology_into_sim(
        topology_id=topology_id,
        new_netlist_dir=new_netlist_dir,
        template_netlist_path=template_netlist_path,
    )
    spec_text = load_spec_text(spec_id=spec_id, tasks_dir=tasks_dir)
    constraints = load_spec_constraints(spec_id=spec_id, tasks_json_path=tasks_json_path)

    param_names = extract_param_names(merged_topology)
    hard_bounds = build_bounds(param_names)
    bounds = hard_bounds.copy()
    log_path = Path(__file__).resolve().parent / f"sizing_mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    history_X: List[np.ndarray] = []
    history_y: List[float] = []
    sim_records: List[Dict[str, Any]] = []
    pending_queue: List[Tuple[str, np.ndarray]] = []
    active_optimizer = "BO"
    active_relations: List[str] = []
    llm_enabled = llm_budget > 0

    best_score = -1e18
    best_params: Optional[Dict[str, float]] = None
    best_perf: Optional[Dict[str, Any]] = None
    solved_params: Optional[Dict[str, float]] = None
    solved_perf: Optional[Dict[str, Any]] = None
    met_specs = False

    init_rag_context = ""
    init_rag_debug = "LLM disabled"
    init_llm_response = "LLM disabled"
    if llm_enabled:
        init_rag_debug = "RAG disabled"
        if use_retrieval:
            init_rag_context, init_rag_debug = get_retrieval_context(
                mymodel=mymodel,
                spec_text=spec_text,
                constraints=constraints,
                topology=merged_topology,
                history_tail=[],
                best_result=None,
                rag_top_k=rag_top_k,
            )

        init_plan = request_llm_plan(
            mymodel=mymodel,
            stage="initial",
            n_samples=initial_llm_samples,
            spec_text=spec_text,
            constraints=constraints,
            topology=merged_topology,
            param_names=param_names,
            bounds=bounds,
            hard_bounds=hard_bounds,
            rag_context=init_rag_context,
            history_tail=[],
            best_result=None,
            default_optimizer=active_optimizer,
        )
        init_llm_response = init_plan.raw_response
        active_optimizer = init_plan.optimizer
        active_relations = init_plan.relations
        bounds = sanitize_bounds_update(
            proposed_ranges=init_plan.ranges,
            param_names=param_names,
            current_bounds=bounds,
            hard_bounds=hard_bounds,
        )
        init_vectors = _sanitize_samples(
            samples=init_plan.samples,
            target_count=initial_llm_samples,
            param_names=param_names,
            bounds=bounds,
            rng=rng,
            fallback_params=None,
        )
        pending_queue.extend([("llm_init", x) for x in init_vectors])

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"topology_id={topology_id}, spec_id={spec_id}, model={mymodel}\n")
        log.write(
            f"max_simulations={max_simulations}, llm_budget={llm_budget}, "
            f"feedback_interval={feedback_interval}\n"
        )
        log.write(f"active_optimizer(init)={active_optimizer}\n")
        log.write(f"active_relations(init)={json.dumps(active_relations, ensure_ascii=False)}\n")
        log.write(f"active_bounds(init)={json.dumps({n: [float(bounds[i,0]), float(bounds[i,1])] for i,n in enumerate(param_names)}, ensure_ascii=False)}\n")
        log.write(f"init_rag_debug={init_rag_debug}\n")
        log.write(f"initial_llm_response:\n{init_llm_response}\n\n")

        sim_count = 0
        while sim_count < max_simulations:
            if llm_enabled and sim_count >= llm_budget:
                
                llm_enabled = False
                pending_queue = []
                # bounds = hard_bounds.copy()
                # active_relations = []
                log.write(
                    f"[switch@{sim_count}] pure_numerical_mode=true, "
                    "ignore_llm_ranges=true, ignore_llm_relations=true, "
                    "keep_optimizer=true, query_llm=false\n"
                )
                log.write(
                    f"[switch@{sim_count}] active_bounds -> "
                    f"{json.dumps({n: [float(bounds[i,0]), float(bounds[i,1])] for i,n in enumerate(param_names)}, ensure_ascii=False)}\n"
                )

            if pending_queue:
                source, x = pending_queue.pop(0)
            else:
                x = propose_with_optimizer(active_optimizer, history_X, history_y, rng, bounds)
                source = f"{active_optimizer.lower()}_proposal"

            params = vector_to_params(x, param_names, bounds)
            params, relation_applied = apply_relations_to_params(
                params=params,
                relations=active_relations,
                param_names=param_names,
                bounds=bounds,
            )
            result = run_single_simulation(params, sim_dir=sim_dir)
            score = score_result(result, constraints)
            violations = get_violations(result, constraints)
            x_effective = params_to_vector(params, param_names, bounds)

            history_X.append(x_effective.copy())
            history_y.append(score)
            sim_count += 1

            record = {
                "sim": sim_count,
                "source": source,
                "optimizer": active_optimizer,
                "score": score,
                "params": params,
                "result": result,
                "violations": violations,
                "relations": active_relations,
                "relation_applied": relation_applied,
            }
            sim_records.append(record)

            log.write(f"[sim {sim_count}] source={source}, optimizer={active_optimizer}, score={score:.6f}\n")
            log.write(f"params={json.dumps(params, ensure_ascii=False)}\n")
            if relation_applied:
                log.write(f"RELATION_APPLIED={json.dumps(relation_applied, ensure_ascii=False)}\n")
            log.write(f"result={json.dumps(result, ensure_ascii=False)}\n")
            log.write(f"VIOLATIONS={json.dumps(violations, ensure_ascii=False)}\n")

            if score > best_score:
                best_score = score
                best_params = params
                best_perf = result

            if result_meets_constraints(result, constraints):
                met_specs = True
                solved_params = params
                solved_perf = result
                log.write(f"PASS at sim={sim_count}\n")
                break

            if (
                llm_enabled
                and sim_count % feedback_interval == 0
                and sim_count < max_simulations
                and sim_count < llm_budget
            ):
                refresh_rag_context = ""
                refresh_rag_debug = "RAG disabled"
                if use_retrieval:
                    refresh_rag_context, refresh_rag_debug = get_retrieval_context(
                        mymodel=mymodel,
                        spec_text=spec_text,
                        constraints=constraints,
                        topology=merged_topology,
                        history_tail=_history_tail(sim_records, 8),
                        best_result={
                            "best_score": best_score,
                            "best_params": best_params,
                            "best_result": best_perf,
                        },
                        rag_top_k=rag_top_k,
                    )

                refresh_plan = request_llm_plan(
                    mymodel=mymodel,
                    stage=f"refresh_after_{sim_count}_sims",
                    n_samples=refresh_llm_samples,
                    spec_text=spec_text,
                    constraints=constraints,
                    topology=merged_topology,
                    param_names=param_names,
                    bounds=bounds,
                    hard_bounds=hard_bounds,
                    rag_context=refresh_rag_context,
                    history_tail=_history_tail(sim_records, 8),
                    best_result={
                        "best_score": best_score,
                        "best_params": best_params,
                        "best_result": best_perf,
                    },
                    default_optimizer=active_optimizer,
                )
                active_optimizer = refresh_plan.optimizer
                active_relations = refresh_plan.relations
                bounds = sanitize_bounds_update(
                    proposed_ranges=refresh_plan.ranges,
                    param_names=param_names,
                    current_bounds=bounds,
                    hard_bounds=hard_bounds,
                )
                log.write(f"[feedback@{sim_count}] active_optimizer -> {active_optimizer}\n")
                log.write(f"[feedback@{sim_count}] active_relations -> {json.dumps(active_relations, ensure_ascii=False)}\n")
                log.write(
                    f"[feedback@{sim_count}] active_bounds -> "
                    f"{json.dumps({n: [float(bounds[i,0]), float(bounds[i,1])] for i,n in enumerate(param_names)}, ensure_ascii=False)}\n"
                )
                log.write(f"[feedback@{sim_count}] rag_debug -> {refresh_rag_debug}\n")
                log.write(f"[feedback@{sim_count}] llm_response:\n{refresh_plan.raw_response}\n")

                # Add one numerical proposal immediately, then add LLM proposals.
                numerical_next = propose_with_optimizer(active_optimizer, history_X, history_y, rng, bounds)
                pending_queue.append((f"{active_optimizer.lower()}_next", numerical_next))

                llm_vectors = _sanitize_samples(
                    samples=refresh_plan.samples,
                    target_count=refresh_llm_samples,
                    param_names=param_names,
                    bounds=bounds,
                    rng=rng,
                    fallback_params=best_params,
                )
                pending_queue.extend([("llm_refresh", x_new) for x_new in llm_vectors])

        log.write("\n=== SUMMARY ===\n")
        log.write(f"met_specs={met_specs}\n")
        log.write(f"sim_count={sim_count}\n")
        log.write(f"best_score={best_score}\n")
        log.write(f"best_params={json.dumps(best_params, ensure_ascii=False)}\n")
        log.write(f"best_perf={json.dumps(best_perf, ensure_ascii=False)}\n")
        log.write(f"best_designs={json.dumps(_best_designs(sim_records, top_n=5), ensure_ascii=False)}\n")

    final_params = solved_params if met_specs else best_params
    final_perf = solved_perf if met_specs else best_perf
    best_designs = _best_designs(sim_records, top_n=5)
    return {
        "topology_id": topology_id,
        "spec_id": spec_id,
        "met_specs": met_specs,
        "simulations": len(sim_records),
        "active_optimizer": active_optimizer,
        "result_params": final_params,
        "result_performance": final_perf,
        "best_score": best_score,
        "best_designs": best_designs,
        "log_file": str(log_path),
        "history": sim_records,
    }


def solve_sizing(topology_id: int, spec_id: int, mymodel: str = "gemini-2.5-flash") -> Tuple[Dict[str, float], Dict[str, Any]]:
    result = sizing_mix(topology_id=topology_id, spec_id=spec_id, mymodel=mymodel)
    return result["result_params"], result["result_performance"]


if __name__ == "__main__":
    final = sizing_mix(topology_id=12, spec_id=5, mymodel="gemini-2.5-flash", max_simulations=5000)
    print("met_specs:", final["met_specs"])
    print("simulations:", final["simulations"])
    print("result_params:", json.dumps(final["result_params"], ensure_ascii=False, indent=2))
    print("result_performance:", json.dumps(final["result_performance"], ensure_ascii=False, indent=2))
    print("log_file:", final["log_file"])
