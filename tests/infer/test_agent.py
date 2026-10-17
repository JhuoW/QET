"""E1.4: the inference agent (SPEC §5.6 c, §7, D-006). A scripted policy following any legal order succeeds and sees,
at every step, exactly the training prompt of the state it is in; invalid actions end the episode; greedy and
8-sample decoding run."""

import random

import pytest
import torch

from qet.core.canonical import action_string
from qet.core.lattice import node_masks, random_topological_order
from qet.envs.toy import ToyEnv, compile_toy, make_tasks
from qet.infer.agent import ScriptedPolicy, prompt_sha256, run_episode
from qet.infer.decoding import Action, DecodeConfig, Generation, HFSampler, Invalid, parse_action
from qet.train.chat import ChatFormat
from qet.train.methods import method
from qet.train.scoring import Pair, forward_logp
from qet.train.trainer import PackBuilder

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def fmt():
    return ChatFormat.load(MODEL_ID)


def toys():
    return make_tasks("functions", range(12), n_core=5, n_given=2) + make_tasks("stateful", range(12))


def actions_for(record, order):
    nodes = [record.nodes[v] for v in order]
    return [action_string(nd.tool, nd.args) for nd in nodes] + [
        action_string(record.final["tool"], record.final["args"])
    ]


def test_legal_orders_succeed_and_see_the_training_prompt_of_every_state(fmt):
    rng = random.Random(0)
    for toy in toys():
        record, _ = compile_toy(toy)
        pack = PackBuilder(fmt, toy.contracts, method("6"), run_seed=0).build(record)
        training = {s: prompt_sha256(ids) for s, ids in pack.prompt_ids.items()}
        masks = node_masks(record.n, record.preds())
        for _ in range(5):
            order = random_topological_order(masks, rng)
            episode = run_episode(ToyEnv(toy), ScriptedPolicy(fmt, actions_for(record, order)), fmt)
            assert (episode.end, episode.success, episode.calls) == ("finish", True, record.n), toy.task_id
            state = 0
            for k, step in enumerate(episode.steps):
                assert step.prompt_sha256 == training[state], (toy.task_id, order, k)
                if k < record.n:
                    state |= 1 << order[k]


@pytest.mark.parametrize("config_id", ["1", "a1"])
def test_order_sensitive_renderers_see_their_training_prompts_along_the_demo_order(fmt, config_id):
    m = method(config_id)
    for toy in toys()[:6] + toys()[12:18]:
        record, _ = compile_toy(toy)
        pack = PackBuilder(fmt, toy.contracts, m, run_seed=0).build(record)
        order = list(range(record.n))
        episode = run_episode(ToyEnv(toy), ScriptedPolicy(fmt, actions_for(record, order)), fmt, renderer=m.renderer)
        assert episode.success
        states = [sum(1 << v for v in order[:k]) for k in range(record.n + 1)]
        assert [s.prompt_sha256 for s in episode.steps] == [prompt_sha256(pack.prompt_ids[s]) for s in states]


INVALID = [
    "hello",
    '{"tool": "no_such_tool", "args": {}}',
    '{"tool": "finish", "args": [1]}',
    '{"tool": "finish"}',
    '{"tool": "finish", "args": {}, "extra": 1}',
    '{"tool": "finish", "args": {"answer": NaN}}',
    '{"tool": "finish", "args": {"answer": 1e400}}',
    '{"tool": "finish", "args": {"answer": 1}} and more',
    '{"tool": "finish", "tool": "finish", "args": {}}',
    '```json\n{"tool": "finish", "args": {"answer": 1}}\n```',
    '{"tool": "finish", "args": {"answer": "<|im_start|>"}}',
    '[{"tool": "finish", "args": {}}]',
    '{"tool": 7, "args": {}}',
    '{"tool": "finish", "args": {"answer": "\\ud800"}}',  # a lone surrogate through a JSON escape
    '{"tool": "finish", "args": {"answer": "\\u003c|im_end|\\u003e"}}',  # special-token text, escaped
    '{"tool": "finish", "args": {"answer": ' + "[" * 5000 + "]" * 5000 + "}}",  # nested too deeply
]


@pytest.mark.parametrize("raw", INVALID)
def test_invalid_actions_end_the_episode(fmt, raw):
    toy = toys()[0]
    env = ToyEnv(toy)

    class Raw:
        def generate(self, prompt_ids, seed):
            return Generation(fmt.encode(raw) + [fmt.eot_id], True, 0.0)

    episode = run_episode(env, Raw(), fmt)
    assert (episode.end, episode.success, episode.calls, len(episode.steps)) == ("invalid_action", False, 0, 1)
    assert episode.steps[0].error and env.snapshot()[1] == 0  # the environment was never called


def test_an_unfinished_output_is_invalid(fmt):
    class Cut:
        def generate(self, prompt_ids, seed):
            return Generation(fmt.encode('{"tool": "finish", "args": {"answer": 1}}'), False, 0.0)

    episode = run_episode(ToyEnv(toys()[0]), Cut(), fmt)
    assert episode.end == "invalid_action" and "end of turn" in episode.steps[0].error


def test_parsing_is_lenient_about_spacing_and_key_order():
    tools = ["f", "finish"]
    assert parse_action('  {"args": {"b": 2.0, "a": 1},\n  "tool": "f" } ', tools) == Action("f", {"a": 1, "b": 2})
    assert parse_action('{"tool":"finish","args":{"answer":"é"}}', tools) == Action("finish", {"answer": "é"})
    assert isinstance(parse_action('{"tool":"g","args":{}}', tools), Invalid)


def test_the_budget_ends_the_episode(fmt):
    toy = toys()[0]
    env = ToyEnv(toy)
    call = action_string(toy.calls[0]["tool"], toy.calls[0]["args"])
    budget = env.step_budget()
    episode = run_episode(env, ScriptedPolicy(fmt, [call] * (budget + 1)), fmt)
    assert (episode.end, episode.calls, len(episode.steps)) == ("budget_exhausted", budget, budget + 1)
    assert episode.steps[-1].error == "budget exhausted" and episode.steps[-1].observation is None


def test_a_negative_budget_is_refused(fmt):
    with pytest.raises(ValueError, match="budget"):
        run_episode(ToyEnv(toys()[0]), ScriptedPolicy(fmt, []), fmt, budget=-1)


def test_resuming_from_a_prefix(fmt):
    toy = toys()[13]
    record, _ = compile_toy(toy)
    env = ToyEnv(toy)
    prefix = [{"tool": nd.tool, "args": nd.args, "obs": env.step(nd.tool, nd.args)} for nd in record.nodes[:2]]
    rest = actions_for(record, range(2, record.n))
    episode = run_episode(env, ScriptedPolicy(fmt, rest), fmt, prefix=prefix)
    assert episode.success and episode.prefix == 2 and episode.calls == record.n - 2


# ----------------------------------------------------------------------------- decoding with a model
def tiny(fmt):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    sizes = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}
    cfg = Qwen2Config(
        vocab_size=151936, num_key_value_heads=2, initializer_range=0.3, attn_implementation="sdpa", **sizes
    )
    return Qwen2ForCausalLM(cfg).to(torch.float64).eval()


def test_greedy_and_8_sample_decoding_run(fmt):
    model = tiny(fmt)
    toy = toys()[0]
    greedy = HFSampler(model, fmt.eot_id, DecodeConfig("greedy", max_new_tokens=12))
    first = run_episode(ToyEnv(toy), greedy, fmt)
    again = run_episode(ToyEnv(toy), greedy, fmt, rollout=5)
    assert first.end in ("invalid_action", "finish", "budget_exhausted") and first.steps[0].output_tokens > 0
    assert first.steps[0].raw_output == again.steps[0].raw_output  # greedy ignores the seed
    sampler = HFSampler(model, fmt.eot_id, DecodeConfig("sample", temperature=1.0, top_p=1.0, max_new_tokens=12))
    rollouts = [run_episode(ToyEnv(toy), sampler, fmt, rollout=r, run_seed=3) for r in range(8)]
    assert len({e.steps[0].raw_output for e in rollouts}) > 1  # independent rollouts
    repeat = run_episode(ToyEnv(toy), sampler, fmt, rollout=4, run_seed=3)
    assert repeat.steps[0].raw_output == rollouts[4].steps[0].raw_output  # reproducible


def test_decoding_log_probability_equals_the_scorer(fmt):
    model = tiny(fmt)
    prompt = fmt.prompt_ids([{"role": "system", "content": "s"}, {"role": "user", "content": "[Task] go"}])
    for decode in (DecodeConfig("greedy", max_new_tokens=10), DecodeConfig("sample", max_new_tokens=10),
                   DecodeConfig("sample", top_p=0.9, temperature=0.7, max_new_tokens=10)):  # fmt: skip
        gen = HFSampler(model, fmt.eot_id, decode).generate(prompt, seed=11)
        scored = forward_logp(model, [Pair(tuple(prompt), tuple(gen.ids))], pad_id=fmt.eot_id).item()
        assert abs(scored - gen.logprob) <= 1e-9 * abs(scored)  # the sampler and the scorer see the same model
    with torch.no_grad():
        logits = model(torch.tensor([prompt])).logits[0, -1]
    greedy = HFSampler(model, fmt.eot_id, DecodeConfig("greedy", max_new_tokens=1)).generate(prompt, seed=0)
    assert greedy.ids == [int(logits.argmax())]
    narrow = HFSampler(model, fmt.eot_id, DecodeConfig("sample", top_p=1e-12, max_new_tokens=5)).generate(prompt, 9)
    full_greedy = HFSampler(model, fmt.eot_id, DecodeConfig("greedy", max_new_tokens=5)).generate(prompt, 0)
    assert narrow.ids == full_greedy.ids  # a vanishing nucleus keeps only the top token


@pytest.mark.gpu
def test_greedy_and_8_sample_decoding_with_the_real_backbone():
    """Qwen2.5-0.5B-Instruct in bf16 on one GPU, untrained: the decoding paths run end to end (no claims)."""
    from transformers import AutoModelForCausalLM

    from qet.train.chat import backbone

    fmt = ChatFormat.load(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=backbone(MODEL_ID).revision,
                                                 dtype=torch.bfloat16).to("cuda").eval()  # fmt: skip
    ends = []
    for toy in make_tasks("functions", range(3), n_core=3, n_given=2) + make_tasks("stateful", range(2)):
        greedy = run_episode(ToyEnv(toy), HFSampler(model, fmt.eot_id, DecodeConfig("greedy", max_new_tokens=96)), fmt)
        sampled = [
            run_episode(
                ToyEnv(toy), HFSampler(model, fmt.eot_id, DecodeConfig("sample", max_new_tokens=96)), fmt, rollout=r
            )  # fmt: skip
            for r in range(8)
        ]
        ends += [greedy.end] + [e.end for e in sampled]
        print(toy.task_id, "greedy:", greedy.end, greedy.success, repr(greedy.steps[0].raw_output[:80]),
              "| sampled successes:", sum(e.success for e in sampled), "/ 8")  # fmt: skip
    assert len(ends) == 45 and set(ends) <= {"finish", "invalid_action", "budget_exhausted"}
