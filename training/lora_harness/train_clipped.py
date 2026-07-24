"""Custom LoRA training entrypoint that adds real gradient clipping, which
`mlx_lm.lora`'s packaged CLI does not expose at all. See the training incident in
AI_GAP_PROJECTS_ROADMAP.md §8.8: Copilot LoRA training diverged twice (loss roughly
doubling every ~20 iterations, destroying the model) -- once at the spec's suggested
lr=1e-4, again after halving lr and adding grad-accumulation. With batch-size=2 a
single anomalous training example can produce an outsized gradient that Adam's
momentum then amplifies with nothing capping its magnitude -- the textbook case for
gradient-norm clipping, which mlx_lm.lora's CLI simply doesn't offer.

This module is `mlx_lm.tuner.trainer.train()` verbatim, with ONE addition:
`mlx.optimizers.clip_grad_norm(grad, max_grad_norm)` right before each
`optimizer.update()`. Model loading, LoRA layer conversion, checkpoint resume, and
reporting are all reused unmodified from mlx_lm (monkeypatches `mlx_lm.lora.train`
so `mlx_lm.lora.run()`'s existing setup path is used as-is) -- this file changes
nothing else about the training mechanics, only adds the missing safety clip.

Usage: identical flags to `mlx_lm.lora`, plus --max-grad-norm (default 1.0):
  python3 train_clipped.py --model <BASE> --train --data <DIR> --adapter-path <DIR> \\
    --batch-size 2 --num-layers 16 --iters 800 --learning-rate 1e-4 \\
    --max-seq-length 4608 --grad-checkpoint --resume-adapter-file <CKPT> \\
    --max-grad-norm 1.0
"""
from __future__ import annotations

import functools
import os
import time
import types
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map

import mlx_lm.lora as lora_cli
from mlx_lm.tuner.trainer import (
    TrainingArgs,
    _clear_cache,
    default_loss,
    evaluate,
    grad_checkpoint,
    iterate_batches,
)


def train_clipped(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback=None,
    max_grad_norm: float = 1.0,
):
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    print(f"Starting training (grad-clip max_norm={max_grad_norm})..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, loss)

    grad_accum_steps = args.grad_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            grad, grad_norm = optim.clip_grad_norm(grad, max_grad_norm)
            optimizer.update(model, grad)
            grad = None
        else:
            grad_norm = mx.array(0.0)

        return lvalue, toks, grad, grad_norm

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None
    grad_norm_sum = 0.0
    grad_norm_count = 0

    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
            comm_group=world,
        ),
    ):
        tic = time.perf_counter()
        if val_dataset and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: Val loss {val_loss:.3f}, Val took {val_time:.3f}s",
                    flush=True,
                )
            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {"iteration": it - 1, "val_loss": val_loss, "val_time": val_time}
                )
            tic = time.perf_counter()

        lvalue, toks, grad_accum, gnorm = step(
            batch, grad_accum, it % grad_accum_steps == 0
        )

        losses += lvalue
        n_tokens += toks
        steps += 1
        grad_norm_sum += gnorm.item()
        grad_norm_count += 1
        mx.eval(state, losses, n_tokens, grad_accum)
        _clear_cache(args.clear_cache_threshold)
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tok = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tok) / train_time
            trained_tokens += n_tok
            peak_mem = mx.get_peak_memory() / 1e9
            avg_grad_norm = grad_norm_sum / max(grad_norm_count, 1)
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Avg Grad Norm {avg_grad_norm:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )
            if training_callback is not None:
                training_callback.on_train_loss_report(
                    {
                        "iteration": it,
                        "train_loss": train_loss,
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem,
                    }
                )
            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0
            grad_norm_sum = 0.0
            grad_norm_count = 0

        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = lora_cli.build_parser()
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient-norm clip threshold (this script's addition; "
        "stock mlx_lm.lora has no equivalent flag).",
    )
    parsed = parser.parse_args()
    max_grad_norm = parsed.max_grad_norm
    config = parsed.config
    args = vars(parsed)
    args.pop("max_grad_norm", None)
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as file:
            config = lora_cli.yaml.load(file, lora_cli.yaml_loader)
        for k, v in config.items():
            if args.get(k, None) is None:
                args[k] = v

    for k, v in lora_cli.CONFIG_DEFAULTS.items():
        if args.get(k, None) is None:
            args[k] = v

    lora_cli.train = functools.partial(train_clipped, max_grad_norm=max_grad_norm)
    lora_cli.run(types.SimpleNamespace(**args))


if __name__ == "__main__":
    main()
