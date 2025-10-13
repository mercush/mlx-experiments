import mlx.core as mx
from mlx_lm.utils import load
from mlx.nn import Module
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import time
# need to make sure these two have the same tokenizers.
TARGET_MODEL = "mlx-community/gemma-3-12b-it-4bit"
DRAFT_MODEL = "mlx-community/gemma-3-270m-it-4bit"
MAX_TOKENS = 500
K = 5

def sample(dist: mx.array) -> mx.array:
    """Sample a token from the probability distribution."""
    token = mx.random.categorical(dist)
    return token


def sample_K(
    context: mx.array, model: Module, cache, eos_token_ids
) -> tuple[mx.array, mx.array]:
    """Samples K tokens from the model. Returns the new tokens and their logprobs"""
    tokens = []
    dists = []
    for _ in range(K):
        dist = model(context[None], cache=cache)[0][-1]  # (vocab,)
        dist = dist - mx.logsumexp(dist)
        tok = sample(dist)
        context = tok[None]
        tokens.append(tok)
        dists.append(dist[None])
        if tok in eos_token_ids:
            break
    return mx.array(tokens, dtype=mx.int32), mx.concat(dists, axis=0)


def rejection_sample_K(
    context: mx.array,
    new_tokens: mx.array,
    new_token_dists: mx.array,
    target_model: Module,
    target_model_cache,
    draft_model_cache,
) -> list:
    n = len(new_tokens)
    target_dists = target_model(
        mx.concat([context, new_tokens[:-1]])[None], cache=target_model_cache
    )[0]  # (seq_len, vocab)
    target_dists = target_dists - mx.logsumexp(target_dists, axis=1)[:, None]
    ret = []
    for i in range(n):
        tok = new_tokens[i]
        draft_dist = new_token_dists[i]
        target_dist = target_dists[-(n - i)][: len(draft_dist)]
        accept_prob = target_dist[tok] - draft_dist[tok]
        if mx.random.uniform() < mx.exp(accept_prob):
            ret.append(tok)
        else:
            trim_prompt_cache(target_model_cache, n - i - 1)
            trim_prompt_cache(draft_model_cache, n - i - 1)
            dist = mx.where(
                target_dist > draft_dist,
                mx.log(mx.exp(target_dist) - mx.exp(draft_dist)),
                -mx.inf,
            )
            dist = dist - mx.logsumexp(dist)
            tok = sample(dist)
            ret.append(tok)
            return ret
    return ret


def speculative_decoder(
    prompt: str,
    target_model,
    target_tokenizer,
    target_model_cache,
    draft_model,
    draft_tokenizer,
    draft_model_cache,
) -> tuple[str, int, float]:
    generated = []
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    context = target_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    context = mx.array(context, mx.int32)  # (seq_len,)
    n = 0
    while len(generated) < MAX_TOKENS:
        new_tokens, new_token_dists = sample_K(
            context, draft_model, draft_model_cache, draft_tokenizer.eos_token_ids
        )
        accepted_tokens = rejection_sample_K(
            context,
            new_tokens,
            new_token_dists,
            target_model,
            target_model_cache,
            draft_model_cache,
        )
        generated += accepted_tokens
        context = accepted_tokens[-1][None]
        if generated[-1] in draft_tokenizer.eos_token_ids:
            generated = generated[:-1]
            break
        n += 1
    return target_tokenizer.decode(generated), len(generated), len(generated) / n


def base(prompt: str, model, tokenizer, cache) -> tuple[str, int]:
    generated = []
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    context = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    context = mx.array(context, mx.int32)  # (seq_len,)
    while len(generated) < MAX_TOKENS:
        dist = model(context[None], cache=cache)[0][-1]
        tok = sample(dist)
        generated.append(tok)
        context = tok[None]
        if tok.item() in tokenizer.eos_token_ids:
            generated = generated[:-1]
            break
    return tokenizer.decode(generated), len(generated)


if __name__ == "__main__":
    prompt = "Write a quicksort algorithm in Python."
    target_model, target_tokenizer = load(TARGET_MODEL)
    draft_model, draft_tokenizer = load(DRAFT_MODEL)
    target_model_cache = make_prompt_cache(target_model)
    draft_model_cache = make_prompt_cache(draft_model)

    # start = time.time()
    # completion, num_tokens, ave_accepted_tokens = speculative_decoder(
    #     prompt,
    #     target_model,
    #     target_tokenizer,
    #     target_model_cache,
    #     draft_model,
    #     draft_tokenizer,
    #     draft_model_cache,
    # )
    # elapsed_time = time.time() - start
    # print(f"{num_tokens / elapsed_time} tokens-per-sec for speculative decoder")
    # print(f"Ave. accepted tokens: {ave_accepted_tokens}")

    start = time.time()
    completion, num_tokens = base(
        prompt, target_model, target_tokenizer, target_model_cache
    )
    elapsed_time = time.time() - start
    print(f"{num_tokens / elapsed_time} tokens-per-sec for base")

    print(completion)
