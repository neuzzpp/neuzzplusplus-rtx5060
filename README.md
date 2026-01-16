# neuzzplusplus-rtx5060 (Docker + TensorFlow/Keras 3, RTX 5060)

This repo is a fork of `boschresearch/neuzzplusplus` updated to run **inside Docker** on
**RTX 5060** with a **new TensorFlow / Keras 3 stack**.

It wires **AFL++** to a **custom ML mutator** (`libml-mutator.so`) that talks to
`scripts/train_cov_oracle.py` over **named pipes (FIFOs)**.

---

## 1) Requirements (host)

- Linux + NVIDIA driver installed
- Docker + NVIDIA Container Toolkit
- An NVIDIA GPU (RTX 5060 in this setup)

Quick GPU check:

```bash
nvidia-smi
```

---

## 2) Build the Docker image

From the repo root:

```bash
docker build -t neuzzpp:cuda128 .
```

---

## 3) Run the dev container

### Create a new container

```bash
docker run -it --gpus all \
  --name neuzzpp-dev \
  -v "$PWD":/workspace \
  neuzzpp:cuda128 bash
```

### Reuse an existing container

```bash
docker start -ai neuzzpp-dev
```

---

## 4) Verify the Python/TensorFlow environment (inside the container)

```bash
which python
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "import keras; print('Keras', keras.__version__)" || true
python -c "import sklearn; print('sklearn ok', sklearn.__version__)" || true

# If you use Poetry in this repo:
poetry run python -c "import tensorflow as tf; print('TF', tf.__version__)"
poetry run python -c "import sklearn; print('sklearn ok', sklearn.__version__)"
```

---

## 5) Run AFL++ with the ML custom mutator

### 5.1 Prepare input seeds

Put some initial seeds in a folder (example):

```bash
mkdir -p /tmp/in_seeds
cp -n ./path/to/your/seeds/* /tmp/in_seeds/ 2>/dev/null || true
ls -la /tmp/in_seeds | head
```

### 5.2 Set environment for the custom mutator

```bash
export AFL_PATH=/workspace/.deps/AFLplusplus
export AFL_CUSTOM_MUTATOR_LIBRARY=/workspace/aflpp-plugins/libml-mutator.so

# Optional but common for Docker:
export AFL_SKIP_CPUFREQ=1
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
```

### 5.3 Run `afl-fuzz`

Replace `./target_binary ... @@` with your real target command.
Make sure `@@` is present.

```bash
$AFL_PATH/afl-fuzz -i /tmp/in_seeds -o /tmp/out_afl -m none -t 3000+ \
  -- ./target_binary --some-arg @@
```

AFL++ will create a session like:

```text
/tmp/out_afl/default/
  queue/
  crashes/
  hangs/
  pipe_to_ml_model
  pipe_from_ml_model
  training.log
```

If everything is wired correctly, after enough seeds are available the ML side will train,
then the AFL++ status line should eventually show activity under the `py/custom/...` counters.

---

## 6) Debug checklist (when ML training/mutations look “stuck”)

These commands help you verify that:
- the FIFOs exist,
- both sides have them open,
- AFL++ is writing seed requests,
- the training process can resolve seed paths.

### 6.1 Check the AFL++ session + FIFOs

```bash
OUT=/tmp/out_afl/default
ls -la "$OUT"
ls -la "$OUT/pipe_to_ml_model" "$OUT/pipe_from_ml_model"

# Quick sanity: FIFOs should show as 'p' in permissions (prw...)
stat "$OUT/pipe_to_ml_model" "$OUT/pipe_from_ml_model"
```

### 6.2 Confirm both processes have the pipes open

```bash
# AFL pid (from fuzzer_stats)
cat "$OUT/fuzzer_stats" | grep '^fuzzer_pid'
AFL_PID=$(awk -F': ' '/^fuzzer_pid/ {print $2}' "$OUT/fuzzer_stats")

# ML training pid
ps aux | grep -E "train_cov_oracle.py" | grep -v grep
ML_PID=$(ps aux | awk '/train_cov_oracle\.py/ && !/awk/ {print $2; exit}')

# Pipe FDs must appear in BOTH processes
ls -l /proc/$AFL_PID/fd | grep -E 'pipe_(to|from)_ml_model' || true
ls -l /proc/$ML_PID/fd  | grep -E 'pipe_(to|from)_ml_model' || true
```

### 6.3 Inspect training logs

```bash
tail -n 200 "$OUT/training.log"
grep -nE "Traceback|ERROR|Exception" "$OUT/training.log" | tail -n 50
```

### 6.4 Verify that a requested seed actually exists

If you see errors like `FileNotFoundError: ... '/id:000013,...'`, the usual cause is
an incorrect join between `seeds_path` and the FIFO line.

Check the actual seed path:

```bash
ls -la "$OUT/queue" | head
ls -la "$OUT/queue"/id:* | head
```

### 6.5 Peek at FIFO traffic (non-destructive)

These can block if nobody is writing/reading; use `timeout`.

```bash
# See if AFL is writing something into the pipe (may block if empty)
timeout 1s head -n 5 "$OUT/pipe_to_ml_model" | cat -A || true
```

### 6.6 Run training manually (only for debugging)

This is useful to isolate Python/TensorFlow issues from AFL++.

```bash
OUT=/tmp/out_afl/default
poetry run python ./scripts/train_cov_oracle.py -f -s 10 \
  "$OUT/pipe_to_ml_model" \
  "$OUT/pipe_from_ml_model" \
  "$OUT/queue" \
  -- ./target_binary --some-arg @@
```

> Note: manual training requires the FIFOs to exist (`mkfifo` is done by the mutator setup).
> If you run training standalone, you can create them yourself:
>
> ```bash
> mkfifo "$OUT/pipe_to_ml_model" "$OUT/pipe_from_ml_model"
> ```

---

## License / upstream

- Upstream project: `boschresearch/neuzzplusplus`
- Original license headers are preserved in the source files.
