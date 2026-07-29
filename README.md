# scs-learn-video-pipeline

Anonymizes CMU lecture recordings so they can be published openly. Student audio
is muted and their questions replaced with rendered cards; student faces are
pixelated while the instructor is left clear.

PSC grant `cis260220p` — *Privacy-Preserving AI Pipeline for Open Publication of
University Lecture Recordings*.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # audio stages
pip install -r requirements-video.txt    # face anonymization (separate env)
cp .env.sample .env                      # then fill it in

python -m src.pipeline --lecture-dir data/15210-lecture12 --dry-run
```

Every stage takes `--lecture-dir`; `src/pipeline.py` runs them in order.

```
sync → transcription → audio → face_anon → cards → captions → assembly
```

**Read [CLAUDE.md](CLAUDE.md) before running anything on PSC.** It covers the
required interactive sign-in, which stages need a GPU (two of them), which GPUs
`face_anon` cannot run on, the login-node policy, and the safety properties that
are deliberately fail-closed. It is also the instruction file for driving this
repo with Claude Code.
