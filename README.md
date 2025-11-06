# Multi Agent Frozen Lake
This repo implements a multi agent version of [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

## Environment design
This enviroment implements all features of the original Frozen Lake, but with adaptation for a multi-agent colaborative setting

## Running example
To run an env example, just:
```bash
uv run python src/test_parallel.py
```

or

```bash
uv run python src/test_aec.py
```