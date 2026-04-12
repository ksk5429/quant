# Shared State — Inter-Agent Communication Layer

This directory is the communication backbone of the K-Fish swarm. All Fish agents and the Master Intelligence read from and write to this directory.

## Directory Structure

```
shared_state/
├── market_data/     # Market questions and metadata (written by Master)
├── events/          # Real-world event injections (written by Master/GOD)
├── analyses/        # Fish agent outputs (written by each Fish)
├── signals/         # Fish-to-Fish direct messages
└── consensus/       # Aggregated swarm predictions (written by Master)
```

## File Naming Conventions

- Market data: `market_{market_id}.json`
- Events: `{YYYYMMDD_HHMMSS}_event.json`
- Analyses: `{fish_name}_{market_id}_{YYYYMMDD_HHMMSS}.json`
- Signals: `{from_fish}_to_{to_fish}_{YYYYMMDD_HHMMSS}.json`
- Consensus: `consensus_{market_id}_{YYYYMMDD_HHMMSS}.json`

## Workflow

1. Master writes market data to `market_data/`
2. Each Fish reads market data, analyzes, writes to `analyses/`
3. Fish read each other's analyses (optional, persona-dependent)
4. Master reads all analyses from `analyses/`, runs aggregation
5. Master writes consensus to `consensus/`
6. When real-world events occur, Master writes to `events/`
7. Fish read events, re-analyze affected markets
