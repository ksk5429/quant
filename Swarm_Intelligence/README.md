# Swarm Intelligence — Multi-Agent Fish Network

## Architecture

Each folder is an **independent LLM agent workspace** designed to be opened as its own VS Code + Claude Code session. The human operator runs multiple VS Code windows simultaneously, each with its own Claude Code instance acting as a specialized Fish agent.

```
Swarm_Intelligence/
├── fish_geopolitical/    # Geopolitical analyst persona
├── fish_quant/           # Quantitative analyst persona
├── fish_contrarian/      # Contrarian thinker persona
├── fish_bayesian/        # Bayesian statistician persona
├── fish_journalist/      # Investigative journalist persona
├── fish_domain_expert/   # Domain specialist persona
├── fish_calibrator/      # Calibration specialist persona
└── (add more as needed)
```

## How to Use

### Step 1: Open Each Fish in VS Code
```bash
# Open each Fish folder in its own VS Code window
code Swarm_Intelligence/fish_geopolitical
code Swarm_Intelligence/fish_quant
code Swarm_Intelligence/fish_contrarian
# ... etc
```

### Step 2: Each Fish Has Its Own CLAUDE.md
Each folder contains a `CLAUDE.md` that tells Claude Code:
- What persona to adopt
- Where to read market data (`../shared_state/market_data/`)
- Where to write analyses (`../shared_state/analyses/`)
- How to communicate with other Fish via signals

### Step 3: Master Intelligence Coordinates
The QUANT root folder (this parent) acts as the **GOD node**:
- Writes events to `shared_state/events/`
- Reads all Fish analyses from `shared_state/analyses/`
- Runs the Bayesian aggregation
- Generates trade signals

### Step 4: Communication Protocol

#### Fish → Shared State (write)
Each Fish writes its analysis as a JSON file:
```
shared_state/analyses/{fish_name}_{market_id}_{timestamp}.json
```

#### Master → Fish (broadcast events)
The GOD node writes events:
```
shared_state/events/{timestamp}_event.json
```

#### Fish → Fish (signals)
Fish can write signals for other specific Fish:
```
shared_state/signals/{from_fish}_to_{to_fish}_{timestamp}.json
```

## Adding a New Fish

1. Create a new folder: `fish_newpersona/`
2. Copy the CLAUDE.md template from any existing Fish
3. Customize the persona in CLAUDE.md
4. Open in VS Code and start prompting
