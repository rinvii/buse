# buse

[![CI](https://github.com/rinvii/buse/actions/workflows/ci.yml/badge.svg)](https://github.com/rinvii/buse/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/buse)](https://pypi.org/project/buse/)
[![Python versions](https://img.shields.io/pypi/pyversions/buse)](https://pypi.org/project/buse/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**buse** (Browser Use CLI) is a stateless, universal CLI driver for the `browser-use` library. It allows AI agents (or humans) to control persistent, background browser instances using standard shell commands and structured output.

## Features

- **Persistent Sessions**: Multiple browser instances (e.g., `browser-1`, `browser-2`) can run simultaneously.
- **Stateless Control**: Every CLI command is atomic—connecting, executing, and returning a "receipt".
- **Universal Primitives**: Includes coordinate clicking, JS execution, and hover support.
- **Vision-Ready**: `observe` outputs minified DOM and a screenshot path.
- **Session Migration**: `save-state` allows exporting cookies/storage for persistent logins.

## Installation

```bash
pip install buse
```

Or from source:

```bash
cd buse
uv pip install -e .
```

## Requirements

- Python 3.12+
- Google Chrome (local install)

## Usage Pattern

`buse <instance_id> <command> [args]`

---

## Command List

### 1. Lifecycle & State

| Command      | Description                             | Example                           |
| :----------- | :-------------------------------------- | :-------------------------------- |
| `<id>`       | Initialize/Start a new browser instance | `buse b1`                         |
| `list`       | Show all active browser instances       | `buse list`                       |
| `stop`       | Stop and kill a browser instance        | `buse b1 stop`                    |
| `save-state` | Export cookies/storage to a file        | `buse b1 save-state cookies.json` |

### 2. Analysis & Extraction

| Command   | Description                                               | Example                              |
| :-------- | :-------------------------------------------------------- | :----------------------------------- |
| `observe` | Get minified DOM (use `--screenshot` to include an image) | `buse b1 observe --screenshot`       |
| `extract` | Use LLM to extract data (set `BUSE_EXTRACT_MODEL`)        | `buse b1 extract "get product info"` |

### 3. Navigation & Interaction

| Command            | Description                                                                                       | Example                                  |
| :----------------- | :------------------------------------------------------------------------------------------------ | :--------------------------------------- |
| `navigate`         | Load a specific URL (supports `--new-tab`)                                                        | `buse b1 navigate "https://google.com"`  |
| `new-tab`          | Open a URL in a new tab (alias for `navigate --new-tab`)                                          | `buse b1 new-tab "https://example.com"`  |
| `search`           | Search the web (engines: `google`, `bing`, `duckduckgo`)                                          | `buse b1 search "query" --engine google` |
| `click`            | Click by index, coordinates, or resolve by `--id`/`--class`                                       | `buse b1 click --x 500 --y 300`          |
| `input`            | Type text into a field by index or `--id`/`--class` (use `--text` when no index)                  | `buse b1 input 12 "Hello"`               |
| `dropdown-options` | List options for a select element by index or `--id`/`--class`                                    | `buse b1 dropdown-options 12`            |
| `select-dropdown`  | Select dropdown option by visible text and index or `--id`/`--class` (use `--text` when no index) | `buse b1 select-dropdown 12 "Option"`    |
| `hover`            | Hover over an element by index or `--id`/`--class`                                                | `buse b1 hover 5`                        |
| `scroll`           | Scroll page or a specific element                                                                 | `buse b1 scroll --down --pages 2`        |
| `refresh`          | Reload the current page                                                                           | `buse b1 refresh`                        |
| `go-back`          | Go back in browser history                                                                        | `buse b1 go-back`                        |
| `wait`             | Wait for N seconds                                                                                | `buse b1 wait 2`                         |
| `evaluate`         | Execute custom JavaScript code                                                                    | `buse b1 evaluate "alert('Hi')"`         |

### 4. Advanced

| Command      | Description             | Example                     |
| :----------- | :---------------------- | :-------------------------- |
| `switch-tab` | Switch by 4-char tab ID | `buse b1 switch-tab "4D39"` |
| `close-tab`  | Close by 4-char tab ID  | `buse b1 close-tab "4D39"`  |

## Examples

```bash
# Start a session
buse b1

# Observe without screenshot (JSON)
buse b1 observe

# Observe with screenshot (JSON + image)
buse b1 observe --screenshot

# Navigate and click by coordinates
buse b1 navigate "https://example.com"
buse b1 click --x 280 --y 220

# Click by id/class fallback
buse b1 click --id "submit-button"
buse b1 click --class "cta-primary"

# Input by id with explicit --text
buse b1 input --id "email" --text "test@example.com"

# Get dropdown options and select by text
buse b1 dropdown-options --id "country"
buse b1 select-dropdown --id "country" --text "Canada"

# Scroll and wait
buse b1 scroll --down --pages 1.5
buse b1 wait 2
```

## Output & Profiling

- `--format json|toon` to switch output format.
- `--profile` (or `-p`) includes timing data in the JSON response.

## Environment Variables

- `BUSE_EXTRACT_MODEL`: model name for `extract` (default: `gpt-4o-mini`).
- `OPENAI_API_KEY`: required for `extract`.
- `BUSE_KEEP_SESSION`: set to `1` to keep the session open within a single process.

Built with `browser-use` and `uv`.
