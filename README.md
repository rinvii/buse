# buse

[![CI](https://github.com/rinvii/buse/actions/workflows/ci.yml/badge.svg)](https://github.com/rinvii/buse/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/buse)](https://pypi.org/project/buse/)
[![Python versions](https://img.shields.io/pypi/pyversions/buse)](https://pypi.org/project/buse/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Control your browser from your terminal.

buse is a stateless CLI designed for AI agents and automation scripts. It turns complex browser interaction into simple, structured command-line primitives.

## Key Features

- **Stateless Control**: Just point the CLI at a browser and go.
- **Persistent Sessions**: Multiple browser instances can run simultaneously.
- **Universal Primitives**: Click, type, scroll, and execute JS with one-liners.
- **Vision-Ready**: `observe` command captures state + screenshots in a single call.
- **Session Migration**: Export cookies/storage via `save-state` to maintain persistent logins.

## Why 'buse'?

Automating a browser usually means writing long, complex scripts or paying for expensive cloud services. buse changes that by letting you control a browser just like any other folder or file on your computer—using simple, one-word commands in your terminal.

For example, open a browser and navigate to a website:

```
uvx --python 3.12 buse browser-1
uvx --python 3.12 buse browser-1 navigate "https://example.com"
uvx --python 3.12 buse browser-2 # open a second browser
uvx --python 3.12 buse browser-2 search "latest tech news"
```

## Installation

With `uv`:

```bash
uvx --python 3.12 buse --help
```

With `pip`:

```bash
pip install buse
```

From source:

```bash
cd buse
uv pip install -e .
```

## Requirements

- Python 3.12
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

#### observe notes

`observe` returns a minified DOM snapshot with element indices usable by actions (click/input/hover/...) like `[12]`.
Those indices are ephemeral: they're only valid for the current page state and may change
after any action (click, input, send-keys, navigate, refresh).
If an index is stale or fails, run `buse <id> observe` to refresh indices. If you need
stability across steps, use `--id` or `--class` instead of indices.

### 3. Navigation & Interaction

| Command            | Description                                                                                         | Example                                  |
| :----------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------------------- |
| `navigate`         | Load a specific URL (supports `--new-tab`)                                                          | `buse b1 navigate "https://google.com"`  |
| `new-tab`          | Open a URL in a new tab (alias for `navigate --new-tab`)                                            | `buse b1 new-tab "https://example.com"`  |
| `search`           | Search the web (engines: `google`, `bing`, `duckduckgo`)                                            | `buse b1 search "query" --engine google` |
| `click`            | Click by index, coordinates, or resolve by `--id`/`--class`                                         | `buse b1 click --x 500 --y 300`          |
| `input`            | Type text into a field by index or `--id`/`--class` (use `--text` when no index)                    | `buse b1 input 12 "Hello"`               |
| `upload-file`      | Upload a file to an element by index                                                                | `buse b1 upload-file 5 "./img.png"`      |
| `send-keys`        | Send special keys or text (use `--list-keys` for names, optional focus with `--index/--id/--class`) | `buse b1 send-keys "Enter"`              |
| `find-text`        | Scroll to specific text on the page                                                                 | `buse b1 find-text "Contact"`            |
| `dropdown-options` | List options for a select element by index or `--id`/`--class`                                      | `buse b1 dropdown-options 12`            |
| `select-dropdown`  | Select dropdown option by visible text and index or `--id`/`--class` (use `--text` when no index)   | `buse b1 select-dropdown 12 "Option"`    |
| `hover`            | Hover over an element by index or `--id`/`--class`                                                  | `buse b1 hover 5`                        |
| `scroll`           | Scroll page or a specific element                                                                   | `buse b1 scroll --down --pages 2`        |
| `refresh`          | Reload the current page                                                                             | `buse b1 refresh`                        |
| `go-back`          | Go back in browser history                                                                          | `buse b1 go-back`                        |
| `wait`             | Wait for N seconds                                                                                  | `buse b1 wait 2`                         |
| `evaluate`         | Execute custom JavaScript code                                                                      | `buse b1 evaluate "alert('Hi')"`         |

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

# Upload a file
buse b1 upload-file 5 "./image.png"

# Send special keys
buse b1 send-keys "Enter"

# Send keys to a focused element
buse b1 send-keys --id "search" "Hello"

# List send-keys names
buse b1 send-keys --list-keys

# Find and scroll to text
buse b1 find-text "Contact Us"

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
- `BUSE_SELECTOR_CACHE_TTL`: selector-map cache TTL in seconds (default: `0`, disabled).
- `BUSE_REMOTE_ALLOW_ORIGINS`: override Chrome `--remote-allow-origins` (default: `http://localhost:<port>,http://127.0.0.1:<port>`).

## References & Inspiration

https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/

https://www.anthropic.com/news/3-5-models-and-computer-use

https://docs.browser-use.com/introduction
