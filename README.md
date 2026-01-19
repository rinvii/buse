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

| Command   | Description                                | Example                              |
| :-------- | :----------------------------------------- | :----------------------------------- |
| `observe` | Snapshot DOM + optionally save screenshots | `buse b1 observe --screenshot`       |
| `extract` | LLM extraction (set `BUSE_EXTRACT_MODEL`)  | `buse b1 extract "get product info"` |

#### observe notes

- DOM indices are ephemeral; refresh with `buse <id> observe` after page changes, or use `--id`/`--class` for stability.
- `observe --omniparser` always captures a screenshot: saves `image.jpg` (input) and `image_som.jpg` (server output) in the screenshots dir or `--path`.
- When available, `screenshot_path` points to `image_som.jpg`. OmniParser `bbox` values are in CSS pixels (not normalized).
- Use `--no-dom` to skip DOM processing and return an empty `dom_minified`.

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
| `scroll`           | Scroll page or a specific element (use `--up` or `--down`)                                          | `buse b1 scroll --up --pages 2`          |
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

### Flag Matrix

Global (all commands):

- `--format` (`json`|`toon`, default: `json`), `-f` alias
- `--profile` (default: `false`), `-p` alias

Commands:

- `list`: no flags
- `<id>`: no flags (start/attach instance)
- `observe`: `--screenshot` (false), `--path` (unset), `--omniparser` (false), `--no-dom` (false)
- `navigate`: `--new-tab` (false)
- `new-tab`: no flags
- `search`: `--engine` (default: `google`)
- `click`: `--x` (unset), `--y` (unset), `--id` (unset), `--class` (unset)
- `input`: `--text` (unset), `--id` (unset), `--class` (unset)
- `upload-file`: no flags
- `send-keys`: `--index` (unset), `--id` (unset), `--class` (unset), `--list-keys` (false)
- `find-text`: no flags
- `dropdown-options`: `--id` (unset), `--class` (unset)
- `select-dropdown`: `--text` (unset), `--id` (unset), `--class` (unset)
- `hover`: `--id` (unset), `--class` (unset)
- `scroll`: `--down/--up` (down default), `--pages` (default: `1.0`), `--index` (unset)
- `refresh`: no flags
- `go-back`: no flags
- `wait`: no flags
- `switch-tab`: no flags
- `close-tab`: no flags
- `save-state`: no flags
- `extract`: no flags
- `evaluate`: no flags
- `stop`: no flags

### Commands

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
buse b1 scroll --up --pages 1
buse b1 wait 2
```

## MCP Server

Expose the active browser instances via the Model Context Protocol.

```
buse mcp-server --host 0.0.0.0 --port 8000
```

- `--transport` selects `streamable-http` (default), `sse`, or `stdio`.
- `--name` changes the MCP server name, `--stateless/--stateful` controls HTTP mode, and `--json-response/--no-json-response` toggles JSON wrapping.
- `--allow-remote` permits non-local clients (default: local-only). `--auth-token` requires `Authorization: Bearer <token>` or `X-Buse-Token` for HTTP requests.
- `--format` (`json`|`toon`, default: `json`), `-f` alias.
- Resources:
  - `buse://sessions` returns a list of session metadata (`instance_id`, `cdp_url`, `user_data_dir`).
  - `buse://session/{id}` returns the metadata for a single session.
- Tools:
  - Supports all CLI actions: `navigate`, `click`, `input_text`, `send_keys`, `scroll`, `switch_tab`, `close_tab`, `search`, `upload_file`, `find_text`, `dropdown_options`, `select_dropdown`, `go_back`, `hover`, `refresh`, `wait`, `save_state`, `extract`, `evaluate`, `stop_session`, `start_session`, `observe`.

The `mcp` SDK ships with buse, so no extra installation is required.

## Output & Profiling

- `--format json|toon` to switch output format.
- `--profile` (or `-p`) includes timing data in the JSON response.

## Environment Variables

- `BUSE_EXTRACT_MODEL`: model name for `extract` (default: `gpt-4o-mini`).
- `OPENAI_API_KEY`: required for `extract`.
- `BUSE_KEEP_SESSION`: set to `1` to keep the session open within a single process.
- `BUSE_SELECTOR_CACHE_TTL`: selector-map cache TTL in seconds (default: `0`, disabled).
- `BUSE_REMOTE_ALLOW_ORIGINS`: override Chrome `--remote-allow-origins` (default: `http://localhost:<port>,http://127.0.0.1:<port>`).
- `BUSE_IMAGE_QUALITY`: JPEG quality (1-100) for OmniParser images.
- `BUSE_MCP_ALLOW_REMOTE`: set to `1` to allow non-local MCP clients.
- `BUSE_MCP_AUTH_TOKEN`: require a Bearer or X-Buse-Token header for MCP HTTP access.

## References & Inspiration

https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/

https://www.anthropic.com/news/3-5-models-and-computer-use

https://docs.browser-use.com/introduction

## Roadmap

- Support all operating systems: Windows, macOS, Linux (right now works on my 10.15 macOS and Windows 11)
- Add automation scripting examples
- Add MCP support
- Add optional daemon for persistent background sessions
