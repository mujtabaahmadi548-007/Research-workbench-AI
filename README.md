# AI Research Learning Workbench

Learn frontier AI from live primary sources instead of stale summaries. The app pulls today's AI news
and model releases (from the labs themselves, tech press, expert newsletters, Hugging Face and Hacker
News), today's papers from Hugging Face Daily Papers and arXiv, and GitHub releases. It turns each
paper into a first-principles breakdown with an interactive calculator and engineering drills that you
review with spaced repetition.

It runs entirely on your computer: a small Python server (`server.py`) and one web page (`index.html`).
It also runs as a **website** (GitHub Pages) that you can open on your phone and computer: see the next section.

---

## Use it as a website (phone and computer)

GitHub builds and publishes the website for you about every 5 minutes (`.github/workflows/update-site.yml`: each
run starts the next one, because GitHub's own schedules run hours late), so it stays live without your computer.
An open page checks for a new build every minute and shows it without a reload, and it checks Hacker News and
Bluesky itself every minute: the **⚡ Live** box at the top of News lists everything from the last 3 hours.

It is built to keep running without attention: the website updates and the alerts loop restart each other if one
stops, both re-enable themselves so GitHub does not switch them off after 60 quiet days (after 30 days without a
commit a small heartbeat file, `.github/heartbeat.txt`, is committed automatically), and if no update succeeds for
3 hours you get one GitHub issue titled "Website updates have stopped". Turn it on once:

1. On GitHub, open the repository, then **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
   (If GitHub says Pages needs an upgrade for private repositories, make the repository public under
   **Settings → General → Danger Zone → Change visibility**, or use the free GitHub Pro from GitHub Education.)
3. Open the **Actions** tab → **Update website** → **Run workflow**. After about 5 minutes the site is at
   `https://<your-github-name>.github.io/Research-workbench-AI/` (also shown on the Settings → Pages screen).

On the website:

- **News, Papers, Repos, Systems Lab, Evolution graph and Review** all work. News and papers are refreshed about every 5 minutes (Hacker News and Bluesky every minute); the sidebar shows "Live · updated … ago".
- **Gemini features** (paper breakdowns, Triage, + Review cards, Frontier Brief): paste your Gemini key in the site's
  **Settings**. The key is saved only in that browser and sent only to Google; it is never part of the site or the repository.
- **Your progress** (review schedule, read marks, breakdowns) is saved in each browser, so your phone and your computer each keep their own.
- **Optional:** add your Gemini key as a repository secret named `GEMINI_API_KEY` (**Settings → Secrets and variables → Actions**).
  Each update then also writes the Frontier Brief, triages the top stories and breaks down 2 top papers on GitHub's side.
- **Update right now:** **Actions → Update website → Run workflow**.

### Highest alerts (notifications)

Every 5 minutes GitHub checks whether a **new Claude, Gemini or GPT model was officially released**, or a **major AI
breakthrough** happened, and notifies you (`alerts.py`, `.github/workflows/alerts.yml`).

- **Only real releases:** a new model ID in the company's own model documentation (Anthropic, OpenAI, Google), the
  lab's own announcement ("Introducing …", "… is now available"), or a Hacker News post linking to it. Rumors, leaks,
  plans and "coming soon" never alert.
- **Breakthroughs:** an AI story with 500+ Hacker News points and breakthrough wording (solves, proof, discovers,
  gold medal…), or an official lab post with that wording that Hacker News also picked up. With the `GEMINI_API_KEY`
  secret, Gemini double-checks that it is really major.
- **How you are notified:** a GitHub issue that @mentions you. Turn on notifications in the **GitHub mobile app** to
  get a push on your phone (you also get an email). The alert also appears at the top of the website's News page,
  and as its own 🚨 story in the News list (the official announcement and press coverage merge into it later). The
  desktop app reads the alerts from the website's `alerts.json`, so it shows them too.
- **Louder push (optional):** install the free **ntfy** app (Android/iOS, or ntfy.sh in a browser), subscribe to a
  long random topic name only you know, and save that same name as the repository secret `NTFY_TOPIC`
  (**Settings → Secrets and variables → Actions**). Alerts then arrive as urgent push notifications.
- **Test it:** **Actions → Highest alerts → Run workflow**, tick **Send a test notification**.
- GitHub sometimes starts scheduled runs a few minutes late, so an alert usually arrives within 5–15 minutes of the
  announcement. GitHub pauses scheduled workflows in repositories with no commits for 60 days (it emails you first);
  click **Enable workflow** on the Actions tab if that happens.

### X & Bluesky page

The **X & Bluesky** tab shows posts from AI labs, researchers and AI reporters (replies and reposts left out).

- **Bluesky** works with no key and costs nothing: 21 accounts that are active there (Simon Willison, Ethan Mollick,
  Nathan Lambert, Emily Bender, Arvind Narayanan, Ai2, Epoch AI, PyTorch, Casey Newton and others). Many AI people
  stopped posting on Bluesky, so they are only on X.
- **X posts are free (no key, no card):**
  1. **The accounts' own timelines:** X's public embed timelines (what an embedded X timeline on any website shows)
     for the listed AI labs and people (Anthropic, OpenAI, Google DeepMind, Gemini, Meta, Mistral, xAI, DeepSeek,
     Qwen, Hugging Face, Karpathy, Sam Altman, Demis Hassabis and others), with likes and pictures. X allows only
     a few reads per window and often refuses GitHub's shared servers, so these come mostly from the desktop app
     on your own connection; the website keeps trying 3 accounts per update and pauses when X asks it to. Some
     people's embed timelines are months old (e.g. Karpathy, Sam Altman); their posts arrive through 2. instead.
  2. **Posts the AI newsletters link to:** Latent Space's daily AI Twitter recap, Zvi, Ben's Bites, Gary Marcus,
     Understanding AI and others, plus Hacker News and Bluesky. Each post is checked on X (X's public oEmbed service)
     and shown with its real text, so nothing is ever made up. Filter: **Linked in newsletters**.
  Only posts from the last 7 days are kept.
- **Optional paid extras (not needed):** Gemini + Google Search can look for the accounts' posts too, but Google
  Search in Gemini needs a key with billing, so it is off (turn on with `WORKBENCH_GX_ENABLE=1`). An X API key reads
  every post of every account, pay per use:

| Service | Price | Get the key |
|---|---|---|
| Official X API | $0.005 per post read | [developer.x.com](https://developer.x.com): create an app, buy credits, set a spending limit, copy the **Bearer Token** |
| twitterapi.io (unofficial) | about $0.15 per 1,000 posts | [twitterapi.io](https://twitterapi.io): sign up (new accounts get $1 free), copy the API key |

- **Website:** add the repository secret `X_API_KEY` (**Settings → Secrets and variables → Actions**). For
  twitterapi.io also add the repository *variable* `X_API_PROVIDER` = `twitterapi`. Optional variables:
  `X_DAILY_CAP` (posts read per day, default 100: at most about $15/month with the official API, $0.45 with
  twitterapi.io) and `X_ACCOUNTS` (`AnthropicAI:lab, karpathy:people, ...`).
- **Desktop app:** without a key of its own it shows the website's posts (from `x.json`) for free. To read X
  directly, add a key in **Settings → X (Twitter)**.
- Only new posts are read each time (since the last one seen), so the real cost is usually well below the cap.

---

## What you need

- **Windows 10 or 11.**
- **Python 3.10 or newer.** Get it from <https://www.python.org/downloads/>. When the installer opens,
  tick **"Add python.exe to PATH"** before clicking Install.
- **A free Gemini API key** from Google AI Studio. This is optional: everything except the AI
  breakdowns of new papers works without it, and 10 papers come with breakdowns already written.

---

## Set up (do this once)

### Step 0. Get the code

**Easiest:** on <https://github.com/mujtabaahmadi548-007/Research-workbench-AI>, click the green
**Code** button, then **Download ZIP**, and unzip it anywhere (for example `Documents\Research-workbench-AI`).

**Or with git** (lets you pull future updates with `git pull`). In PowerShell, in the folder where you want it:

```powershell
git clone https://github.com/mujtabaahmadi548-007/Research-workbench-AI.git
```

Your database (`workbench.db`, which holds your papers, review history and Gemini key) is never uploaded
to GitHub. A fresh copy starts with the 10 starter papers; paste your Gemini key in Settings again.

### Step 1. Open PowerShell in this folder

1. Open **File Explorer** and go to the folder that contains `server.py`.
2. Click once on the **address bar** at the top (where the folder path is shown).
3. Type `powershell` and press **Enter**.

A PowerShell window opens, already inside this folder. Type every command below into that window
and press **Enter** after each one.

### Step 2. Create a virtual environment

```powershell
python -m venv .venv
```

This makes a private `.venv` folder for the app's Python packages. It takes a few seconds and prints nothing when it works.

### Step 3. Install the requirements

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Wait until it says `Successfully installed ...`.

> You do not need to "activate" the virtual environment. Calling `.\.venv\Scripts\python.exe`
> directly avoids PowerShell's "running scripts is disabled on this system" error.

---

## Run the app (do this every time)

### Step 4. Start the server

In the PowerShell window (opened in this folder, as in Step 1):

```powershell
.\.venv\Scripts\python.exe server.py
```

Leave this window open while you use the app. It prints what it is doing in plain language, for
example `Hugging Face: got 50 papers (50 new).`

### Step 5. Open the app in your browser

Go to **<http://localhost:8000>**.

### Step 6. Paste your Gemini key in Settings

1. Go to <https://aistudio.google.com/apikey>, sign in with a Google account and click
   **Create API key**. Copy the key.
2. In the app, click **Settings** in the left sidebar.
3. Paste the key into **API key** and click **Save key**.
4. Click **Check key & list models** to confirm it works.

The key is stored only in the file `workbench.db` on your computer. It is not in the code, and the
included `.gitignore` keeps that file out of git.

**To stop the app:** click the PowerShell window and press **Ctrl + C**.

---

## Using it

| Tab | What it does |
| --- | --- |
| **News** | New AI models, agents and AI technology. Sources: official blogs (OpenAI, Google DeepMind, Google AI, Google Research, Microsoft Research, NVIDIA, Hugging Face, AWS), tech press (TechCrunch, The Verge, MIT Technology Review, Ars Technica, MarkTechPost), expert newsletters (Simon Willison, Import AI, Latent Space, Interconnects), trending and new models on Hugging Face, and AI stories on Hacker News. Reports about the same launch are merged into one event card, led by the lab's own announcement. Filter by Model releases / Agents / Tools & infra / Research / Industry & policy. Add your own RSS feeds in Settings. |
| **Papers** | Live papers (Hugging Face Daily Papers and the newest arXiv cs.AI / cs.CL / cs.LG), deduplicated by arXiv ID, with full-text search. **Refresh** pulls new data; each source shows when it was last fetched and any error. The **Repos & releases** tab tracks inference engines, agent frameworks and open-weight model repos on GitHub. |
| **Paper page** | Click any paper, then **Deconstruct with Gemini**. You get: (a) the bottleneck it attacks, (b) the mechanism step by step and what changed vs prior baselines, (c) an interactive parameter calculator, (d) lineage, (e) 6–18-month implications and (f) 3 engineering questions. Each paper is analyzed once and cached. Breakdowns are generated from the abstract only and are labeled as unverified. |
| **Systems Lab** | Four calculators that work without a key: KV-cache memory (MHA / GQA / MQA / quantized KV), test-time compute curves, a multi-model router simulator, and an agent state-machine trace inspector. |
| **Evolution graph** | A zoomable map of foundation concepts (attention, scaling laws, MoE, KV-cache, RLHF, RLVR/GRPO, RAG, tool use/MCP, agents and more). Analyzed papers attach to the concepts they build on. |
| **Review** | Spaced repetition (SM-2). Due questions come first, then new ones. Grade **Again / Hard / Good / Easy** (keys 1–4) after seeing the explanation. |
| **Settings** | Gemini key and model, startup fetch, papers per refresh, new questions per day, and the analysis queue. |

### Working with the News page

- **Today's Frontier Brief** (top of the page): Gemini reads your *unread* items and writes a 15-second
  summary of the top 3 model releases, top 3 agent/tooling items and the main research trend. It is
  written once per refresh and cached. Without a key you get automatic picks instead.
- **Fresh drops only (<14 days)** is on by default for Trending and Latest, so months-old models with huge
  lifetime download counts don't bury this week's releases. Most important shows everything.
- **Unread / New since last visit / Mark current view as seen** keep track of what you have read
  (clicking a story also marks it seen). **Cards / Compact** switches the layout.
- **Model cards** show real specs from Hugging Face: parameters, license, quantization (GGUF, AWQ, FP8…),
  base model, gated status, and the README's own description. A model created months ago whose files
  changed recently is labeled **Updated weights/repo (created …)**, not a new release.
- **GitHub** releases of tracked projects and new agent repos appear as News events too.
- **Card actions:**
  - **✨ Triage**: Gemini writes *What it is & architecture / Why it matters / Hardware & runability*
    (done automatically for the top events, cached).
  - **+ Review (SM-2)**: Gemini turns the item into a concept card in Review (it comes up first among new cards).
  - **+ Systems Lab**: a ready-to-run quickstart (Ollama for GGUF repos, vLLM, transformers or diffusers,
    README commands, an MCP config for MCP servers, and a VRAM table you can send to the KV-cache calculator).
  - **Paper: arXiv:… → Add to Papers & Graph**: downloads the paper, adds it to Papers and the Evolution
    graph (as a pending node until Gemini's breakdown is ready).
- **Trending** = `(0.6 × importance + 0.4 × percentile within its type + 12 × (sources − 1)) × 0.5^(age / half-life)`.
  Half-life is 48 h for articles and Hacker News, 4 days for GitHub releases, 7 days for models. Then:
  - Each further post from the same feed (AWS, NVIDIA, the Hugging Face Hub…) within 24 h loses 8 points,
    unless another source covers the same story.
  - Stories covered by 3+ independent sources within 72 h always rank above everything else.
  - Community fine-tunes and quantizations count at 70%; old models with re-uploaded weights at 60%.
  - At least 3 of the top 30 slots go to fresh models when there are any.
- **The Frontier Brief never makes you wait:** the last one shows instantly ("generated X min ago ·
  Regenerate") and Gemini rewrites it in the background after each refresh.
- **GitHub CI builds** (like llama.cpp's `b12383`), pre-releases and patch bumps without release notes are
  left out of News. Real releases such as `v1.0.0` merge with blog coverage of the same release.
- **+ Systems Lab** appears only on things you can run (models, repos, SDKs, technical tutorials).
  **+ Papers** appears when a story links an arXiv paper, or, for research news without one, searches arXiv for it.

### Images, importance and latestness

- **Images.** News cards use the image from the feed or, when there is none, the article's own preview
  image. Papers show a real figure from the paper (read from arXiv's HTML version, preferring the
  architecture or method diagram), else media the authors uploaded to Hugging Face, else Hugging Face's
  preview card. Repos show GitHub's preview card and the owner's avatar. Anything without an image gets
  a generated cover. arXiv asks for 15 seconds between page requests, so paper figures fill in gradually
  in the background (switch this off in Settings); opening a paper loads its figures right away.
- **Importance (0–100).** Every item shows a score and a tier (Major / High / Notable / Low). Hover it,
  or open a paper, to see exactly what it is made of. Papers: Hugging Face upvotes, code and GitHub
  stars, models built on it, discussion, being featured on HF Daily, and frontier topics. News: how
  authoritative the source is, how many sources cover the story, engagement (Hacker News points and
  comments, Hugging Face likes and downloads) and what it is (a model release counts most). Papers too
  new to have any signals say **No signals yet** instead of pretending they are unimportant.
- **Latestness.** Badges show **New today**, **Fresh** (under 3 days), **This week** and **This month**,
  with relative times everywhere. Sort by **Trending** (importance that fades with age), **Latest** or
  **Most important**.

### Starter content

Ten real papers ship with pre-written breakdowns and drills (DeepSeek-R1, DeepSeek-V3, test-time
compute scaling, Mamba-2, Native Sparse Attention, KIVI, RouteLLM, SWE-agent, Qwen2.5-VL and
GraphRAG). Their arXiv IDs were checked against arXiv. Abstracts and author lists are downloaded
from arXiv the first time the app is online.

### Offline, rate limits and errors

- **Offline:** the app works with whatever is cached. Each source card on the Feed shows its own error.
- **Gemini rate limits, no key, or no internet:** the paper is queued, the app shows why, and it
  retries automatically with backoff. Nothing crashes. See **Settings → Analysis queue**.
- **arXiv** is called at most once every 3 seconds (its API) and once every 15 seconds (its web pages, for
  figures). **GitHub** without a token allows 60 requests per hour, so releases are re-checked at most
  every 30 minutes.
- **News feeds** (67: lab and company blogs incl. Anthropic, Mistral, Google Developers, Microsoft, Apple ML, METR;
  AI press incl. Techmeme, NYT, The Guardian, Wired, IEEE Spectrum; expert newsletters and research blogs incl.
  Epoch AI, Import AI, TLDR AI, Zvi, Gary Marcus, Lilian Weng, Karpathy; Lobsters) that fail show their own error on the News page under "Feeds (show all)". Anthropic has no RSS
  feed, so its news page is read directly. Meta AI and xAI have no usable feed; they are covered through press,
  newsletters, Hacker News and the X page.
- **Time window:** News shows only items published (or, for models and repos, created) within the window you pick
  above the list: 3 h, 24 h, 3 days, 7 days or 14 days (the longest). Nothing older is shown, in any sort; posts on
  the X & Bluesky page are at most 7 days old; the Live box covers the last 3 hours.
- **Nothing gets lost:** the website keeps its database between updates (GitHub Actions cache). If that cache is
  ever lost, the next update rebuilds the news history from the published website instead of starting empty.
- **Duplicates:** reports of one event are merged into one card (same model, same link, or the same headline in
  other words). A post that a site re-published under a new link, or whose title was corrected, is shown once.

---

## Troubleshooting

**`python` is not recognized.** Python is not installed or was not added to PATH. Reinstall it and
tick "Add python.exe to PATH", then close and reopen PowerShell. You can also try `py` instead of
`python` in Step 2.

**Port 8000 is already in use.** Start on another port, then open <http://localhost:8001>:

```powershell
$env:PORT = "8001"; .\.venv\Scripts\python.exe server.py
```

**Start fresh.** Stop the server, delete `workbench.db`, and start it again. The starter papers are
re-created automatically. (This also deletes your saved key and review history.)

**"HTTP 503 … experiencing high demand".** Google's servers are overloaded for that model. It is temporary
and not caused by the app. The app retries twice automatically and then tries another Flash model your key
can use (Settings → "When the model is overloaded…"). If it still fails, the News page shows automatic
picks with a **Try Gemini again** button, and it waits 5 minutes before asking Gemini on its own again.
If it keeps happening, pick a different model in Settings (for example one with "flash-lite" in its name).

**The Gemini model is not found.** In Settings, click **Check key & list models** and pick a model
from the list (any model with "flash" in its name is a good choice), then click **Save model**.

---

## Files

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI server: paper, news, model and GitHub ingestion, figures and preview images, importance scoring, SQLite + FTS5 storage, Gemini calls with JSON validation and a retry queue, SM-2 scheduling, starter content. |
| `index.html` | The whole user interface: vanilla JavaScript and SVG, no build step. The same page runs the website (it switches to website mode when `data.js` is present). |
| `build_site.py` | Builds the website into `site/`: fetches sources, then writes `data.js`, `version.json` and `index.html`. GitHub Actions runs it about every 5 minutes. |
| `.github/workflows/update-site.yml` | The GitHub Actions workflow that builds and publishes the website to GitHub Pages. |
| `alerts.py` | Highest alerts: detects official Claude / Gemini / GPT releases and major breakthroughs, and notifies you. |
| `.github/workflows/alerts.yml` | Runs `alerts.py` every 5 minutes on GitHub. |
| `requirements.txt` | Python packages: FastAPI, Uvicorn, httpx, Pydantic. |
| `workbench.db` | Created on first run. Your papers, breakdowns, review history and settings (including the key). |
