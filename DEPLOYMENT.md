# Deployment

Speak2Sign runs as four hosted parts and a browser extension. Each part deploys separately, and a release that changes more than one part deploys each of them.

| Part | Host | Identifier | Source | Deploys on |
|---|---|---|---|---|
| Web app | Vercel, team `nguyen-vu-nguyens-projects` | project `s2s-app`, production alias https://s2s-app-alpha.vercel.app | `Speak2Sign_Frontend` | `vercel deploy --prod` from the CLI; the project has no Git integration |
| API | Render, free plan, Singapore | web service `s2s-backend`, `srv-dah2qopt0dsc73ebk8r0`, https://s2s-backend-b718.onrender.com | `Speak2Sign_Backend`, branch `main` of `github.com/Siu-er/Speak2Sign_Backend` | every push to `main` (auto-deploy is on) |
| Sign recognition | Modal | app `wlasl-i3d`, class `WLASL` (A10G GPU), volume `wlasl-weights` | `deploy/wlasl.py` in this repository | `modal deploy deploy/wlasl.py` |
| Signing avatar | Cloudflare R2 public bucket | `https://pub-233bfe19474b4be58fb4a245b6800823.r2.dev/jas/loc2025/` | JASigning CWASA build | uploaded by hand; the app only reads it |
| Google Meet extension | Chrome or Edge, loaded unpacked on each machine | `S2S for Google Meet` | `Speak2Sign_MeetExtension` | reload on the browser's extensions page; the side panel loads the web app's `/meet` page, so a web app deploy reaches it without a reload |

The sign dictionary (`data/signs/*.sigml`) is part of the API and goes live with an API deploy.

The local Git remote `github.com/notabota/Speak2Sign_Backend` redirects to `github.com/Siu-er/Speak2Sign_Backend`, the repository Render builds from. A push to either name reaches Render.

## Credentials

Credential values are kept out of Git.

| Credential | Held in | Used by |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_AZURE_ENDPOINT`, `OPENAI_API_VERSION`, `AZURE_OPENAI_GENERATOR_DEPLOYMENT` | backend `.env`; Render service environment | sentence generation and translation |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | backend `.env`; Render service environment; `~/.modal.toml` for the Modal CLI | API calls to the recognition service; `modal deploy` |
| `RENDER_API_KEY`, `RENDER_SERVICE_ID` | backend `.env` | triggering and checking API deploys |
| `NEXT_PUBLIC_API_URL` | Vercel project environment (Production and Preview); frontend `.env.local` for local runs | the web app's API address, fixed at build time |
| Vercel login | Vercel CLI login in the user profile | `vercel deploy` |
| GitHub login | `gh` CLI and Git credential store | pushes |

The Render service environment also sets `PYTHON_VERSION` and `WHISPER_MODEL`. `render.yaml` declares the same variables, and its `sync: false` entries are filled in the Render dashboard, not from the file.

Commands below that need a credential read it from `.env` inside the command:

```bash
set -a; . ./.env; set +a
```

## Deploying the API

1. Commit the change and push `main`:

   ```bash
   git push origin main
   ```

2. Render starts a deploy from the pushed commit. The build installs `requirements.txt`, downloads `en_core_web_sm` and starts gunicorn; a deploy of the full sign dictionary took 2 min 41 s. Follow its status:

   ```bash
   curl -s -H "Authorization: Bearer $RENDER_API_KEY" \
     "https://api.render.com/v1/services/$RENDER_SERVICE_ID/deploys?limit=1"
   ```

   The deploy is live when `status` is `live` and `commit.id` is the pushed commit.

3. A deploy of the current `main` without a new commit (for example, after changing an environment variable in the dashboard):

   ```bash
   curl -s -X POST -H "Authorization: Bearer $RENDER_API_KEY" \
     "https://api.render.com/v1/services/$RENDER_SERVICE_ID/deploys"
   ```

4. Verify the running service:

   ```bash
   curl -s https://s2s-backend-b718.onrender.com/health
   curl -s -X POST -H "Content-Type: application/json" -d '{"gloss":"HELLO"}' \
     https://s2s-backend-b718.onrender.com/gloss-to-sigml
   ```

   `/health` reports `"status": "healthy"` with `asl_glosser` and `sigml_generator` both `true`. The `gloss-to-sigml` response carries the HamNoSys tags of `data/signs/hello.sigml` from the deployed commit.

## Deploying the web app

The Vercel CLI uploads the working directory, and `.vercelignore` excludes only `public/jas`, `public/BSLExamples`, `public/test-videos` and `.next/cache`. Untracked local folders would be uploaded with the deploy. The deploy runs from an export of the committed tree instead:

```bash
cd Speak2Sign_Frontend
git push origin main
OUT="$(mktemp -d)"
git archive HEAD | tar -x -C "$OUT"
cp -r .vercel "$OUT"/
cd "$OUT" && npx vercel deploy --prod --yes
```

Vercel builds the export and points https://s2s-app-alpha.vercel.app at the new deployment. Confirm the alias and status:

```bash
npx vercel inspect <deployment-url>
```

`NEXT_PUBLIC_API_URL` is compiled into the build. A change to `NEXT_PUBLIC_API_URL` takes effect only after a new production deploy.

## Deploying the recognition service

```bash
cd Speak2Sign_Backend
.venv/Scripts/modal.exe deploy deploy/wlasl.py
```

The I3D weights live on the `wlasl-weights` volume. A new Modal workspace fills the volume once with `modal run deploy/wlasl.py::prepare`.

Every recording gets a reply. Signs the classifier ranks above the 0.40 gate are the recognition; when none clears it, the best-ranked sign of each segment is sent to the sentence stage as a guess, and a clip in which the segmenter finds no sign is classified as a whole. `/video-to-sentence` reports which case applied in `confident`. The API requires `confident` in the recognition result, and a release that changes both deploys the Modal app first.

## Request limits

Spending on recognition and translation has no ceiling; the request limits below are what bound it.

The API limits requests per client in process memory (Flask-Limiter; the service runs as one gunicorn process). The paid routes, `/video-to-sentence` and `/translate-to-english`, allow `RATE_LIMIT_PAID` (5 per minute); every other route allows `RATE_LIMIT_DEFAULT` (60 per minute). A refused request receives HTTP 429 with an `error` message, and the API logs the client it refused. The two limits are settings in the Render service environment and take effect on the next restart. The Socket.IO relay is not limited.

Visitors at a venue share one network address, and the API therefore counts each browser separately. The web app creates a random device ID once per browser, keeps it in local storage and sends it as `X-S2S-Device-Id`. A request without a well-formed ID is counted by network address. A caller can send a new ID with every request; the device count stops a visitor pressing the button repeatedly, not a script written to evade it.

The team's own devices are not limited. `RATE_LIMIT_EXEMPT_KEYS` in the Render service environment holds a comma-separated list of keys, and a device enters one under Settings, Team device; the app then sends it as `X-S2S-Device-Key`. A new key is generated with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Render's proxies keep any `X-Forwarded-For` entries the client sent and append their own. On this service the caller's address is the third entry from the right; the two after it are proxy addresses that change from request to request. The API reads the client address `TRUSTED_PROXY_HOPS` entries from the right, and the Render service environment sets it to 3. A change to Render's proxy chain changes that position, and the setting is checked after a deploy by sending six requests with a forged leading entry and reading the refusal in the service log:

```bash
for i in 1 2 3 4 5 6; do
  curl -s -o /dev/null -w "%{http_code} " -X POST -H "Content-Type: application/json" \
    -H "X-Forwarded-For: 192.0.2.$i" -d '{"text":""}' \
    https://s2s-backend-b718.onrender.com/translate-to-english
done; curl -s https://api.ipify.org
```

The sixth request returns 429, and the log line `rate limit ... reached by address:<address>` names the public address `api.ipify.org` printed. An internal `10.x` address in that line, or six responses without a 429, means the hop count no longer matches the proxy chain.

## Starting without a cold delay

Two parts scale to zero when idle, and the first request after idling waits for them to start. Measured start times:

| Part | Idle behaviour | First request when cold | When warm |
|---|---|---|---|
| API on Render free plan | instance spins down after 15 minutes without inbound requests | 72.5 s for `/health` after 17 minutes idle | 0.3 s for `/health` |
| Recognition on Modal | GPU container stops 120 s after its last call (`scaledown_window=120`) | 17.1 s for `ping` | 3.6 s for `ping` |
| Web app on Vercel | no sleeping instance | 0.1 s for `/` | 0.1 s for `/` |

`scripts/warm_up.py` wakes all three and reports the time each took, failing with the service's name when one does not answer. For a session such as a demonstration, start the script before the session with a hold time that covers the session:

```bash
cd Speak2Sign_Backend
.venv/Scripts/python.exe scripts/warm_up.py --hold 45
```

With `--hold`, the script repeats the round every 60 s, half the Modal idle limit, until the hold time ends. A single run without `--hold` keeps the GPU warm for only 120 s.

Keeping the parts warm at all times is a billing decision. Only free Render instances spin down, and moving `s2s-backend` to a paid instance type (Starter or above) in the Render dashboard removes the API cold start; Render sets the instance type per service. On Modal, `min_containers=1` on the `WLASL` class keeps one A10G container running continuously, billed for every second it runs whether or not anyone signs. The class scales to zero.

## Installing the Google Meet extension

The extension holds no S2S logic of its own: its side panel embeds `https://s2s-app-alpha.vercel.app/meet?code=<meeting code>`, and the rooms are relayed by the API. Each machine in a demonstration installs it once:

1. Open `chrome://extensions` or `edge://extensions` and turn on Developer mode.
2. Choose **Load unpacked** and select the `Speak2Sign_MeetExtension` folder.
3. In a Meet call, select the S2S toolbar icon, choose **Allow access** once, then choose a role.

`APP_ORIGIN` in the extension's `config.js` selects the web app the panel embeds. `Speak2Sign_MeetExtension/README.md` describes the extension in full.

## Running locally

```bash
# API on http://localhost:5000
cd Speak2Sign_Backend
.venv/Scripts/python.exe run.py

# Web app on http://localhost:3000, with NEXT_PUBLIC_API_URL=http://localhost:5000 in .env.local
cd Speak2Sign_Frontend
npm run dev
```

The local API reads the same `.env` and calls the deployed Modal app and Azure OpenAI.
