# Deploying

The site is a static MkDocs build (`./site`) hosted on **Cloudflare Pages**, project
`inference-engineering-deep-dive`, served at **https://inference.kimambo.de/**.

There are two ways to deploy: a one-command manual push, and an automatic GitHub Action on every push
to `main`.

## One-time setup

1. **Create the GitHub repo and push** (you do this once):
   ```bash
   git remote add origin git@github.com:maxkimambo/inference-engineering-deep-dive.git
   git push -u origin main
   ```

2. **First Cloudflare deploy — creates the Pages project** (run locally, opens a browser to log in):
   ```bash
   make deploy-cf
   ```
   This authenticates `wrangler`, creates the `inference-engineering-deep-dive` Pages project, and
   publishes the current build.

3. **Bind the custom domain** (Cloudflare dashboard → Workers & Pages → the project → Custom domains
   → add `inference.kimambo.de`). Because `kimambo.de` is already on Cloudflare, this is one click —
   Cloudflare writes the CNAME and provisions TLS automatically. (To use a different subdomain or the
   apex, change `site_url` in `mkdocs.yml` to match.)

4. **Add two GitHub repo secrets** so the Action can deploy (Settings → Secrets and variables →
   Actions):
   - `CLOUDFLARE_API_TOKEN` — a token with the **Cloudflare Pages: Edit** permission
     (My Profile → API Tokens → Create Token).
   - `CLOUDFLARE_ACCOUNT_ID` — from any Cloudflare dashboard URL or the account overview.

## Day-to-day

- **Automatic:** every push to `main` runs [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml),
  which builds with `mkdocs build --strict` and publishes to Cloudflare Pages. A failing build (broken
  link/anchor) blocks the deploy.
- **Manual:** `make deploy-cf` builds and publishes immediately from your machine.

## Local preview

```bash
make install   # once: create venv, install deps
make serve     # live-reload at http://127.0.0.1:8000
make build     # strict build into ./site (same gate as CI)
```
