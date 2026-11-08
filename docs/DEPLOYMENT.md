# Deployment

Target: **live URL in under 20 minutes, cost ₹0/month.**

```
Vercel (web, free)  →  Render (api, free 512MB)  →  Neon (Postgres, free)  →  R2 (uploads, free)
```

---

## 0. Prerequisites

- GitHub repo with this code pushed
- Accounts: [Vercel](https://vercel.com), [Render](https://render.com),
  [Neon](https://neon.tech), [Cloudflare](https://dash.cloudflare.com) (R2)
- A trained `rec.onnx` + `charset.txt`, **or** accept the stub engine for the demo

---

## 1. Neon — database (~2 min)

1. Create project → region closest to your Render region (Singapore for India).
2. Copy the **pooled** connection string.
3. It looks like:
   ```
   postgresql://user:pass@ep-xxx-pooler.ap-southeast-1.aws.neon.tech/varnam?sslmode=require
   ```

> Use the **pooler** host, not the direct one. Render's free instance opens and
> closes connections on cold start, and the pooler is what prevents
> "too many connections" after a few deploys.

Tables are created automatically on boot (`init_db()` → `create_all`). No
migration step is required for a first deploy.

---

## 2. Cloudflare R2 — upload storage (~3 min)

Render's free tier has no persistent disk, so uploads written to `/app/uploads`
disappear on redeploy.

1. R2 → **Create bucket** → `varnam-ocr`
2. **Manage R2 API Tokens** → create token with *Object Read & Write*
3. Record: Account ID, Access Key ID, Secret Access Key

> Skipping this is survivable: the API still returns results, it just cannot show
> the original image after a restart. Set `STORAGE_BACKEND=local` and accept that.

---

## 3. Render — the API (~5 min)

1. **New → Blueprint** → select the repo. Render reads `render.yaml`.
2. It will create `varnam-ocr-api` with the Docker build.
3. Fill the `sync: false` variables:

| Variable | Value |
| --- | --- |
| `DATABASE_URL` | Neon pooled URL |
| `CORS_ORIGINS` | `https://your-app.vercel.app` *(add after step 4)* |
| `R2_ACCOUNT_ID` | from step 2 |
| `R2_ACCESS_KEY_ID` | from step 2 |
| `R2_SECRET_ACCESS_KEY` | from step 2 |
| `R2_BUCKET` | `varnam-ocr` |
| `API_SHARED_SECRET` | any long random string |

4. **Create Web Service.** First build takes 4-6 minutes (pip install dominates).
5. Confirm: `https://varnam-ocr-api.onrender.com/health/ready` returns
   `{"status":"ok","model_loaded":true,...}`.

> If it returns `503` with `model_loaded:false`, the model file is missing from
> `/app/models`. Either run `ml/export_onnx.py --out services/api/models` and
> commit the artifacts (via Git LFS), or set `VARNAM_ENGINE=fake` to demo the UI.

### Model delivery, three options

| Option | How | Trade-off |
| --- | --- | --- |
| Baked into image | commit `rec.onnx` to the image | Free, but repo grows ~4MB |
| Git LFS | `git lfs track "*.onnx"` | Clean repo, Render must fetch LFS |
| Download at build | `RUN curl -o /app/models/rec.onnx $MODEL_URL` | Needs a public URL + checksum |

The Dockerfile currently expects the file to exist at `services/api/models/rec.onnx`.

---

## 4. Vercel — the web app (~3 min)

1. **Add New → Project** → import the repo.
2. **Root Directory:** `apps/web` ← this is the step people miss.
3. Framework preset: **Next.js** (auto-detected).
4. Environment variable:
   ```
   NEXT_PUBLIC_API_BASE_URL = https://varnam-ocr-api.onrender.com
   ```
5. **Deploy.** Copy the production URL.
6. Go back to Render → update `CORS_ORIGINS` to include that URL exactly (no
   trailing slash) → **Manual Deploy → Restart**.

> `NEXT_PUBLIC_*` is inlined at **build** time. Changing it requires a redeploy,
> not just a restart — a classic 30-minute debugging session.

---

## 5. Verify (~2 min)

```bash
bash scripts/smoke_test.sh https://varnam-ocr-api.onrender.com
```

Expected:

```
==> 1/3 health
    ready after 1 attempt(s)
==> 2/3 openapi
    schema served
==> 3/3 ocr
    ok: engine=fake lines=3 confidence=0.93 41ms
smoke test passed
```

Then open the Vercel URL → `/app` → upload a page. First request may show
"Waking the OCR service"; that is the free instance booting, not a bug.

---

## 6. CI/CD

`.github/workflows/ci.yml` runs on every push: ruff, mypy, pytest (75% gate),
eslint, tsc, vitest, `next build`, and a Docker build.

`.github/workflows/deploy.yml` runs on a `v*` tag:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

Required repository secrets:

| Secret | Where to get it |
| --- | --- |
| `RENDER_DEPLOY_HOOK_URL` | Render → service → Settings → Deploy Hook |
| `API_BASE_URL` | `https://varnam-ocr-api.onrender.com` |
| `VERCEL_TOKEN` | Vercel → Account → Tokens |
| `VERCEL_ORG_ID` / `VERCEL_PROJECT_ID` | `.vercel/project.json` after `vercel link` |

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `403`/CORS error in browser console | `CORS_ORIGINS` missing the exact origin | Add it verbatim, no trailing slash, restart |
| `503` on every request | Model still loading, or missing | Check `/health/ready`; check `MODEL_DIR` |
| Cold start > 60s | Image too large | `docker build` locally and check size; target <200MB |
| OOM kill (container restarts) | Two workers, or model too big | Confirm `--workers 1`; quantize the model |
| `429` immediately | Shared IP (office/campus NAT) | Raise `RATE_LIMIT_PER_MINUTE` for demos |
| Results disappear after redeploy | Ephemeral disk | Switch `STORAGE_BACKEND=r2` |
| `NEXT_PUBLIC_*` change ignored | Inlined at build | Redeploy on Vercel |

---

## 8. Cost beyond free

| Component | Free | First paid step |
| --- | --- | --- |
| Render | 512MB, sleeps | Starter $7/mo — 512MB always-on |
| Vercel | 100GB bandwidth | Pro $20/mo |
| Neon | 0.5GB storage | Launch $19/mo |
| R2 | 10GB, no egress fee | $0.015/GB |

The cheapest meaningful upgrade is **Render Starter**: removing the sleep removes
the ~30s cold start, which is by far the biggest UX problem on the free tier.
