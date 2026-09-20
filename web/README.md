# Web client

Run the development server with `npm run dev` (from `web/`). The API URL is configured with `VITE_API_URL`; it defaults to `http://127.0.0.1:8000` when unset or blank. For a desktop API use that default, and for a phone on the LAN set the machine's LAN URL and start the API with `--host 0.0.0.0`. Only absolute `http://` and `https://` URLs are accepted; trailing slashes are removed.

Commands:

- `npm test` — Vitest suite
- `npm run lint` — ESLint
- `npm run typecheck` — TypeScript checking
- `npm run build` — typecheck and production build
