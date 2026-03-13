# Git: Push To GitHub Using HTTPS

GitHub no longer accepts password auth for `git push` over HTTPS. Use a **Personal Access Token (PAT)**.

## One-Time Setup

1. Create a PAT on GitHub:
   - GitHub -> Settings -> Developer settings -> Personal access tokens
   - Use a token with `repo` scope (classic) or appropriate fine-grained repo permissions.
1. Keep the token safe. Treat it like a password.

## Push Commands

From the repo root:

```bash
cd /home/henry/roco509z
git remote set-url origin https://github.com/yangejnr/roco510z.git || \
  git remote add origin https://github.com/yangejnr/roco510z.git

git branch -M main
git push -u origin main
```

When prompted:

- Username: `yangejnr`
- Password: paste the PAT (not your GitHub password)

## If You See "could not read Username"

That usually means Git couldn’t prompt for credentials (non-interactive session) or a credential helper issue.

Two common workarounds:

1. Use a terminal session where Git can prompt, then retry `git push`.
1. Configure a credential helper (system-dependent), then retry `git push`.

