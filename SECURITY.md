# Security policy

Do not report API keys or other credentials in a public issue.

If a credential is accidentally committed, revoke it with the API provider
immediately and remove it from the repository history. MiniMem reads
credentials from `.env`, which is ignored by Git; only `.env.example` should
be committed.

For security vulnerabilities, contact the repository maintainers privately
through the security reporting channel configured on the hosting platform.
