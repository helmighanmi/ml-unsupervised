<!--
Path: SECURITY.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Security policy

Do not commit credentials, private datasets, model-provider tokens or sensitive customer data.

The repository has no required secret for its default local workflows. If future integrations require credentials, use environment variables or an external secret manager.

Untrusted serialized Python/Joblib artifacts must not be loaded. Joblib/pickle deserialization can execute code; only load artifacts produced by trusted pipelines and controlled storage.

Report security issues privately to the repository maintainer rather than publishing exploit details in a public issue.
