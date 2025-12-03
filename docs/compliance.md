# Compliance & Data Protection

## GDPR/Data Protection
- No PII stored by default; inputs are ephemeral in session state.
- If persistence is enabled, ensure data retention policies and consent mechanisms.

## Audit Trails
- Use CI logs and application logs to maintain traceability.
- For full audit trails, integrate a centralized log backend (e.g., ELK) and store request IDs.

## Retention Policies
- Define retention for any stored predictions or uploaded datasets.
- Purge schedules and backup lifecycles to be documented per environment.

## Industry Regulations
- Medical use disclaimer is present in UI; for clinical use, integrate certified processes (HIPAA, ISO).
