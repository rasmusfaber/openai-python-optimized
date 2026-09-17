<!-- Modified by Rasmus Faber: distinguish fork and upstream disclosure responsibilities. -->
# Security Policy

## Reporting a vulnerability

Rasmus Faber maintains the `openai-python-optimized` fork. For vulnerabilities
introduced by this fork, arrange private disclosure with
[@rasmusfaber](https://github.com/rasmusfaber). Any public request for a private
contact method must omit vulnerability details. This repository does not currently
offer GitHub private vulnerability reporting.

For vulnerabilities that also affect the upstream SDK or OpenAI services, report
privately through OpenAI's
[coordinated vulnerability disclosure process](https://openai.com/policies/coordinated-vulnerability-disclosure-policy).
For questions about that process, contact disclosure@openai.com.

OpenAI's process covers the upstream SDK and the official
[`openai` Python package](https://pypi.org/project/openai/), including its
published source distributions and wheels. OpenAI does not maintain this fork.

## Threat model authority

This file governs disclosure for the fork. The inherited upstream threat model,
trust boundaries, and reportability guidance are retained as reference in
[`docs/architecture/security-model.md`](docs/architecture/security-model.md).
That document describes upstream infrastructure, including automation that this
fork has retired. Current fork automation is described in
[.github/README.md](.github/README.md).

Do not report security vulnerabilities through public GitHub issues, pull requests, or discussions.

## What to include

- The affected package or product and version, or the relevant source commit.
- A clear description of the security impact.
- Sanitized steps to reproduce the issue.

For either the fork or upstream package, include the Python version, operating system,
and affected source distribution or wheel when relevant.

Do not include live credentials, API keys, customer data, or unredacted sensitive logs.

Redact authentication headers and private keys, and replace other secrets with
clearly fake values.

## Coordinated disclosure

Keep fork vulnerability details confidential until disclosure is coordinated with
Rasmus Faber. Reports to OpenAI follow OpenAI's linked coordinated-disclosure terms.

Thank you for helping us keep this SDK and the systems it interacts with secure.
