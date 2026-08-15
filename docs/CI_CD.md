# CI/CD Guide

## Pipelines

This repository now includes two GitHub Actions workflows:

- `CI`: linting, tests, coverage, dependency audit, Bandit scan, Docker build, Trivy scans, and Sonar analysis
- `CD`: Docker image build and publish to GitHub Container Registry on `main` and version tags

## Required GitHub Secrets

Set these repository secrets before enabling Sonar:

- `SONAR_TOKEN`
- `SONAR_HOST_URL`

If you are using SonarCloud instead of a self-hosted SonarQube server, set `SONAR_HOST_URL` to the SonarCloud URL used by your organization.

## CI Checks

- `ruff check`
- `pytest` with coverage XML output
- `bandit` on `src/`
- `pip-audit` for dependency vulnerabilities
- `trivy` filesystem scan
- `trivy` container image scan

## CD Flow

On pushes to `main` or tags like `v0.1.0`, the workflow:

1. Builds the container image
2. Pushes the image to `ghcr.io/<owner>/<repo>`
3. Validates the Kubernetes manifests in `k8s/`

## Notes

- The CD workflow currently publishes images but does not auto-deploy to a cluster because no target environment or credentials are defined in the repository.
- If you want real cluster deployment next, provide the target platform: Kubernetes, EKS, GKE, AKS, Render, Railway, or another runtime.
