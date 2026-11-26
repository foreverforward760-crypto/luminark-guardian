# LUMINARK Guardian — Rule-Based Text Review Prototype

**Status: working software prototype; not a validated AI safety or factual-verification system.**

LUMINARK Guardian is a Python application that scans text for selected language patterns associated with overconfidence, hostile language, coercion, deception, and other categories defined in its rule set. It can produce scores, flags, recommendations, and suggested rewrites.

## What the code currently does

- Uses Python regular expressions and configured keyword/phrase lists.
- Calculates heuristic sentence-alignment and threat scores.
- Maps the resulting scores to the project's 0–9 stage labels.
- Selects alert/defense labels from configured thresholds.
- Applies dictionary-based text substitutions to produce suggested rewrites.
- Includes command-line, FastAPI, and Streamlit application components.
- Generates text and structured reports.

## What it does not establish

Guardian does not independently determine whether a statement is true, verify citations against source documents, or reliably detect hallucinations through semantic model evaluation. A keyword match is not proof of deception, harm, or factual error. A lack of keyword matches is not proof that content is safe or accurate.

The rewrite engine uses substitutions. Those substitutions can change meaning or remove necessary nuance, so every rewrite requires human review. Scores and stage labels are heuristic outputs, not calibrated probabilities or clinical, legal, or regulatory determinations.

No independent benchmark, customer deployment, or measured detection accuracy is established by the repository materials reviewed so far. Do not describe this prototype as production-ready or as a substitute for human review or a specialized safety evaluation.

## Run locally

Install dependencies from `requirements.txt`, then use the CLI, API, or dashboard entry points documented in the source tree. Review configuration and authentication before exposing any service to a network.

## Licensing and provenance

The project metadata refers to MIT, but the repository currently has no root `LICENSE` file. The license position therefore needs to be resolved before commercial distribution or third-party reuse. Do not rely on the README's former MIT badge or wording as proof of a complete license grant.

The repository owner has disclosed that AI tools were used during development. Human-authored contributions, AI-generated portions, third-party code, dependencies, and model terms still require a provenance and rights review.

## Next validation steps

1. Build a labeled test set with human-reviewed examples.
2. Measure false positives and false negatives by category.
3. Test rewrite meaning preservation.
4. Add source-grounded fact and citation verification only if a separate verified method is implemented.
5. Resolve the repository license and dependency obligations.
