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

No independent benchmark, customer deployment, or measured detection accuracy is established by the repository materials reviewed so far. HRV/polyvagal adjustments and cultural/domain modifiers are project-defined heuristics, not validated physiological, psychological, or clinical measurements. Do not describe this prototype as production-ready or as a substitute for human review or a specialized safety evaluation.

## Run locally

Install dependencies from `requirements.txt`, then use the CLI, API, or dashboard entry points documented in the source tree. Review configuration and authentication before exposing any service to a network.

## Conceptual provenance

The project owner states that the underlying concepts, product idea, and 0-9-0 / SAP framework applications in this repository originated with him. AI systems generated the implementation code and suggested technical mechanisms used to express those concepts. NSDT and the Container Rule were introduced by AI as supporting implementation mechanisms within the owner's framework, not as independent conceptual origins.

This records the project owner's account; it is not a legal determination of copyright ownership, inventorship, or exclusive rights.

## Licensing and provenance

The repository currently has no root `LICENSE` file and no license is declared in the package metadata. The license position therefore remains unresolved. Do not assume that the software is MIT-licensed or that it can be redistributed or commercially transferred without a rights review.

The repository owner has disclosed that AI tools were used during development. Human-authored contributions, AI-generated portions, third-party code, dependencies, and model terms still require a provenance and rights review.

## Next validation steps

1. Build a labeled test set with human-reviewed examples.
2. Measure false positives and false negatives by category.
3. Test rewrite meaning preservation.
4. Add source-grounded fact and citation verification only if a separate verified method is implemented.
5. Resolve the repository license and dependency obligations.
