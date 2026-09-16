# Hiver SDE Intern — Spotify AI Support Agent

A retrieval-grounded AI support agent built for the Hiver SDE Intern Take-Home Assignment.

The system takes an incoming Spotify customer-support message and:

1. Classifies the customer intent.
2. Retrieves similar historical Spotify support cases.
3. Drafts a reply grounded in historical Spotify responses.
4. Decides whether to auto-handle the request or escalate it to a human.

---

## 1. Project Overview

Customer-support conversations contain repeated problems and recurring support patterns.

This project uses historical Spotify customer-support conversations from the Customer Support on Twitter dataset to build a focused single-brand support agent.

The system follows this pipeline:

```text
Customer Message
       ↓
Preprocessing
       ↓
Intent Classification
       ↓
Historical Similar-Case Retrieval
       ↓
Grounded Reply Draft
       ↓
Escalation Decision
       ↓
AUTO-HANDLE / ESCALATE