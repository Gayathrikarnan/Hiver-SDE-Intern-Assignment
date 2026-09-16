# Hiver SDE Intern — Spotify AI Support Agent

A retrieval-grounded AI support agent built for the Hiver SDE Intern Take-Home Assignment.

The system takes an incoming Spotify customer-support message and:

1. Classifies the customer intent.
2. Retrieves similar historical Spotify support cases.
3. Drafts a reply grounded in historical Spotify responses.
4. Decides whether to auto-handle the request or escalate it to a human.

---

## 1. Overview

This project implements an AI-powered customer support agent for Spotify using historical customer-support conversations from the **Customer Support on Twitter** dataset.

The system takes a new customer message and:

- Identifies the customer's intent.
- Finds similar historical Spotify support cases.
- Generates a draft response based on previous Spotify responses.
- Decides whether the case can be auto-handled or should be escalated to a human.
- Provides the reason for the escalation decision.

The main goal is to build a small but explainable support pipeline where responses are grounded in how the brand historically handled similar customer issues.

---

## 2. Problem Framing

Customer-support messages are often short, informal, and difficult to interpret without context. A support agent therefore needs to understand the customer's intent and use relevant historical conversations before drafting a response.

This project frames the problem as four connected tasks:

1. **Intent Classification**  
   Identify what the customer is asking about.

2. **Historical Case Retrieval**  
   Find previous Spotify customer-support conversations that are similar to the new message.

3. **Grounded Reply Generation**  
   Draft a response using the retrieved historical Spotify support response as evidence.

4. **Auto-Handle vs Escalate**  
   Automatically handle messages when the model has sufficient confidence and historical evidence; otherwise, escalate them to a human.

### What This Project Does Not Attempt to Build

This is a prototype and does not attempt to:

- Access real Spotify customer accounts.
- Process payments, refunds, or subscription changes.
- Access private customer information.
- Send messages directly to customers.
- Perform real customer-support actions.
- Completely replace human support agents.
- Build a full production-scale customer-support platform.

---

## 3. Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle.

The dataset contains millions of customer-support tweets and responses from multiple brands. For this project, only conversations involving **Spotify** were selected.

### Dataset Summary

| Property | Value |
|---|---:|
| Processed Spotify rows | 43,206 |
| Usable customer messages | 42,749 |
| Unique customer messages | 40,689 |
| Date range | April 7, 2017 – September 27, 2017 |

The processed data contains customer messages together with historical Spotify support responses.

A manually labelled **golden evaluation set** of 200 examples was created. After removing two examples with missing customer text, **198 usable examples** were evaluated.

The golden set was kept separate from training data, and exact customer-message overlap was removed from the weakly labelled training set.

---

## 4. Intent Taxonomy

Based on the Spotify support conversations, the project defines the following 11 intent categories:

| Intent | Description |
|---|---|
| `ACCOUNT_LOGIN` | Login, account access, and account-related issues |
| `APP_CRASH_PERFORMANCE` | App crashes, freezing, and performance problems |
| `AVAILABILITY_REGION` | Country, region, and content availability issues |
| `BILLING_SUBSCRIPTION` | Billing, Premium, subscription, and payment-related issues |
| `BUG_REPORT` | General reports of unexpected behaviour or bugs |
| `CONNECTIVITY_SYNC` | Internet connection, device connection, and synchronization issues |
| `FEATURE_REQUEST` | Requests or suggestions for new features |
| `FEEDBACK` | General opinions, comments, and user feedback |
| `OTHER_UNCLEAR` | Messages that are incomplete, ambiguous, or difficult to classify |
| `PLAYBACK_ISSUE` | Problems with playing music or audio |
| `PLAYLIST_LIBRARY` | Playlist, saved music, and library-related issues |

`OTHER_UNCLEAR` is treated mainly as a fallback category because many short Twitter messages cannot be reliably understood without the surrounding conversation context.

---

## 5. System Architecture

The overall pipeline is:

```text
                 Customer Message
                       |
                       v
              +-------------------+
              | Preprocessing     |
              | & Text Cleaning   |
              +-------------------+
                       |
                       v
              +-------------------+
              | Intent            |
              | Classification    |
              +-------------------+
                       |
                       v
              +-------------------+
              | Historical Case   |
              | Retrieval         |
              +-------------------+
                       |
                       v
              +-------------------+
              | Grounded Reply    |
              | Generation        |
              +-------------------+
                       |
                       v
              +-------------------+
              | Escalation        |
              | Decision          |
              +-------------------+
                    /       \
                   /         \
                  v           v
           AUTO-HANDLE     ESCALATE
