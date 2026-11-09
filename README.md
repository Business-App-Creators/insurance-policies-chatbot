# Insurance Policies Chatbot

An agent that answers questions about Northline Assurance claims handling by reading the company's own policy manuals. It is built for claims staff, adjusters, and anyone who needs a procedure from the library without hunting through ten long documents.

The manuals in `policies-documents/` are the source of truth. They cover the life of a claim, from first notice through coverage, investigation, reserves, payment, denial, fraud referral, recovery, complaints, and catastrophe handling. The agent quotes those manuals. It does not invent a coverage result, a payment amount, or a deadline that the library does not state.

## Goals

- Answer a claims question from the policy that actually governs it, and name the document.
- Keep answers inside the library. If the manuals do not decide the question, say so.
- Treat the issued insurance contract and the law as controlling when a manual says they are stricter than the internal rule.
- Give a procedure someone can follow: who may act, what must be in the file, and when a supervisor has to approve.
- Stay useful as the library grows, without mixing one policy's rule into another.

## What is in the library

| Manual | Subject |
| --- | --- |
| CLM-001 | Claim submission and first notice of loss |
| CLM-002 | Coverage verification and policy interpretation |
| CLM-003 | Investigation, inspection, and evidence |
| CLM-004 | Reserves, estimating, and loss measurement |
| CLM-005 | Claim payments, settlements, and releases |
| CLM-006 | Denials, partial denials, and reservations of rights |
| CLM-007 | Fraud, special investigation, and misrepresentation |
| CLM-008 | Subrogation, salvage, and third-party recovery |
| CLM-009 | Appeals, complaints, and dispute resolution |
| CLM-010 | Catastrophe and large-loss claims |

## What comes next

The policy text is in place. The agent around it is not built yet.

- A data ingestion pipeline will chunk the manuals and load them into the database.
- The chatbot will retrieve the relevant sections and answer from those sections.
- An interface will be added later, so people can ask questions in a chat window instead of calling the agent from code.

Until that interface exists, this repository is the policy library and the place the agent will be built.
