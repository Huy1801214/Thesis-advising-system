Intelligent Academic & Career Advisory System
Multi-Agent System Design Specification
1. Problem Statement

The current academic counseling system mainly answers questions based on university regulations and curriculum information. It lacks the ability to provide personalized recommendations regarding careers, fields of study, and future course planning.

A simple Retrieval-Augmented Generation (RAG) system is insufficient because career advising requires reasoning over multiple heterogeneous knowledge sources:

Student academic records
Student interests and goals
Curriculum structure
Subject learning outcomes
University regulations
Career knowledge
Labor market trends

No single source contains enough information to make an informed recommendation.

Furthermore, career knowledge changes continuously. New technologies emerge, required skills evolve, and industry expectations shift. Therefore, the system must continuously maintain and update its knowledge rather than relying on static documents.

The objective of this system is to construct a collaborative multi-agent architecture capable of maintaining an evolving knowledge base and performing explainable decision support for students.
2. Why Multi-Agent Instead of One LLM?

A single LLM can retrieve documents and generate responses.

However, career advising is not a single task.

It consists of several independent reasoning processes.

For example:

Understanding the student's academic status
Understanding the university curriculum
Understanding current career requirements
Comparing competencies
Validating recommendations

Each process requires different knowledge and different tools.

Separating them into specialized agents provides:

Better modularity
Easier maintenance
Independent evaluation
Easier expansion
Better explainability
Parallel execution

Instead of asking one model to solve everything, each agent becomes an expert responsible for one domain.

3. Overall Architecture

The system consists of two independent subsystems.

Offline Knowledge Maintenance

Responsible for continuously maintaining the knowledge base.

Runs automatically according to a schedule.

No student interaction.

Online Advisory System

Activated only when a student asks a question.

Uses the maintained knowledge base to generate recommendations.

4. Offline Knowledge Maintenance

The purpose of this subsystem is to ensure that the knowledge base always reflects current career information and curriculum knowledge.

Instead of storing raw web pages, it stores structured knowledge.

Agent 1 — Knowledge Curation Agent
Purpose

Maintain the quality and freshness of career knowledge.

This agent is responsible for discovering new information from trusted sources and converting it into structured knowledge suitable for reasoning.

Responsibilities
Receive scheduled update tasks.
Generate search plans based on supported careers.
Retrieve documents from trusted knowledge sources.
Remove duplicate documents.
Filter irrelevant information.
Extract structured knowledge using LLMs.
Normalize terminology.
Produce candidate knowledge updates.
Input
Supported career list.
Trusted source list.
Existing knowledge base.
Output

Structured career knowledge.

Not raw articles.

Internal Workflow
Generate search queries for every supported career.
Retrieve documents.
Clean document contents.
Extract structured information.
Normalize terminology.
Pass structured knowledge to the Comparison Agent.
Why it exists

The Internet contains noisy and inconsistent information.

This agent converts unstructured information into standardized knowledge.

Agent 2 — Knowledge Comparison Agent
Purpose

Detect differences between newly collected knowledge and existing knowledge.

Responsibilities
Compare current knowledge with newly extracted knowledge.
Detect new concepts.
Detect removed concepts.
Detect modified concepts.
Detect conflicting information.
Measure confidence.
Produce a change report.
Input
Existing knowledge.
Newly extracted knowledge.
Output

Knowledge difference report.

Internal Workflow
Load existing knowledge.
Compare entities.
Compare relationships.
Detect additions.
Detect removals.
Detect modifications.
Compute confidence.
Produce update proposal.
Why it exists

Not every new document should immediately modify the knowledge base.

The system should understand what has changed before updating.

Agent 3 — Knowledge Update Agent
Purpose

Maintain the knowledge graph.

Responsibilities
Receive update proposals.
Validate update rules.
Apply approved changes.
Update Neo4j.
Update Qdrant.
Preserve knowledge consistency.
Input

Knowledge difference report.

Output

Updated knowledge base.

Internal Workflow
Receive proposed changes.
Evaluate update policy.
Update graph entities.
Update relationships.
Update vector documents.
Record update history.
Why it exists

Knowledge maintenance should be separated from knowledge collection.

This makes the knowledge base stable and explainable.

5. Online Advisory System

The online subsystem provides personalized consultation.

Unlike traditional chatbots, recommendations are generated through collaboration between multiple specialized agents.

Agent 4 — Planner Agent
Purpose

Control the execution of the entire system.

Responsibilities
Receive user requests.
Understand user intent.
Determine required reasoning tasks.
Decide which agents should participate.
Execute agents in parallel whenever possible.
Collect intermediate results.
Input

Student request.

Output

Execution plan.

Internal Workflow
Analyze user intent.
Decompose the task.
Identify required agents.
Dispatch tasks.
Wait for completion.
Forward all results to the Reasoning Agent.
Why it exists

Different questions require different expertise.

Not every request needs every agent.

Agent 5 — Student Analysis Agent
Purpose

Understand the student's current academic profile.

Responsibilities
Analyze transcript.
Analyze completed subjects.
Analyze GPA.
Analyze interests.
Analyze career objectives.
Analyze learning progress.
Infer current competencies.
Input

Student profile.

Output

Student competency profile.

Internal Workflow
Load academic records.
Load subject history.
Map completed subjects to competencies.
Evaluate academic progress.
Produce structured competency profile.
Why it exists

Career recommendations must depend on the student's actual capabilities rather than only career requirements.

Agent 6 — Knowledge Reasoning Agent
Purpose

Perform the main reasoning process.

This is the core intelligence of the system.

Responsibilities
Query Neo4j.
Query Qdrant.
Retrieve career knowledge.
Retrieve curriculum knowledge.
Retrieve regulations.
Compare student competencies with career requirements.
Identify competency gaps.
Generate learning pathways.
Produce evidence-supported recommendations.
Input
Student profile.
Knowledge graph.
Vector knowledge.
University regulations.
Output

Reasoning result.

Internal Workflow
Receive execution context.
Retrieve related knowledge.
Compare competencies.
Compute skill gaps.
Evaluate curriculum options.
Generate recommendation evidence.
Produce structured recommendation.
Why it exists

Recommendations should be based on reasoning over multiple knowledge sources rather than document retrieval.

Agent 7 — Critic Agent
Purpose

Validate recommendation quality.

Responsibilities
Verify logical consistency.
Verify prerequisite constraints.
Verify regulation compliance.
Detect contradictions.
Detect missing evidence.
Request revision if necessary.
Input

Recommendation.

Output

Approval or revision request.

Internal Workflow
Analyze recommendation.
Validate supporting evidence.
Validate academic constraints.
Detect inconsistencies.
Approve or reject.
Why it exists

The recommendation generator should not evaluate itself.

Independent validation improves reliability.

6. Agent Collaboration

The system follows a cooperative architecture, not a pipeline.

The Planner Agent orchestrates all participating agents.

Whenever possible, specialized agents execute independently and in parallel because they rely on different data sources and have separate responsibilities.

Each agent produces a structured result rather than a final answer.

These intermediate results are passed to the Knowledge Reasoning Agent, which synthesizes them into a coherent recommendation.

The Critic Agent performs an independent review. If the recommendation fails validation, it requests the Reasoning Agent to regenerate or refine the output.

This design minimizes unnecessary dependencies while still allowing targeted communication between agents when additional evidence is required.

7. Data Flow
Offline

Scheduler
        │
Knowledge Curation Agent
        │
Knowledge Comparison Agent
        │
Knowledge Update Agent
        │
Neo4j + Qdrant


Online

Student
        │
Planner Agent
        │
 ┌──────┼──────────────┐
 │      │              │
Student Knowledge   (other specialized agents if needed)
Analysis Reasoning
 │         │
 └──────┬──┘
        │
Critic Agent
        │
Final Recommendation
8. Research Contribution

The contribution of this work is not simply using an LLM for recommendations.

The contribution is the design of a collaborative multi-agent decision-support architecture that:

Continuously curates and maintains evolving career knowledge from trusted sources.
Constructs a structured knowledge graph connecting careers, competencies, university subjects, and regulations.
Separates complex advisory tasks into specialized reasoning agents with clear responsibilities.
Produces explainable recommendations based on explicit evidence rather than opaque LLM responses.
Supports long-term knowledge evolution through autonomous maintenance instead of relying on a static RAG database.
One architectural refinement

After reviewing everything we've discussed, I would merge the Knowledge Comparison Agent and Knowledge Update Agent into a single "Knowledge Maintenance Agent." Their responsibilities are tightly coupled (compare → decide → update), and keeping them separate increases implementation complexity without adding much research value.

That leaves you with six well-defined agents:

Knowledge Curation Agent (collects and structures trusted career knowledge)
Knowledge Maintenance Agent (compares, validates, and updates the knowledge base)
Planner Agent (orchestrates execution)
Student Analysis Agent (builds the student's competency profile)
Knowledge Reasoning Agent (performs cross-domain reasoning and recommendation)
Critic Agent (validates and refines recommendations)

This architecture is balanced: each agent has a distinct responsibility, the collaboration is easy to explain, and the overall system remains achievable for a thesis while clearly demonstrating the advantages of a multi-agent approach over a single RAG-based chatbot.