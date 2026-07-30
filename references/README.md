# Agent Memory Research Notes

This directory contains background research collected while studying memory
systems for LLM agents. These materials are provided for context only and are
not required to install or run MiniMem.

Most notes are written in Chinese. Paper titles, repository links, commit
references, and technical identifiers retain their original English names.

## Paper indexes

| File | Entries | Description |
| --- | ---: | --- |
| [all-papers.csv](all-papers.csv) | 45 | Complete paper and project index. |
| [papers-with-open-source-code.csv](papers-with-open-source-code.csv) | 25 | Papers and projects with a public implementation identified during the research. |
| [papers-without-open-source-code.csv](papers-without-open-source-code.csv) | 20 | Papers for which no public implementation was confirmed during the research. |

The `.csv` files currently contain Markdown-style tables rather than
comma-separated records. They are best viewed as plain text or with a Markdown
table viewer.

The open-source classification is a research snapshot. Repository
availability, publication status, and GitHub star counts may change over time.

## Detailed reading notes

Start with the [comparative project summary](paper-reading/0_project_summary.md)
for a cross-project overview of memory representation, construction,
retrieval, management, and evaluation.

| Project | Main topic |
| --- | --- |
| [A-MEM](paper-reading/A-MEM.md) | Agentic memory inspired by Zettelkasten-style linked notes. |
| [G-Memory](paper-reading/G-Memory.md) | Hierarchical memory tracing for multi-agent systems. |
| [LatentMem](paper-reading/LatentMem.md) | Customizable latent memory for multi-agent systems. |
| [LightMem](paper-reading/Light-Mem.md) | Lightweight multi-stage memory with low-cost updates. |
| [MIRIX](paper-reading/MIRIX.md) | A multi-agent system coordinating several memory types. |
| [Mem0](paper-reading/Mem0.md) | A production-oriented long-term memory layer for agents. |
| [MemEvolve](paper-reading/MemEvolve.md) | Meta-evolution of modular memory architectures. |
| [MemGPT](paper-reading/MemGPT.md) | Operating-system-inspired context and memory management. |
| [MemOS](paper-reading/MemOS.md) | Memory as a managed resource for augmented generation. |
| [MemoryBank](paper-reading/MemoryBank.md) | Long-term conversational memory with forgetting behavior. |
| [MemoryOS](paper-reading/MemoryOS.md) | Hierarchical short-, mid-, and long-term agent memory. |
| [SimpleMem](paper-reading/SimpleMem.md) | Structured compression and intent-aware retrieval. |
| [Zep](paper-reading/Zep.md) | Temporal knowledge-graph memory built around Graphiti. |

## How to read the notes

The detailed notes compare the projects across a common set of questions:

1. What is stored as a memory?
2. When and how is memory constructed?
3. How is memory retrieved?
4. How is retrieved memory added to the LLM prompt?
5. How are duplication, contradiction, consolidation, and forgetting handled?
6. Which benchmarks and baselines are used?
7. How does the paper differ from the inspected implementation?

Where available, code observations include repository commit identifiers and
`file:line` references. Those references are valid only for the recorded
snapshot and may not match newer versions of the upstream project.

## Scope

These files summarize third-party projects and papers. They do not imply that
MiniMem implements the reviewed systems, and they should not be treated as
reproducible benchmark results. Consult the original papers and repositories
for authoritative descriptions, licenses, and current implementation details.
