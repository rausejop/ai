# Chapter 8.4: Structured Prompting with RISEN and CARE

## 1. Why Use a Structured Framework?
In the early era of AI development, prompts were often created through ad-hoc, messy experimentation (informally known as "prompt hacking"). However, to transition from a prototype to a stable, industrial-grade production system, the process must be formalized. **Structured Frameworks** ensure that prompts are comprehensive, reproducible, and optimized for the model's self-attention mechanism, minimizing the risk of "off-topic" generation.

## 2. RISEN Framework: Role, Instruction, Steps, Examples, Negative Constraints
The **RISEN** framework provides a definitive checklist for a "Perfect Prompt":
- **Role**: Setting the model's persona (e.g., "Act as a Senior Cyber-Security Auditor").
- **Instruction**: The unambiguous primary task.
- **Steps**: Breaking down the process into a multi-step sequence.
- **Examples**: Providing few-shot context to define quality and formatting.
- **Negative Constraints**: Defining the "Blacklist" of behaviors (e.g., "Do not use technical jargon" or "Only use information from the provided context"). This is essential for ensuring safety and precision.

## 3. CARE Framework: Context, Action, Response, Examples
The **CARE** framework is often preferred for creative, summarization, or synthesis tasks:
- **Context**: The background story or specific document to be analyzed.
- **Action**: What the model needs to build or generate.
- **Response**: Detailed constraints on the output (e.g., "Output as a Markdown table with 3 columns").
- **Examples**: Style references designed to guide the model's "voice."

## 4. Demo: Applying a Framework to a Complex Task
By applying RISEN to a task like "Summarize a Technical whitepaper," a developer can move from a vague prompt like *"Summarize this"* to a high-density instruction: *"Act as an expert technical reviewer. 1. Read the provided text. 2. Identify the top 3 innovations. 3. Format as a bulleted list. 4. Do NOT mention the author names. Example: [Formatted Example]."*. The result is a consistent, high-fidelity output that can be reliably parsed by a downstream software application.

## 5. Designing Your Own Custom Template
In most production apps, prompts are **Dynamic Templates**. Within the backend code, developers define a fixed "System Instruction" using a framework like RISEN, and then use placeholders (e.g., `{{COMPANY_NAME}}`, `{{SOURCE_DOC}}`) to inject user-specific data at runtime. This modular architecture allows a single core prompt to handle millions of unique, personalized interactions, providing the stability and scale required for mission-critical enterprise AI deployments.
