# Chapter 8.4: Structured Prompting with RISEN and CARE

## 1. Why Use a Structured Framework?
In the early era of AI development, prompts were often created through ad-hoc, messy experimentation (informally known as "prompt hacking"). However, to transition from a prototype to a stable, industrial-grade production system, the process must be formalized. **Structured Frameworks** ensure that prompts are comprehensive, reproducible, and optimized for the model's self-attention mechanism, minimizing the risk of "off-topic" generation.

## 2. RISEN Framework: Role, Instruction, Steps, Examples, Negative Constraints
The **RISEN** framework provides a definitive checklist for a "Perfect Prompt":
- **Role**: Setting the model's persona (e.g., "Act as a Senior Cyber-Security Auditor").
- **Instruction**: The unambiguous primary task.
- **Steps**: Breaking down the process into a multi-step sequence.
- **Examples**: Providing few-shot context to define quality and formatting.
- **Negative Constraints**: Defining the "Blacklist" of behaviors (e.g., "Do not use technical jargon").

## 3. CARE Framework: Context, Action, Response, Examples
The **CARE** framework is often preferred for creative, summarization, or synthesis tasks:
- **Context**: The background story or specific document to be analyzed.
- **Action**: What the model needs to build or generate.
- **Response**: Detailed constraints on the output (e.g., "Output as a Markdown table").
- **Examples**: Style references designed to guide the model's "voice."

## 4. Demo: Applying a Framework to a Complex Task
By applying RISEN to a task like "Summarize a Technical whitepaper," a developer can move from a vague prompt like *"Summarize this"* to a high-density instruction: *"Act as an expert technical reviewer. 1. Read text. 2. Identify innovations. 3. Format as a list. 4. Do NOT mention author names."*.

## 5. Designing Your Own Custom Template
In most production apps, prompts are **Dynamic Templates**. Within the backend code, developers define a fixed "System Instruction" using a framework like RISEN, and then use placeholders (e.g., `{{COMPANY_NAME}}`) to inject user-specific data at runtime. This modular architecture allows a single core prompt to handle millions of unique interactions.

## 📊 Visual Resources and Diagrams

- **The RISEN vs. CARE Comparison Chart**: A visual matrix explaining when to use each framework.
    ![The RISEN vs. CARE Comparison Chart](https://www.microsoft.com/en-us/research/uploads/prod/2021/04/Prompt-Frameworks.png)
    - [Source: Microsoft Research - Formalizing Prompt Structures](https://www.microsoft.com/en-us/research/uploads/prod/2021/04/Prompt-Frameworks.png)
- **Modular Prompt Architecture**: An infographic showing the separation of System Instructions vs. User Data.
    ![Modular Prompt Architecture](https://www.anthropic.com/images/prompt-modular.png)
    - [Source: Anthropic Tech Blog - Designing for Reliability](https://www.anthropic.com/images/prompt-modular.png)

## 🐍 Technical Implementation (Python 3.14.2)

A **RISEN Framework Validator** that ensures a prompt contains all necessary professional components on Windows.

```python
import re # Importing the regular expression module for high-fidelity pattern matching within the prompt string

def validate_risen_structure(prompt: str): # Defining a function to audit a prompt for compliance with the RISEN framework
    """ # Start of the function's docstring
    Checks for the presence of the 5 RISEN components in a raw string. # Explaining the pedagogical goal of structured prompt validation
    Compatible with Python 3.14.2. # Specifying the target version for 2026 industrial platforms
    """ # End of docstring
    # Defining regex patterns for each of the five critical RISEN components
    components = {
        "Role": r"(Act as|You are a|Role:)", # Identifying the semantic anchor for the model's persona
        "Instruction": r"(Instruction:|Task:|Analyze|Create)", # Identifying the core mission directive
        "Steps": r"(Step 1|Firstly|Process:)", # Identifying the logical decomposition of the task
        "Examples": r"(Example:|Sample:)", # Identifying the few-shot context for quality guidance
        "Negative Constraints": r"(Do not|Avoid|Negative Constraints:)" # Identifying the behavioral blacklists for safety and tone
    } # Closing regex component dictionary
    
    findings = {} # Initializing a dictionary to store the logical audit results
    for name, pattern in components.items(): # Iterating through the target RISEN components
        # Performing a case-insensitive search for the current component's pattern in the target prompt
        findings[name] = bool(re.search(pattern, prompt, re.IGNORECASE)) # Mapping the component name to a Boolean presence flag
        
    # Calculating the final framework compliance score as a percentage of identified components
    score = sum(findings.values()) / len(findings) # Deriving a normalized scalar score
    return findings, score # Returning the component breakdown and the final architectural quality score

if __name__ == "__main__": # Entry point check for script execution
    # Defining a sample prompt that follows the RISEN architectural requirements
    my_prompt = """
    Act as a Senior Auditor. Instruction: Check the JSON for errors.
    Process: 1. Validate keys. 2. check types. 
    Example: { "key": "val" }. 
    Do not include conversational text.
    """ # Closing the sample industrial prompt string
    
    report, final_score = validate_risen_structure(my_prompt) # Executing the RISEN audit routine on the sample prompt
    print(f"Framework Quality Score: {final_score:.0%}") # Displaying the final overall compliance percentage
    for part, present in report.items(): # Iterating through the individual component results for a detailed diagnostic report
        print(f" - {part}: {'[PASS]' if present else '[MISSING]'}") # Outputting the specific PASS/MISSING status for each RISEN pillar
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Mesko (2023)**: *"Prompt Engineering as a New Domain in Medicine: The RISEN Framework"*. One of the first academic applications of RISEN.
    - [Link to Journal of Medical Systems](https://link.springer.com/article/10.1007/s10916-023-01961-z)
- **Yao et al. (2023)**: *"A Survey on Prompt Engineering for Foundation Models"*.
    - [Link to ArXiv](https://arxiv.org/abs/2307.03152)

### Frontier News and Updates (2025-2026)
- **Google DeepMind (Late 2025)**: Development of *Auto-Framework*, an LLM that automatically wraps raw user queries in the optimal RISEN structure before execution.
- **NVIDIA AI Blog**: "The Token Latency of Frameworks"—How to optimize long RISEN prompts for the Rubin hardware cache.
- **Anthropic Tech Blog**: "The Ethics of Role-Play"—A study on how setting extreme 'Roles' in RISEN impacts model safety boundaries.
