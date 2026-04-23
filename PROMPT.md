# PROMPT.md — Prompts Used for KDE Extraction

Each prompt is run per-chunk (documents are split into ~20k-char overlapping
chunks), and the resulting KDEs are merged by element name across chunks.
`{document_text}` is substituted with one chunk's content at inference time.

## zero-shot

```
Analyze the following CIS Benchmark security requirements document.

Identify all Key Data Elements (KDEs). A Key Data Element is a distinct security control or configuration area (e.g., "Logging", "Kubelet Configuration", "Network Policies"). Each KDE may map to one or more specific requirements or recommendations.

For each KDE, provide:
- name: The name of the key data element
- requirements: A list of specific requirements or recommendations that belong to this element

Format your response as YAML with this structure:
element1:
  name: "<element name>"
  requirements:
    - "<requirement 1>"
    - "<requirement 2>"
element2:
  name: "<element name>"
  requirements:
    - "<requirement 1>"

DOCUMENT TEXT:
{document_text}

Respond ONLY with the YAML output. Do not include any other text.
```

## few-shot

```
Analyze a CIS Benchmark security requirements document and extract Key Data Elements (KDEs). A KDE is a distinct security control area with its associated requirements.

Here are examples of correct KDE extraction:

EXAMPLE INPUT:
"3.1 Worker Node Configuration Files
3.1.1 Ensure that the kubeconfig file permissions are set to 644 or more restrictive
3.1.2 Ensure that the kubelet kubeconfig file ownership is set to root:root
3.2 Kubelet
3.2.1 Ensure that the Anonymous Auth is Not Enabled"

EXAMPLE OUTPUT:
element1:
  name: "Worker Node Configuration Files"
  requirements:
    - "Ensure that the kubeconfig file permissions are set to 644 or more restrictive"
    - "Ensure that the kubelet kubeconfig file ownership is set to root:root"
element2:
  name: "Kubelet"
  requirements:
    - "Ensure that the Anonymous Auth is Not Enabled"

EXAMPLE INPUT:
"2.1 Logging
2.1.1 Enable audit Logs
4.1 Pod Security Standards
4.1.1 Ensure that the cluster has at least one active policy control mechanism in place
4.1.2 Ensure that the default namespace is not used"

EXAMPLE OUTPUT:
element1:
  name: "Logging"
  requirements:
    - "Enable audit Logs"
element2:
  name: "Pod Security Standards"
  requirements:
    - "Ensure that the cluster has at least one active policy control mechanism in place"
    - "Ensure that the default namespace is not used"

Now extract all KDEs from this document:

DOCUMENT TEXT:
{document_text}

Respond ONLY with the YAML output. Begin your response with 'element1:' — do not write any text before the YAML.
```

## chain-of-thought

```
Analyze the following CIS Benchmark document and extract Key Data Elements (KDEs).

Before writing your answer, think through:
- What are the major security section headings in this document? (e.g. "Kubelet", "Logging", "Pod Security Standards")
- What specific requirements (numbered items like "3.1.1 Ensure...") appear under each section?
- How should requirements be grouped under their parent section name?

Output ONLY the YAML below — no preamble, no reasoning, no markdown fences:
element1:
  name: "<section name>"
  requirements:
    - "<requirement text>"
element2:
  name: "<section name>"
  requirements:
    - "<requirement text>"

DOCUMENT TEXT:
{document_text}
```
