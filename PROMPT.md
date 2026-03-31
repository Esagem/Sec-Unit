# PROMPT.md — Prompts Used for KDE Extraction

## Zero-Shot

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

## Few-Shot

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

Respond ONLY with the YAML output. Do not include any other text.
```

## Chain-of-Thought

```
Analyze the following CIS Benchmark security requirements document to extract Key Data Elements (KDEs).

Follow these steps carefully:

Step 1: Read through the document and identify all major section headings. These represent security control areas such as "Control Plane Components", "Worker Node Configuration", "Logging", "Policies", etc.

Step 2: For each major section, identify the specific recommendations or requirements listed under it. These typically start with numbered items like "3.1.1 Ensure that..." or "4.2.3 Restrict...".

Step 3: Group the requirements under their parent section. Each parent section becomes a Key Data Element (KDE) with a name and a list of requirements.

Step 4: Format the results as YAML with this exact structure:
element1:
  name: "<section name>"
  requirements:
    - "<requirement text>"
    - "<requirement text>"

DOCUMENT TEXT:
{document_text}

Now, work through the steps and provide your final YAML output. Respond ONLY with the YAML output after your reasoning.
```
