import json
import os

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class ReportGenerator:
    """
    Handles LLM prompt injection and generation for the AI Report feature.
    """
    
    SYSTEM_PROMPT = """You are an objective, highly rigorous bioinformatics reporting assistant. 
You are receiving a JSON payload containing explicitly calculated statistical findings from a lipidomics analysis pipeline.

CRITICAL RULES:
1. No Hallucination: You may ONLY report on the lipid classes, species, PCA separation, and statistical values explicitly provided in the JSON input. Do not invent findings.
2. No Speculation: DO NOT infer downstream biological mechanisms, pathways, or clinical disease relevance unless explicitly instructed.
3. No External Literature: DO NOT cite external literature or prior knowledge. Rely ONLY on the provided data.
4. Objective Tone: State the magnitude and direction of change objectively. Use terms like "significantly enriched" only if a p-value/FDR is provided and is < 0.05.
5. Report Level Constraints: If the JSON specifies "report_level": "descriptive", you MUST NOT discuss statistical significance, p-values, or confident enrichments. You must only discuss general trends and visual separation.
6. Image Embedding: If the payload contains an 'available_images' list, you MUST embed those exact markdown images (e.g., `![Image Title](/reports/filename.png)`) directly into the corresponding sections of your report. You must provide 1-2 paragraphs immediately below the image explaining what the reader is looking at, using the statistical data to guide your explanation of the visual trends.
7. Extreme Verbosity: You must expand heavily on all provided data. Write a massive, comprehensive, multi-page report. Use at least 2-3 detailed paragraphs per analytical section. Do not summarize briefly; expand on the nuances of the shifts.
"""

    AUDIENCE_PROMPTS = {
        "Layperson": "Explain the findings so a high school biology student can understand. Use simple analogies. Refer to 'lipid molecules' instead of complex nomenclature (e.g., say 'a type of cell membrane fat' instead of 'Phosphatidylcholine'). Do not report raw p-values or log2 fold changes, but describe the size of the shift. Focus heavily on which groups separate in PCA.",
        "Clinician": "Focus on the overall profile shifts, directional consistency, and the most dramatic outliers. Keep it concise and actionable. Include p-values but prioritize the big picture (e.g., 'Cohort A exhibits a distinct lipidomic signature from B').",
        "Researcher": "Write a formal results section suitable for a peer-reviewed scientific journal. Use precise lipid nomenclature. Report exact log2 fold changes, PCA variance, and FDR-corrected p-values. Be terse and statistically rigorous."
    }

    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        if HAS_OPENAI and self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            self.client = None

    async def generate_report_stream(self, context_dict, audience="Researcher"):
        """
        Yields markdown chunks from the LLM.
        """
        payload_json = json.dumps(context_dict, indent=2)
        
        audience_instruction = self.AUDIENCE_PROMPTS.get(audience, self.AUDIENCE_PROMPTS["Researcher"])
        
        user_msg = f"""AUDIENCE INSTRUCTION:
{audience_instruction}

DATA PAYLOAD:
```json
{payload_json}
```

Now write the report in EXACTLY this order. Use a simple markdown header (e.g. `## Section Name`) for each section. If an image from `available_images` corresponds to a section, embed it as a markdown image and explain it:
1. Executive Summary
2. Dataset & Confidence Overview
3. PCA & Global Profile Separation
4. Acyl Chain Length Distribution & Shifts
5. Unsaturation Analysis
6. Lipid Class & Head Group Dynamics
7. Cohort-Specific Outliers
8. Limitations (Use the verbatim limitations from the payload)

Generate the comprehensive multi-page markdown report now."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        if not self.client:
            # Stub mode for local testing if no API key is present
            yield "### AI Report Generation Simulator\n\n"
            yield "> **Note:** OpenAI API key not found or library missing. Returning simulated report.\n\n"
            yield "#### Dataset & Confidence Summary\n"
            overview = context_dict.get("experiment_overview", {})
            yield f"- Total Samples: {overview.get('total_samples', 'Unknown')}\n"
            yield f"- Comparison Cohorts: {', '.join(overview.get('comparison_cohorts', []))}\n"
            yield "\n#### PCA Findings\n"
            for finding in context_dict.get("pca_clustering_findings", []):
                yield f"- {finding}\n"
            yield "\n#### Top Findings\n"
            for change in context_dict.get("top_consistent_changes_across_cohorts", []):
                yield f"- {change['lipid']} showed a {change['trend']} trend (Log2FC: {change['log2FC_avg']}).\n"
            yield "\n#### Limitations\n"
            for limit in context_dict.get("calculated_limitations", []):
                yield f"- {limit}\n"
            return
            
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.2  # Low temperature for analytical consistency
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"LLM stream error: {e}", flush=True)
            yield "\n\n**Error communicating with the AI provider. Please check your API key and try again.**"
