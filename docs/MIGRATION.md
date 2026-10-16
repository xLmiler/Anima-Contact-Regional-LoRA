# Workflow migration

Menu-facing nodes use stable functional identifiers. Executable compatibility aliases retain the exact input keys, socket types and output positions used by all five legacy example graphs. Depth/VACE integrations and unused duplicate editors are outside this interface. Installing two plugin directories simultaneously can re-register old nodes and is not supported.

Use the generated functional workflows for a clean baseline. To retain a supported older workflow, run the migration tool with distinct source and destination paths:

```bash
python tools/migrate_workflow.py old_workflow.json migrated_workflow.json
```

The original file is preserved. Supported ownership, identity, scene, detection, SAM2, LLLite, Tile and inspection nodes are mapped to their functional counterparts. Adapter stacks and recognized input values are preserved; the matching policy is removed and strict matching applies. An embedded foreground setting is converted into an independent foreground node.

Retired pose/depth routes and unknown custom control formats require explicit manual replacement; the tool stops rather than guessing an equivalent model. It cannot reconstruct values that an earlier front end has already saved incorrectly. Review the migrated workflow before sampling.

Ownership editor width and height outputs are available after its existing report output. Connecting them to EmptyLatentImage keeps the latent grid aligned when switching reference sources. The supplied workflows already make these connections.
