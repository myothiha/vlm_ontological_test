## Conceptual Knowledge and its Five Dimensions

Conceptual knowledge broadly refers to the understanding of categories, classifications, and the relationships between entities within any given domain. It is highly faceted, requiring an ontological representation that captures not only a concept's intrinsic baseline properties but also its dynamic behaviors and its network of relationships. To comprehensively model this information, a concept can be characterized across five distinct dimensions, representing different facets ranging from its structural characteristics to its behavior under specific contextual conditions. While we apply this five-dimensional framework specifically to the biomedical domain as a primary application, these principles can be applied to represent conceptual knowledge in any field.

### 1. Data Properties
**Description:** Data properties refer to the intrinsic, measurable, structural, chemical, or physiological characteristics of the class itself in a baseline state. This dimension includes quantitative and qualitative attributes such as size, weight, shape, color, tissue composition, and baseline concentration levels.
**Examples:**
- *Liver:* "The normal size of the liver is about 15–17 cm in adults."
- *Liver:* "The typical color of the liver is reddish-brown."
Certain concepts naturally lack data properties. For example, diseases, pathological processes, diagnostic modalities, and visual descriptors typically do not have this dimension because they represent states, procedures, or abstract patterns rather than physical, baseline objects.

### 2. Functions or Behaviors
**Description:** This dimension captures the biological, pathological, clinical, or diagnostic roles that the class performs or exhibits within healthcare contexts. It focuses on the active purpose or typical physiological actions of the entity.
**Examples:**
- *Liver:* "The liver detoxifies harmful substances by metabolizing toxins and drugs through enzymatic pathways."
- *X-ray:* "X-ray is used to detect bone fractures by visualizing disruptions in bone continuity."
- *Necrosis:* "Necrosis plays a role in tissue injury by causing irreversible cell death and loss of tissue integrity."
Visual or histological descriptors represent static observational patterns rather than dynamic entities. As such, they generally do not perform actionable functional roles and thus lack this dimension.

### 3. Relational Knowledge
**Description:** Relational knowledge defines how the class is connected to, associated with, contained within, affects, or is affected by other biomedical entities. This dimension maps the topological, anatomical, and pathological networks of the concept.
**Examples:**
- *Liver:* "The liver belongs to the digestive system."
- *Breast cancer:* "Breast cancer affects breast epithelial tissue and surrounding stromal tissue."
- *X-ray:* "X-ray is commonly used to examine bones, lungs, chest, and joints."
This dimension is often not applicable to visual or histological descriptors. Because they are abstract visual findings, they do not typically maintain independent topological, anatomical, or causal relationships outside of the specific condition they describe.

### 4. Contextual Properties
**Description:** Contextual properties represent the measurable characteristics of a class whose values change in response to specific physiological, pathological, or clinical conditions. This dimension captures states of structural or compositional alteration (e.g., changes in size, color, density, or measurable biochemical levels due to disease).
**Examples:**
- *Liver:* "The liver accumulates excess fat during metabolic disorders such as obesity and chronic alcohol use."
- *Liver:* "The liver changes color in conditions like jaundice or advanced liver disease."
Some concepts, such as diseases, pathological processes, diagnostic modalities, and visual descriptors, usually lack contextual properties. This is because they are not baseline physical objects that undergo intrinsic structural changes.

### 5. Contextual Behaviors
**Description:** Contextual behaviors describe the dynamic modulation of a class's predefined functions, actions, or roles under specific physiological, pathological, clinical, diagnostic, or environmental conditions. This dimension isolates how external or systemic changes affect the entity's functional capacity.
**Examples:**
- *Liver:* "The liver reduces its detoxification activity during advanced liver disease and hepatic failure."
- *X-ray:* "X-ray becomes less effective when patient movement causes motion artifacts."
Visual and histological descriptors generally do not possess contextual behaviors, as they lack foundational functions or active roles that could be dynamically modulated by an environment.

---

### Ontological Variance Across Dimensions
Not all biomedical concepts possess valid assertions across all five dimensions. The presence or absence of a dimension is heavily dictated by the ontological category and inherent nature of the concept being modeled:

- **Anatomical Entities (e.g., Liver):** These typically populate all five dimensions. As physical, biological objects, they possess measurable physical characteristics (Data Properties), perform biological roles (Functions or Behaviors), interact structurally with other organs (Relational Knowledge), and undergo measurable changes in disease states (Contextual Properties and Contextual Behaviors).
- **Diseases and Pathological Processes (e.g., Breast Cancer, Necrosis):** These entities are processes or disease states rather than baseline anatomical objects. Consequently, they often lack intrinsic physical measurements (Data Properties) or Contextual Properties as standalone physical parameters. Instead, they are primarily defined by what structures they affect (Relational Knowledge), their characteristic mechanisms (Functions or Behaviors), and how their impact worsens or progresses under distinct conditions (Contextual Behaviors).
- **Diagnostic Modalities (e.g., X-ray):** Diagnostic tools and procedures are human-engineered actions or technologies. They perform distinct functions (e.g., imaging) and exhibit relational knowledge (e.g., application to specific body regions). Their effectiveness can also fluctuate based on patient conditions (Contextual Behaviors). However, they inherently lack physiological properties such as cellular composition, biological weight, or metabolic pH, rendering their Data Properties and Contextual Properties dimensions conceptually inapplicable.
- **Visual or Histological Descriptors (e.g., Whorled appearance):** Descriptive concepts serve as abstract observational patterns. They do not possess intrinsic biological functions, physical mass, or active behaviors. Consequently, representing these concepts yields empty assertions across most, if not all, conceptual dimensions within this framework.
