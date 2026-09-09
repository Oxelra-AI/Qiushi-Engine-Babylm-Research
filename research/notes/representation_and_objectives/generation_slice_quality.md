# generation slice quality — Qwen generation slice quality and provenance check
## Purpose
The slice tests whether the leader analysis and route pivot plan can safely scale Qwen-generated SimpleWiki derivatives before spending both H100s on the full 50k generation, 40k tokenizer, and 12×384 training package. The measurements are source/output process checks, not a substitute for official training evaluation.
## Run evidence
- Prompts: `experiments/archive/representation_and_objectives/training/data/generation_slice/slice_prompts.jsonl` (384 rows, sha256 `83cf27a6cd3ac729013d339d129ef440680fde48bca72f62bb8f8457966020f6`)
- Outputs: `experiments/archive/representation_and_objectives/training/runs/generation_slice_qwen/outputs.jsonl` (384 rows, sha256 `2f17430fbe95e086c013e276c129297487599afaf6f4d26a6f5e91c8b9acafd5`)
- Summary JSON: `experiments/archive/representation_and_objectives/training/data/generation_slice/slice_quality_summary.json`
- Row metrics JSONL: `experiments/archive/representation_and_objectives/training/data/generation_slice/slice_quality_rows.jsonl`

## Aggregate measurements
- simplification: n=128, words=3645, mean ratio=0.882, entity recall=0.546, number recall=0.944, new entities/row=1.406, bad-pattern rows=1, high-risk rows=47
- paraphrase: n=128, words=3965, mean ratio=0.953, entity recall=0.504, number recall=0.908, new entities/row=1.641, bad-pattern rows=0, high-risk rows=54
- expansion: n=128, words=15906, mean ratio=4.401, entity recall=0.528, number recall=0.888, new entities/row=4.555, bad-pattern rows=2, high-risk rows=99

Overall domain row hits: {"science_physical": 38, "quant_numeric": 127, "causal_relational": 52, "geography_places": 98, "people_history": 106, "institutions_society": 65, "media_culture": 64}

## Scientific interpretation
- The expansion prompt is scientifically unsafe as a bulk pretraining source without filtering: it often injects new names/numbers or meta-text not anchored in the SimpleWiki source.
- Simplification/paraphrase outputs are closer to the source than expansions and are better candidates for a low-risk source-paired data arm.
- The larger 50k generation should not proceed unchanged; a safer arm should keep original SimpleWiki/official text plus source-conservative rewrites, and treat free expansions as either excluded or heavily filtered.

## High-risk examples to inspect

### exp_004881 (expansion, article=Charles Evers)
- length_ratio=4.23, entity_recall=0.50, new_entity_count=1, number_recall=1.00, new_number_count=0, bad_patterns=['the original text', 'not named in the original', 'while the specific .* is not named']
- Source: He served as mayor of Fayette in Jefferson County. Thereafter, he ran for governor in 1971 and the United States Senate in 1978. He ran both times as an Independent candidate.
- Output: After serving as the mayor of Fayette, a city located in Jefferson County, the individual leveraged his local political experience to pursue higher office on the state and national stages. In 1971, he entered the gubernatorial race, seeking the governorship of his state, and subsequently attempted to secure a seat in the United States Senate in 1978. Notably, in both of these high-profile elections, he chose to run as an Independent candidate rather than aligning with the major political parties, a strategic decision that often reflects a desire to operate outside the traditional partisan framework and appeal to a broader, non-aligned electorate. While the specific state is not named in the original text, the sequence of events highlights a trajectory from municipal leadership to statewide and federal ambitions, illustrating a significant

### exp_007017 (expansion, article=John King Jr.)
- length_ratio=2.92, entity_recall=0.80, new_entity_count=5, number_recall=1.00, new_number_count=5, bad_patterns=['potentially a confusion']
- Source: The former Secretary of Education, Arne Duncan, was given the job of putting into effect the No Child Left Behind Act. However, King is required to put into effect the law that replaces that law, the Every Student Succeeds Act.
- Output: Arne Duncan, who served as the United States Secretary of Education from 2009 to 2015 under President Barack Obama, was the primary architect responsible for implementing the No Child Left Behind Act (NCLB) of 2001, a legislation that fundamentally reshaped American education by mandating standardized testing and holding schools accountable for student performance through a system of sanctions for underperforming institutions. In contrast, the current Secretary of Education, Miguel Cardona (often referred to as "King" in the context of recent appointments or potentially a confusion with another official, though the role described fits the current administration's mandate), is tasked with executing the Every Student Succeeds Act (ESSA), which was signed into law in 2015 to replace N

### simp_028595 (simplification, article=Dirty Laundry (Don Henley song))
- length_ratio=1.10, entity_recall=0.43, new_entity_count=5, number_recall=1.00, new_number_count=0, bad_patterns=["i (cannot|can't)"]
- Source: "Dirty Laundry" is a 1982 song by Don Henley and was taken from his debut studio album "I Can't Stand Still". It went to number 1 in Canada, number 2 in South Africa, number 3 in the United States, number 7 in New Zealand and number 8 in Austria.
- Output: "Dirty Laundry" is a 1982 song by Don Henley. It is on his first album, "I Can't Stand Still". The song reached number 1 in Canada. It was number 2 in South Africa. It was number 3 in the United States. It was number 7 in New Zealand. It was number 8 in Austria.

### exp_000293 (expansion, article=Zika fever)
- length_ratio=2.30, entity_recall=0.00, new_entity_count=1, number_recall=0.00, new_number_count=2, bad_patterns=[]
- Source: People who do have symptoms usually have a low fever, conjunctivitis, joint pain (mainly in the hands and feet), and a rash. The rash often starts on the face and then spreads to the rest of the body. Usually, the symptoms are not very bad, and get better after 2 to 7 days.
- Output: Individuals infected with the virus responsible for this condition typically present with a constellation of mild, non-severe symptoms that include a low-grade fever, conjunctivitis (inflammation of the membranes covering the eyes), and significant arthralgia, with joint pain predominantly affecting the hands and feet. A hallmark feature of the illness is a characteristic erythematous rash that frequently begins on the face, often sparing the palms and soles initially, before spreading rapidly to the trunk and extremities; this rash is often accompanied by a fine, sandpaper-like texture. The onset of these symptoms is usually abrupt and follows an incubation period of approximately 4 to 21 days after exposure, though the clinical course is generally self-limiting, with the majority of patients experiencing full recovery within

### exp_018717 (expansion, article=Hylodes nasus)
- length_ratio=3.25, entity_recall=0.00, new_entity_count=0, number_recall=0.00, new_number_count=2, bad_patterns=[]
- Source: The adult male frog is 25.3-28.1 mm long from nose to rear end. The skin of the frog's back can be brown or gray in color, with stripes and marks. The skin of the frog's belly is white in color.
- Output: The adult male frog exhibits a compact body structure, measuring between 25.3 and 28.1 millimeters in length from the tip of the nose to the rear end, a size range typical for many small amphibian species where males often remain slightly smaller than females to facilitate mating behaviors. The dorsal surface, or back, displays a protective coloration of brown or gray, patterned with distinct stripes and irregular marks that serve as effective camouflage against leaf litter, soil, and rocky substrates, helping the frog evade visual predators such as birds and snakes. In stark contrast, the ventral surface, or belly, is uniformly white, a common trait in amphibians that counters the silhouette of the animal when viewed from below against the bright sky, thereby reducing the likelihood of detection by ground

### exp_004252 (expansion, article=Grimsaem)
- length_ratio=2.49, entity_recall=0.25, new_entity_count=6, number_recall=0.00, new_number_count=0, bad_patterns=[]
- Source: Grimsaem Animation Co. Ltd., Seoul, Korea () was an animation studio located in South Korea. It is a division of Sunwoo Entertainment. Grimsaem creates feature-length work for Nickelodeon and Wild Brain. They also worked on the Fox TV series "Family Guy", animating the full first and third seasons of the show.
- Output: Grimsaem Animation Co., Ltd., a prominent animation studio based in Seoul, South Korea, operates as a specialized division of the larger entertainment conglomerate Sunwoo Entertainment. While the company is known for its domestic presence, it has established a significant international footprint by serving as a key production partner for major global networks, notably creating feature-length animated content for Nickelodeon and Wild Brain. Beyond its work in feature films, Grimsaem played a pivotal role in the production history of the long-running American sitcom "Family Guy" on Fox; the studio was responsible for animating the entirety of the show's first and third seasons, a period that coincided with the series' rise to critical and commercial prominence. This collaboration highlights the studio's capability to handle high-volume, complex production schedules required

### exp_007421 (expansion, article=Puzhal Lake)
- length_ratio=2.47, entity_recall=0.29, new_entity_count=6, number_recall=0.00, new_number_count=1, bad_patterns=[]
- Source: Puzhal Lake also known as Red Hills lake is located in Red Hills, Thiruvallur district, Tamil Nadu. It is one of the two reservoirs from where the drinking water is drawn for Chennai city. The other reservoir is Chembarambakkam Lake. The full capacity of the lake is 3300 million cubic feet.
- Output: Puzhal Lake, alternatively known as Red Hills Lake, is a critical artificial reservoir situated within the Red Hills area of the Thiruvallur district in the state of Tamil Nadu, India. Serving as a cornerstone of the city's water security, it functions as one of the two primary sources for Chennai's drinking water supply, working in tandem with the Chembarambakkam Lake to meet the metropolitan region's escalating demand. With a full storage capacity of approximately 3,300 million cubic feet, the lake plays an indispensable role in mitigating water scarcity issues, particularly during the dry summer months when rainfall is insufficient. Beyond its utility as a water source, the reservoir is integral to the city's flood management system, absorbing excess monsoon runoff to prevent urban flooding in low

### exp_005600 (expansion, article=John Sergeant Cram)
- length_ratio=7.27, entity_recall=0.75, new_entity_count=5, number_recall=0.00, new_number_count=5, bad_patterns=[]
- Source: John Sergeant Cram, Sr. (1852-1936) was the Head of the New York Public Service Commission.
- Output: John Sergeant Cram, Sr. (1852–1936) served as the Chairman of the New York Public Service Commission (PSC) from 1911 to 1913, a tenure that occurred during a pivotal era of Progressive Era reform in New York State. Appointed by Governor Charles Evans Hughes, Cram led the PSC, an independent regulatory agency established in 1899 to oversee the safety, rates, and operations of public utilities including electric power, gas, and railroads, which were then largely dominated by private monopolies. As Chairman, Cram worked to enforce the state's strict regulatory statutes, ensuring that utility companies provided reliable service at reasonable rates while preventing anti-competitive practices that had previously stifled public interest.

## Next experimental implication
Do a matched factor cross before any confounded leader imitation: (1) build a source-conservative SimpleWiki rewrite data arm under the protected 8×480/16k recipe, with exact word accounting and provenance; (2) separately change recipe/architecture on fixed clean-Qwen or mix25 data only if there is a representative reason to believe the previous 12×384/LAMB/40k negative came from data mismatch rather than the recipe itself.
