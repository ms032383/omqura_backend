import os
import json
import re
import shutil
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# 1. Custom Vocabulary-Based Embedding Function for 100% Offline Reliability
class SimpleVocabularyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Key terms representing dimensions in vector space, including basic anatomy & clinical oncology
        self.vocabulary = [
            "chordoma", "resection", "clival", "recurrence", "schwannoma", "stereotactic", "radiosurgery",
            "microsurgical", "craniotomy", "gliomas", "motor-evoked", "potential", "amplitude", "decline",
            "antithrombotic", "guidelines", "antiplatelet", "pituitary", "adenomas", "cerebrospinal", "fluid",
            "leak", "glioblastoma", "bevacizumab", "ependymoma", "posterior", "fossa", "decompressive", "craniectomy",
            "infarction", "hydrocephalus", "shunt", "trigeminal", "neuralgia", "microvascular", "decompression",
            "aneurysm", "clipping", "subarachnoid", "hemorrhage", "subdural", "hematoma", "drainage", "meningioma",
            "radiotherapy", "spinal", "cord", "moyamoya", "revascularization", "cavernous", "malformations",
            "mortality", "hypertension", "craniosynostosis", "transfusion", "pseudarthrosis", "cervical", "fusion",
            "neuron", "synapse", "cerebrum", "cerebellum", "brainstem", "cortex", "axon", "dendrite", "neuroanatomy",
            "brain", "nervous", "spinal"
        ]

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            text_lower = text.lower()
            vector = []
            for word in self.vocabulary:
                # Count frequency of key term
                count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
                vector.append(float(count))
            # L2 normalization
            norm = sum(x**2 for x in vector)**0.5
            if norm > 0:
                vector = [x/norm for x in vector]
            else:
                vector = [0.0] * len(self.vocabulary)
            embeddings.append(vector)
        return embeddings

# 2. Layer 1: Basic Neuroanatomy Chunks
ANATOMY_CHUNKS = [
    {
        "chunk_id": "chunk_A001",
        "paper_id": "ANAT001",
        "doi": "10.1016/neuroanatomy.2021.01",
        "url": "https://www.anatomy-database.org/neuron",
        "journal": "Basic Neuroanatomy Reference",
        "year": 2021,
        "authors": "Standring S, et al.",
        "section": "Cytology of Nervous System",
        "layer": 1,
        "raw_text": "A neuron or nerve cell is an electrically excitable cell that communicates with other cells via specialized connections called synapses. It is the primary structural and functional unit of the nervous system."
    },
    {
        "chunk_id": "chunk_A002",
        "paper_id": "ANAT002",
        "doi": "10.1016/neuroanatomy.2021.02",
        "url": "https://www.anatomy-database.org/synapse",
        "journal": "Basic Neuroanatomy Reference",
        "year": 2021,
        "authors": "Standring S, et al.",
        "section": "Synaptic Transmission",
        "layer": 1,
        "raw_text": "A synapse is a highly specialized junction where transmission of information occurs between a neuron and a target cell (either another neuron, muscle fiber, or gland) using electrical or chemical signals."
    },
    {
        "chunk_id": "chunk_A003",
        "paper_id": "ANAT003",
        "doi": "10.1016/neuroanatomy.2021.03",
        "url": "https://www.anatomy-database.org/cerebrum",
        "journal": "Basic Neuroanatomy Reference",
        "year": 2021,
        "authors": "Standring S, et al.",
        "section": "Forebrain Structures",
        "layer": 1,
        "raw_text": "The cerebrum is the largest part of the brain, composed of left and right hemispheres. It governs high-level functions including sensory processing, voluntary motor control, reasoning, language, and memory storage."
    },
    {
        "chunk_id": "chunk_A004",
        "paper_id": "ANAT004",
        "doi": "10.1016/neuroanatomy.2021.04",
        "url": "https://www.anatomy-database.org/cerebellum",
        "journal": "Basic Neuroanatomy Reference",
        "year": 2021,
        "authors": "Standring S, et al.",
        "section": "Hindbrain Structures",
        "layer": 1,
        "raw_text": "The cerebellum, located under the occipital lobes of the cerebrum, regulates motor coordination, precision, timing, and balance. It receives sensory inputs and fine-tunes motor movements."
    },
    {
        "chunk_id": "chunk_A005",
        "paper_id": "ANAT005",
        "doi": "10.1016/neuroanatomy.2021.05",
        "url": "https://www.anatomy-database.org/brainstem",
        "journal": "Basic Neuroanatomy Reference",
        "year": 2021,
        "authors": "Standring S, et al.",
        "section": "Brainstem Anatomy",
        "layer": 1,
        "raw_text": "The brainstem connects the cerebrum and cerebellum to the spinal cord. It consists of the midbrain, pons, and medulla oblongata, regulating autonomic functions like respiration, heartbeat, and blood pressure."
    }
]

# 3. Layer 2 & 3: Clinical & Neuro-Oncology Chunks
MOCK_CHUNKS = [
    {
        "chunk_id": "chunk_Q001",
        "paper_id": "PMC5829102",
        "doi": "10.1007/s00586-018-5512-y",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5829102/",
        "journal": "European Spine Journal",
        "year": 2018,
        "authors": "M. Al-Mefty, et al.",
        "section": "Discussion",
        "layer": 3,
        "raw_text": "For clival chordoma, achieving wide margin resection (greater than 1-2 mm margin) is difficult due to critical surrounding structures. Gross total resection with margins reduces recurrence rates to 25% at 5 years, compared to 65% for subtotal resection. Local recurrence in the clival region remains the primary failure mode."
    },
    {
        "chunk_id": "chunk_Q002",
        "paper_id": "PMC6120987",
        "doi": "10.3171/2018.4.JNS172651",
        "url": "https://thejns.org/view/journals/j-neurosurg/129/3/article-p612.xml",
        "journal": "Journal of Neurosurgery",
        "year": 2018,
        "authors": "Lunsford LD, et al.",
        "section": "Results",
        "layer": 3,
        "raw_text": "In patients with vestibular schwannoma, the 5-year progression-free survival (PFS) rate was 92% for stereotactic radiosurgery (SRS) and 96% for microsurgical resection. Although microsurgery showed a slightly higher PFS, SRS was associated with a much lower rate of facial nerve dysfunction (2% vs 15%)."
    },
    {
        "chunk_id": "chunk_Q003",
        "paper_id": "PMC7021143",
        "doi": "10.1093/neuros/nyz412",
        "url": "https://academic.oup.com/neurosurgery/article/86/2/234/5678912",
        "journal": "Neurosurgery",
        "year": 2020,
        "authors": "Duffau H, et al.",
        "section": "Intraoperative Monitoring Protocols",
        "layer": 3,
        "raw_text": "During awake craniotomy for eloquent-area gliomas, intraoperative motor-evoked potential (MEP) monitoring is critical. A threshold decline of greater than 50% in MEP amplitude mandates immediate intervention, such as stopping resection, irrigating with warm saline, or elevating mean arterial pressure, to prevent permanent motor deficits."
    },
    {
        "chunk_id": "chunk_Q004",
        "paper_id": "PMC6549210",
        "doi": "10.1227/NEU.0000000000002135",
        "url": "https://journals.lww.com/neurosurgery/fulltext/2021/04000/antithrombotic_management_guidelines.aspx",
        "journal": "Neurosurgery Guidelines",
        "year": 2021,
        "authors": "AANS/CNS Joint Committee",
        "section": "Guidelines Summary",
        "layer": 2,
        "raw_text": "For patients undergoing elective craniotomy who are on dual antiplatelet therapy (DAPT), the current AANS/CNS guidelines recommend that Aspirin should be continued, and Clopidogrel should be discontinued 5-7 days prior to surgery to minimize intracranial hemorrhage risk."
    },
    {
        "chunk_id": "chunk_Q005",
        "paper_id": "PMC6802145",
        "doi": "10.1055/s-0039-1698204",
        "url": "https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-0039-1698204",
        "journal": "Journal of Neurological Surgery Part B",
        "year": 2019,
        "authors": "Hadad G, et al.",
        "section": "Results",
        "layer": 2,
        "raw_text": "The incidence of cerebrospinal fluid (CSF) leak following endoscopic endonasal approach (EEA) to pituitary adenomas is approximately 5% overall. The use of a vascularized nasoseptal flap (Hadad-Bassagasteguy flap) is an evidence-based repair strategy that reduces the post-operative leak rate to less than 2%."
    },
    {
        "chunk_id": "chunk_Q006",
        "paper_id": "PMC5421098",
        "doi": "10.1200/JCO.2017.74.3210",
        "url": "https://ascopubs.org/doi/10.1200/JCO.2017.74.3210",
        "journal": "Journal of Clinical Oncology",
        "year": 2017,
        "authors": "Friedman HS, et al.",
        "section": "Oncologic Outcomes",
        "layer": 3,
        "raw_text": "For recurrent glioblastoma, the median overall survival (OS) benefit of Bevacizumab monotherapy is 9.2 months compared to 6.0 months for best supportive care in the post-radiation setting, representing a net survival benefit of 3.2 months."
    },
    {
        "chunk_id": "chunk_Q007",
        "paper_id": "PMC7812984",
        "doi": "10.1007/s11060-020-03612-z",
        "url": "https://link.springer.com/article/10.1007/s11060-020-03612-z",
        "journal": "Journal of Neuro-Oncology",
        "year": 2020,
        "authors": "Merchant TE, et al.",
        "section": "Radiographic Definitions",
        "layer": 3,
        "raw_text": "In pediatric posterior fossa ependymoma, gross total resection (GTR) is defined radiographically as no visible tumor on post-operative MRI within 48 hours. Near-total resection (NTR) is defined as a residual tumor volume of less than 1.5 cm^3."
    },
    {
        "chunk_id": "chunk_Q008",
        "paper_id": "PMC5120932",
        "doi": "10.1161/STROKEAHA.115.012345",
        "url": "https://www.ahajournals.org/doi/10.1161/STROKEAHA.115.012345",
        "journal": "Stroke",
        "year": 2016,
        "authors": "Jüttler E, et al.",
        "section": "Clinical Trial Discussion",
        "layer": 2,
        "raw_text": "In decompressive craniectomy for malignant middle cerebral artery (MCA) infarction, clinical trials recommend a time window of within 48 hours from stroke onset to surgery for optimal functional outcome (defined as modified Rankin Scale mRS <= 4)."
    },
    {
        "chunk_id": "chunk_Q009",
        "paper_id": "PMC6450212",
        "doi": "10.3171/2019.2.PEDS18342",
        "url": "https://thejns.org/view/journals/j-neurosurg-pediatr/23/5/article-p543.xml",
        "journal": "Journal of Neurosurgery: Pediatrics",
        "year": 2019,
        "authors": "Riva-Cambrin J, et al.",
        "section": "Complications Analysis",
        "layer": 2,
        "raw_text": "The rate of postoperative hydrocephalus requiring ventriculoperitoneal (VP) shunt placement after posterior fossa tumor resection in children under 5 years of age is approximately 30%."
    },
    {
        "chunk_id": "chunk_Q010",
        "paper_id": "PMC5621098",
        "doi": "10.3171/2017.3.JNS162012",
        "url": "https://thejns.org/view/journals/j-neurosurg/127/4/article-p820.xml",
        "journal": "Journal of Neurosurgery",
        "year": 2017,
        "authors": "Barker FG, et al.",
        "section": "Long-Term Outcomes",
        "layer": 2,
        "raw_text": "For patients with trigeminal neuralgia refractory to medical management, microvascular decompression (MVD) provides a long-term pain-free rate of approximately 70% at 10 years, making it the most durable surgical treatment option."
    },
    {
        "chunk_id": "chunk_Q011",
        "paper_id": "PMC6902143",
        "doi": "10.1227/NEU.0000000000002341",
        "url": "https://journals.lww.com/neurosurgery/fulltext/2020/09000/intraoperative_hemodynamic_targets.aspx",
        "journal": "Neurosurgery",
        "year": 2020,
        "authors": "Connolly ES, et al.",
        "section": "Intraoperative Management Guidelines",
        "layer": 3,
        "raw_text": "During aneurysm clipping in patients with subarachnoid hemorrhage (SAH), intraoperative targets for maintaining cerebral perfusion pressure (CPP) recommend keeping CPP between 60-80 mmHg, avoiding mean arterial pressure (MAP) drops below 80 mmHg during temporary occlusion."
    },
    {
        "chunk_id": "chunk_Q012",
        "paper_id": "PMC4120934",
        "doi": "10.1016/S0140-6736(09)61612-4",
        "url": "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(09)61612-4/fulltext",
        "journal": "The Lancet",
        "year": 2009,
        "authors": "Santarius T, et al.",
        "section": "Results",
        "layer": 2,
        "raw_text": "In the management of chronic subdural hematoma, randomized trials demonstrate that the recurrence rate is 9% following burr-hole drainage with a subdural drain placed, compared to 24% for burr-hole drainage alone without a drain."
    },
    {
        "chunk_id": "chunk_Q013",
        "paper_id": "PMC7432109",
        "doi": "10.1016/j.ijrobp.2020.04.015",
        "url": "https://www.redjournal.org/article/S0360-3016(20)31024-X/fulltext",
        "journal": "International Journal of Radiation Oncology * Biology * Physics",
        "year": 2020,
        "authors": "Rogers L, et al.",
        "section": "Adjuvant Radiotherapy Guidelines",
        "layer": 3,
        "raw_text": "For atypical meningioma (WHO Grade II) following Simpson Grade I resection, the recommended dose and fractionation schedule for adjuvant radiotherapy is 54-60 Gy in 1.8-2.0 Gy fractions delivered over a 6-week period."
    },
    {
        "chunk_id": "chunk_Q014",
        "paper_id": "PMC6201934",
        "doi": "10.3171/2018.8.SPINE18210",
        "url": "https://thejns.org/view/journals/j-neurosurg-spine/29/6/article-p690.xml",
        "journal": "Journal of Neurosurgery: Spine",
        "year": 2018,
        "authors": "McCormick PC, et al.",
        "section": "Complications Discussion",
        "layer": 2,
        "raw_text": "For intramedullary spinal cord tumors, the rate of immediate postoperative neurological deterioration following gross total resection (GTR) in the cervical region is approximately 15%, with 10% having long-term deficits at 1 year."
    },
    {
        "chunk_id": "chunk_Q015",
        "paper_id": "PMC6982041",
        "doi": "10.3171/2019.9.JNS191240",
        "url": "https://thejns.org/view/journals/j-neurosurg/132/2/article-p420.xml",
        "journal": "Journal of Neurosurgery",
        "year": 2020,
        "authors": "Steinberg GK, et al.",
        "section": "Surgical Decision-Making",
        "layer": 2,
        "raw_text": "For patients with moyamoya disease, direct revascularization (e.g. STA-MCA bypass) is indicated in adults with patent, appropriately sized donor vessels, while indirect revascularization (e.g. EDAS or EMS) is preferred in pediatric patients or when donor/recipient vessels are hypoplastic."
    },
    {
        "chunk_id": "chunk_Q016",
        "paper_id": "PMC5821098",
        "doi": "10.1227/NEU.0000000000001845",
        "url": "https://journals.lww.com/neurosurgery/fulltext/2018/06000/brainstem_cavernous_malformations.aspx",
        "journal": "Neurosurgery",
        "year": 2018,
        "authors": "Spetzler RF, et al.",
        "section": "Vascular Neurosurgery Review",
        "layer": 3,
        "raw_text": "Brainstem cavernous malformations exhibit an annual hemorrhage risk of 2.7% initially, which escalates to 15% after a second bleeding episode. Surgical approaches include the suboccipital telovelar approach for lesions on the floor of the 4th ventricle, and the supracerebellar infratentorial approach for dorsal midbrain cavernomas."
    },
    {
        "chunk_id": "chunk_Q017",
        "paper_id": "PMC6340212",
        "doi": "10.3171/2018.1.JNS172401",
        "url": "https://thejns.org/view/journals/j-neurosurg/129/5/article-p1140.xml",
        "journal": "Journal of Neurosurgery",
        "year": 2018,
        "authors": "Lawton MT, et al.",
        "section": "Vascular Outcomes",
        "layer": 3,
        "raw_text": "For patients over 70 years of age, the 30-day mortality rate for elective clipping of unruptured anterior circulation aneurysms is approximately 1.5%, which is comparable to endovascular coiling outcomes in this age demographic."
    },
    {
        "chunk_id": "chunk_Q018",
        "paper_id": "PMC7012934",
        "doi": "10.1016/j.ophtha.2019.08.012",
        "url": "https://www.aaojournal.org/article/S0161-6420(19)32104-2/fulltext",
        "journal": "Ophthalmology",
        "year": 2020,
        "authors": "Wall M, et al.",
        "section": "Functional Outcomes",
        "layer": 2,
        "raw_text": "In patients with idiopathic intracranial hypertension (IIH) refractory to medical therapy, ventriculoperitoneal (VP) shunting has a success rate of 85% in preserving or improving visual function at 2 years."
    },
    {
        "chunk_id": "chunk_Q019",
        "paper_id": "PMC6812043",
        "doi": "10.1097/SCS.0000000000005230",
        "url": "https://journals.lww.com/jcraniofacialsurgery/fulltext/2019/07000/craniosynostosis_blood_transfusion_requirements.aspx",
        "journal": "Journal of Craniofacial Surgery",
        "year": 2019,
        "authors": "Fearon JA, et al.",
        "section": "Pediatric Surgery Results",
        "layer": 2,
        "raw_text": "For infants under 12 months undergoing total calvarial vault remodeling for craniosynostosis, the blood transfusion requirement is approximately 100%, with a median transfusion volume of 45 mL/kg required intraoperatively."
    },
    {
        "chunk_id": "chunk_Q020",
        "paper_id": "PMC6432109",
        "doi": "10.3171/2018.12.SPINE181045",
        "url": "https://thejns.org/view/journals/j-neurosurg-spine/30/4/article-p450.xml",
        "journal": "Journal of Neurosurgery: Spine",
        "year": 2019,
        "authors": "Benizel EC, et al.",
        "section": "Spinal Surgery Results",
        "layer": 2,
        "raw_text": "In posterior cervical fusion using lateral mass screws and rods for multilevel degenerative disease, the reported pseudarthrosis rate is approximately 5%, reflecting high construct stability."
    }
]

def main():
    print("Starting clinical knowledge chunk generation...")
    chunks_dir = "data/chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    # Scan and load custom documents
    custom_anatomy_dir = "data/documents/anatomy"
    custom_clinical_dir = "data/documents/clinical"
    os.makedirs(custom_anatomy_dir, exist_ok=True)
    os.makedirs(custom_clinical_dir, exist_ok=True)

    loaded_anatomy_chunks = list(ANATOMY_CHUNKS)
    loaded_clinical_chunks = list(MOCK_CHUNKS)

    # Ingest custom anatomy files
    custom_anat_count = 0
    if os.path.exists(custom_anatomy_dir):
        for filename in os.listdir(custom_anatomy_dir):
            if filename.endswith(".txt") or filename.endswith(".md"):
                file_path = os.path.join(custom_anatomy_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for idx, p in enumerate(paragraphs, 1):
                        custom_anat_count += 1
                        loaded_anatomy_chunks.append({
                            "chunk_id": f"custom_anat_{filename.replace('.', '_')}_{idx}",
                            "paper_id": "CUSTOM_ANAT",
                            "doi": "N/A",
                            "url": "https://www.anatomy-database.org/custom",
                            "journal": "User Uploaded Anatomy Reference",
                            "year": 2026,
                            "authors": "Custom Knowledge Base",
                            "section": "General Anatomy",
                            "layer": 1,
                            "raw_text": p
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    if custom_anat_count > 0:
        print(f"Loaded {custom_anat_count} custom anatomy chunks from {custom_anatomy_dir}")

    # Ingest custom clinical files
    custom_clin_count = 0
    if os.path.exists(custom_clinical_dir):
        for filename in os.listdir(custom_clinical_dir):
            if filename.endswith(".txt") or filename.endswith(".md"):
                file_path = os.path.join(custom_clinical_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for idx, p in enumerate(paragraphs, 1):
                        custom_clin_count += 1
                        loaded_clinical_chunks.append({
                            "chunk_id": f"custom_clin_{filename.replace('.', '_')}_{idx}",
                            "paper_id": "CUSTOM_CLIN",
                            "doi": "N/A",
                            "url": "https://www.ncbi.nlm.nih.gov/pmc/custom",
                            "journal": "User Uploaded Clinical Literature",
                            "year": 2026,
                            "authors": "Custom Clinical Base",
                            "section": "Clinical Studies",
                            "layer": 2,
                            "raw_text": p
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    if custom_clin_count > 0:
        print(f"Loaded {custom_clin_count} custom clinical chunks from {custom_clinical_dir}")

    # Save individual JSON files to data/chunks/
    for chunk in loaded_anatomy_chunks + loaded_clinical_chunks:
        file_path = os.path.join(chunks_dir, f"{chunk['chunk_id']}.json")
        with open(file_path, "w") as f:
            json.dump(chunk, f, indent=2)
        print(f"Saved: {file_path}")

    print("Initializing Chroma DB collections...")
    chroma_path = "data/chroma_db"
    
    # Clear old database if exists to reseed cleanly
    if os.path.exists(chroma_path):
        print(f"Removing existing database at {chroma_path} to reseed cleanly.")
        shutil.rmtree(chroma_path)
        
    client = chromadb.PersistentClient(path=chroma_path)
    embedding_fn = SimpleVocabularyEmbeddingFunction()
    
    # 1. Collection for Layer 1 Basic Neuroanatomy
    collection_anatomy = client.create_collection(
        name="anatomy_chunks",
        embedding_function=embedding_fn
    )
    
    # 2. Collection for Layer 2 & 3 Clinical Guidelines/Oncology
    collection_clinical = client.create_collection(
        name="neurosurgery_chunks",
        embedding_function=embedding_fn
    )

    # Ingest Layer 1
    anatomy_ids = [c["chunk_id"] for c in loaded_anatomy_chunks]
    anatomy_docs = [c["raw_text"] for c in loaded_anatomy_chunks]
    anatomy_metas = []
    for c in loaded_anatomy_chunks:
        meta = c.copy()
        meta.pop("raw_text")
        anatomy_metas.append(meta)
    
    collection_anatomy.add(
        ids=anatomy_ids,
        documents=anatomy_docs,
        metadatas=anatomy_metas
    )
    print(f"Ingested {len(loaded_anatomy_chunks)} Layer 1 anatomy chunks.")

    # Ingest Layer 2 & 3
    clinical_ids = [c["chunk_id"] for c in loaded_clinical_chunks]
    clinical_docs = [c["raw_text"] for c in loaded_clinical_chunks]
    clinical_metas = []
    for c in loaded_clinical_chunks:
        meta = c.copy()
        meta.pop("raw_text")
        clinical_metas.append(meta)
        
    collection_clinical.add(
        ids=clinical_ids,
        documents=clinical_docs,
        metadatas=clinical_metas
    )
    print(f"Ingested {len(loaded_clinical_chunks)} Layer 2 & 3 clinical chunks.")
    
    print("Database seeding completed successfully!")

if __name__ == "__main__":
    main()
