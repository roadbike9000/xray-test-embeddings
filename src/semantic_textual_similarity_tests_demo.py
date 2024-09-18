"""Semantic Textual Similarity
For Semantic Textual Similarity (STS), we want to produce embeddings for all 
texts involved and calculate the similarities between them. The text pairs with
the highest similarity score are most semantically similar. See also the 
Computing Embeddings documentation
https://www.sbert.net/examples/applications/computing-embeddings/README.html
for more advanced details on getting embedding scores."""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences
test1 = [
    'NAV-10113, Smart1 plate loaded into TrayZ triggers special handling for user warning and protection, Open debug console (Ctrl+Alt+D) → Enter "cmdtray FakeNextScan 0 smart1" → Press enter\nNote, the cmdtray parameters ARE case sensitive',
    'Dock basin "0"',
    'Open debug console (Ctrl+Alt+D) → Enter "cmdtray ClearFakeScan 0" ("0" represents basin) → Press Enter\nNote, the cmdtray parameters ARE case sensitive',
    'Dock basin "0"',
    'Select basin "1" → Measure Baseline → Wait until baseline will be completed',
    'Open debug console (Ctrl+Alt+D) → Enter "cmdtray FakeNextScan 0 smart1" → Press enter\nNote, the cmdtray parameters ARE case sensitive',
    'Dock basin "0"'
]

test2 = [
    'NAV-10176, Use Navigator Remote Control API to GET Platemap file and Configuration File, # Click Tools → Enable Remote Control\n# Click Tools again',
    'Open Plate Map Editor by double-clicking on a mini-map of loaded .raw file; Click on lock icon',
    'Select raw A → Fill well info → Click on an accept tick',
    'Select raw B → Fill well info → Click on an accept tick',
    'Right click on C2 → Auto Color',
    'Select D4 well → Turn it off',
    'Double click on the first column of electrodes; Download the latest Axion.RemoteControl-#####.zip from [\nhttp://jenkins.axionbio.com:808/|http://jenkins.axionbio.com:808/]\n→ unzip it; # Download .zip file which is attached to this test case\n# Unzip it\n# Replace _Axion.Files.dll_ and A_xion.RemoteControl.dll_ files to the newest from the _Axion.RemoteControl-#####.zip_; Run WpfApp.exe from test.zip and press Click Me; # Return to the Plate Map\n# Highlight all the wells\n# Click Clear Wells',
    '# Turn all the wells OFF\n# Turn all the wells ON; Click Import… → Choose exported.platemap near WpfApp.exe → Open'
]

# Compute embeddings for both lists
embeddings1 = model.encode(test1)
embeddings2 = model.encode(test2)

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for idx_i, t1 in enumerate(test1):
    print(t1)
    for idx_j, t2 in enumerate(test2):
        print(f" - {t2: <30}: {similarities[idx_i][idx_j]:.4f}")
