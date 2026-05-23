import os
import pickle
import numpy as np
import evalTask


import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi


class SimpleRAG:
    def __init__(self):
        self.docs = []               
        self.embedding_matrix = None
        self.bm25 = None
    def load_data(self, pkl_folder, txt_folder):
        self.docs = []
        for txt_file in os.listdir(txt_folder):
            if not txt_file.endswith(".txt"):
                continue
            base_name = txt_file.replace(".txt", "")
            pkl_name = f"data{base_name}.pkl"
            txt_path = os.path.join(txt_folder, txt_file)
            pkl_path = os.path.join(pkl_folder, pkl_name)
            if not os.path.exists(pkl_path):
                print(f"Missing embedding for {txt_file}")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            with open(pkl_path, "rb") as f:
                embedding = pickle.load(f)
                embedding = np.array(embedding).reshape(-1)
            self.docs.append({
                "filename": txt_file,
                "text": text,
                "embedding": embedding
            })
        if not self.docs:
            raise ValueError("No valid documents loaded.")
        self.embedding_matrix = np.vstack(
            [doc["embedding"] for doc in self.docs]
        )

        print("Loaded docs:", len(self.docs))
        print("Embedding shape:", self.embedding_matrix.shape)
        tokenized_corpus = [
            doc["text"].split()
            for doc in self.docs
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 built.")
    def cosine_similarity(self, q_embedding):
        q = q_embedding / (np.linalg.norm(q_embedding) + 1e-10)
        m = self.embedding_matrix
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
        return np.dot(m, q)
    def vector_query(self, q_embedding, top_k=5):
        q_embedding = np.array(q_embedding).reshape(-1)
        sims = self.cosine_similarity(q_embedding)
        all_idx = np.argsort(-sims)

        for rank, i in enumerate(all_idx):
            print(
                f"Rank {rank+1}: "
                f"{self.docs[i]['filename']} "
                f"score={sims[i]:.6f}"
            )
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {
                "idx": int(i),
                "filename": self.docs[i]["filename"],
                "text": self.docs[i]["text"],
                "vector_score": float(sims[i])
            }
            for i in top_idx
        ]
    def bm25_query(self, query_text, top_k=5):
        scores = self.bm25.get_scores(query_text.split())
        all_idx = np.argsort(-scores)

        for rank, i in enumerate(all_idx):
            print(
                f"Rank {rank+1}: "
                f"{self.docs[i]['filename']} "
                f"score={scores[i]:.6f}"
            )
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {
                "idx": int(i),
                "filename": self.docs[i]["filename"],
                "text": self.docs[i]["text"],
                "bm25_score": float(scores[i])
            }
            for i in top_idx
        ]
    def query_dual(self, query_text, q_embedding,
                   bm25_k=5, vector_k=5):
        alpha = 0.5
        hybrid_k = 5
        vector_results = self.vector_query(q_embedding, vector_k)
        bm25_results = self.bm25_query(query_text, bm25_k)
        q_embedding = np.array(q_embedding).reshape(-1)
        sims = self.cosine_similarity(q_embedding)
        bm25_scores = self.bm25.get_scores(query_text.split())
        vec_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-10)
        bm_norm = (bm25_scores - bm25_scores.min()) / (
            bm25_scores.max() - bm25_scores.min() + 1e-10
        )
        hybrid_scores = alpha * vec_norm + (1 - alpha) * bm_norm

        #print all scores for debugging after ranking
        all_idx = np.argsort(-hybrid_scores)
        for rank, i in enumerate(all_idx):
            print(
                f"Rank {rank+1}: "
                f"{self.docs[i]['filename']} "
                f"vector_score={sims[i]:.6f} "
                f"bm25_score={bm25_scores[i]:.6f} "
                f"hybrid_score={hybrid_scores[i]:.6f}"
            )


        hybrid_sorted_idx = np.argsort(-hybrid_scores)[:hybrid_k]
        hybrid_results = [
            {
                "idx": int(i),
                "filename": self.docs[i]["filename"],
                "text": self.docs[i]["text"],
                "vector_score": float(sims[i]),
                "bm25_score": float(bm25_scores[i]),
            }
            for i in hybrid_sorted_idx
        ]
        combined = {}
        def add_items(items):
            for item in items:
                idx = item["idx"]
                if idx not in combined:
                    combined[idx] = {
                        "filename": item["filename"],
                        "text": item["text"],
                        # "vector_score": item["vector_score"],
                        # "bm25_score": item["bm25_score"],
                    }

        add_items(vector_results)
        add_items(bm25_results)
        add_items(hybrid_results)
        print(f"Combined unique results: {len(combined)}")
        print("Combined results:")
        for idx, item in combined.items():
            print(f"Filename: {item['filename']}")
            # print(f"Text: {item['text']}")
            # print(f"Vector Score: {item.get('vector_score', 'N/A')}")
            # print(f"BM25 Score: {item.get('bm25_score', 'N/A')}")
            print("-----")
        return list(combined.values())


from openai import OpenAI

from openai import OpenAI

client = OpenAI(
    api_key=""
)
import re


def extract_code(text):
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]


def embedding(myText):
    response = client.embeddings.create(input=myText, model="text-embedding-3-large")

    # print(response.data[0].embedding)
    return response.data[0].embedding


def askLLM(prompt, mymodel):

    if mymodel == "gpt-4o" or mymodel == "gpt-5":
        client = OpenAI(
            api_key=""
        )
    else:
        client = OpenAI(
            api_key="",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    response = client.chat.completions.create(
        model=mymodel,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in analog circuit design.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def getNetlist(id):
    answer = ""
    i = id
    with open(
        f"",
        "r",
    ) as f:
        netlist = f.read()
    answer += f"Netlist id: {i}\n"
    answer += netlist + "\n\n"
    return answer


def getCandidates(query_text, spec, queryID,knowledge_docs):
    rag = SimpleRAG()
    rag.load_data("",
                  "")
    second_query = f"""
Topology description: 
{query_text}
Please write a prompt to search for this topology design in a RAG system and only output the topology description prompt text directly without any additional explanation.
Sample output: 
The circuit is a two-stage operational amplifier. The first stage consists of an NMOS differential pair with PMOS diode-connected loads, while the second stage is a common-source amplifier with a self-cascoded PMOS driver and an NMOS current-source load to achieve high gain.
"""
    print("query_text")
    print(query_text)
    answer = askLLM(second_query, "gemini-2.5-flash")
    print("Second query:")
    print(second_query)
    query_embedding = embedding(answer)
    # results = rag.query(query_embedding)
    results = rag.query_dual(answer, query_embedding, bm25_k=5, vector_k=5)
    mynewcandidate = ""
    for res in results:
        text = res["filename"]
        # similarity = res["similarity"]
        print(text)
        id = int(text.split("_")[0])
        mynewcandidate += getNetlist(id)
    third_query = f"""
I have the spec requirements of an op-amp (under skywater PDK, SKY130 process node) as follows:
{spec}
And I have a topology design description that could meet the spec requirements as follows:
{query_text}
I also have some candidate netlists that may meet the spec requirements as follows:
{mynewcandidate}
There are some retrievaled knowledge.
{knowledge_docs}
Please help me to select the best candidate netlist that meets the spec requirements.
First compare each candidate with the topology design description and spec requirement and eliminate those that do not match the topology design and finally output the topology candidate netlist id directly sorted based on the possibility that they meet the spec requirements.
Example:
```python
mycandidate = [35,17] # 35 is the best candidate, 17 is the second best candidate, etc. Give the best 5 candidates.
```
Please do not modify the variable name "mycandidate" and you MUST put it in a python code block like the example.
"""
    print("Third query:")
    print(third_query)
    answer = askLLM(third_query, "gemini-2.5-flash")
    print("Final candidate selection result:")
    print(answer)
    env = {}
    try:
        candidates = extract_code(answer)

        exec(candidates[0], env)
        totalcandidate = env["mycandidate"]
        # print("candidates:")
        # print(totalcandidate)
        # print(mycandidate[0])
        # exec(mycandi)
        eval_result = evalTask.evalTask(queryID, totalcandidate)
    except Exception as e:

        print(f"Error: {e}")
        totalcandidate = []
        eval_result = 0
    print(f"Eval: {eval_result}")
    return (mynewcandidate, totalcandidate)
if __name__ == "__main__":
    spec="""

- The power consumption of the amplifier should be no larger than 10 W.
- The DC gain of the amplifier should be at least 40 dB.
- The common-mode rejection ratio (CMRR) at DC should be at least 70 dB.
- The power supply rejection ratio (PSRR) should be at least 65 dB.
- The gain-bandwidth product (GBW) should be at least 3000000.0 Hz.
- The phase margin should be at least 60 degrees.
- The power supply noise rejection (PSRN) should be at least 80 dB.
"""
    getCandidates("The **Two-Stage Fully Differential Miller-Compensated Op-Amp with Gain-Boosted Folded Cascode First Stage** is a robust and flexible topology. It addresses the low intrinsic gain and headroom limitations of the SKY130 process through the use of gain boosting and a folded cascode structure. It leverages well-established techniques for stability (Miller compensation with nulling resistor) and common-mode control (CMFB). The combination of high output impedance from gain-boosting and differential operation ensures the stringent CMRR, PSRR, and PSRN requirements are met. The 3 MHz GBW is comfortably achievable with this structure, allowing for optimization towards low power within the generous 10W budget.", spec, 11)
