# New version: sizing_mix.py

from openai import OpenAI
from datetime import datetime

def askLLM(prompt, mymodel):

    if mymodel == "gpt-4o" or mymodel == "gpt-5":
        client = OpenAI(
            api_key="abcd"
        )
    else:
        client = OpenAI(
            api_key="abcd",
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




def getMySizing(queryId,topology,topologyId):
    import time
    currentTime = time.time()
    logFilePath = f"logqueryid{queryId}_topologyId{topologyId}_{currentTime}.log"
    log = open(logFilePath, "w")
    with open(f"", "r") as f:
        spec = f.read()
    iteration = 0
    current_best = ""
    history_design = []
    MAX_ITERATION = 3
    iter = ""
    while iteration <=MAX_ITERATION:
        iteration += 1
        if history_design:
            iter += "In the previous iterations, I have the following sizing design and the results."
            for i, h in enumerate(history_design):
                iter += f"\n Design {i+1}: {h}"
            iter += "\n Now please try to size the topology again. \n"
        prompt=iter + f"""
Please design an operational amplifier using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
Now I have the following topology in netlist format:
{topology}
In this step, you would need to size the topology.

There is a knowledge agent that can help to find relevant knowledge documents about analog circuit design from a knowledge base.
You can write a query to ask the knowledge agent to get relevant knowledge documents to help you design the op-amp.
To ask the agent, please first analyze the specifications and then output the query text in a code block. 
You do not need to do detail sizing now, just need to find relevant knowledge documents.

Example
```text
How to size a two-stage operational amplifier with Miller compensation?
```
"""
        print("First query:")
        log.write("First query:\n" + prompt + "\n")
        print(prompt)
        first_query = askLLM(prompt, "gemini-2.5-flash")
        print("First query result:")
        print(first_query)
        log.write("First query result:\n" + first_query + "\n")

        import knowledge
        query_text_list = knowledge.extract_code(first_query)
        print("Extracted query text list:")
        print(query_text_list)
        log.write("Extracted query text list:\n" + str(query_text_list) + "\n")
        knowledge_results = knowledge.getKnowledge(query_text_list, mytop_k=5)
        prompt = f"""
Please design an operational amplifier using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
Now I have the following topology in netlist format:
{topology}
In this step, you would need to size the topology.

The following relevant knowledge documents are retrieved to help to design the op-amp:
{knowledge_results}

The maximum possible range for each parameter is given below:
For L, from 0.1 to 1. 
For W, from 0.3 to 10. 
For M, from 1 to 100. 
For capacitor, from 1e-12 to 1e-10. 
For resistor, from 1 to 1000000.0. 
For bias voltage, from 0V to 1.8V. 
For current source current, from 2e-6 to 2e-5.

There is a sizing agent that can find the simulation result in a range of sizing.
You can write a python code to ask the sizing agent to do simulation in a given range.
To ask the agent, please write a python code to list all parameters used in the netlist and their range.
Every parameter should have a lower bound and an upper bound.

For example:
```
myparams = {{
    'L_M0': (0.1, 1),
    'W_M0': (0.3, 10),
    'M_M0': (1, 100),
}}
```
- Never change the format of the example.
- Never change the variable names in the netlist.
- The desired search space size should be at least 8/10 and at most 9/10 of the original full search space.
- Never give too tight ranges, especially when the previous designs are wrong.
- If the previous designs are wrong, please give a *larger range* to avoid missing the feasible sizing. Do *NOT repeat* the previous wrong sizing.

1. First analyze the specifications, topology and knowledge base
2. Then analyze the necessary parameters in general description.
3. Next, you need to analyze the given sizing range and choosing a proper range for each parameter. DO NOT give too tight ranges because it is easy to miss the feasible sizing.
4. Finally, decide the parameter ranges and then write the code.

DO NOT DIRECTLY GIVE A NEW SIZING RANGE. Follow the above steps to give the sizing range.
"""
        history = prompt
        print("Second query:")
        print(prompt)
        log.write("Second query:\n" + prompt + "\n")
        second_query = askLLM(prompt, "gemini-2.5-flash")
        print("Second query result:")
        print(second_query)
        log.write("Second query result:\n" + second_query + "\n")
        history += "\n" + second_query

        print("Extracted sizing code:")
        sizing_code = knowledge.extract_code(second_query)
        print(sizing_code)
        log.write("Extracted sizing code:\n" + str(sizing_code) + "\n")
        print("Sizing code:")
        print(sizing_code[0])
        ns = {}
        exec(sizing_code[0], {}, ns)  
        myparams = ns['myparams']
        import simulation
        answer = simulation.doSimulate(myparams, queryId,topologyId)
        log.write("Simulation result:\n" + str(answer) + "\n")
        if answer[1][0] == "true":
            print("Found a feasible sizing:")
            print(answer[1][1])
            return answer

        history += "\n" + str(answer[1])
        import compress
        print("History before compression:")
        print(history)
        log.write("History before compression:\n" + history + "\n")
        observation = compress.compress_feedback("\n".join(history),"\n In the next iteration, we will do sizing again. Please keep the useful information that can help to do sizing again. Do not include the useless information that is not related to sizing. Do not directly give a new sizing range here.\n")
        print("Feedback from compression:")
        print(observation)
        log.write("Feedback from compression:\n" + observation + "\n")
        history_design.append(str(answer[1]) + "\n observation:" + observation)
    return answer


if __name__ == "__main__":
    evalDesigns = [2,27,27,12,12,27,27,35,16,38]
    now_str = datetime.now().strftime("%m%d_%H%M%S")
    
    result_test_path = f"testResult__{now_str}.txt"
    result_file = open(result_test_path, "w")
    for taskId in [1,2,3,4,5,6,7,8,9,10]:
        for tests in [1,2,3]:
            try:
                result_file.write(f"Task {taskId}, Test {tests}\n")
                result_file.flush()
                myDesignId = evalDesigns[taskId-1]
                with open(f"", "r") as f:
                    topology = f.read()
                ans = getMySizing(taskId, topology,topologyId=myDesignId)
                result_file.write(f"Sizing result: {ans}\n")
                result_file.flush()
            except Exception as e:
                result_file.write(f"Task {taskId}, Test {tests} failed with exception: {e}\n")
                result_file.flush()
