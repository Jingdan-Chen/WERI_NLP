def plsa_analyze():
    import os
    import json
    curr_dir = "./"
    plsa_path = curr_dir + "strategy/plsa/"
    for file in [curr_dir+ "result_dict.json"]:
        f = open(file, "w")
        f.close()

    # os.system("python "+plsa_path+"plsa_ana2.py")
    from .plsa_ana2 import run_plsa_ana2
    run_plsa_ana2()
    with open(curr_dir + 'result_dict.json') as f:
        data = json.load(f)
    #     temp=f.read()
    #     print(temp)
    #     data = eval()
    res_lis = [True if len(data[key][0]) > 0 else False for key in data.keys()]

    return res_lis
