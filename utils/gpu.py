import os
import torch


def get_gpu_prop():
    ngpus = torch.cuda.device_count()

    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            "total_momory": round(prop.total_memory / 1073741824, 2),  # unit GB
            "sm_count": prop.multi_processor_count
        })

    print("cuda: {}".format(torch.cuda.is_available()))
    print("available GPU(s): {}".format(ngpus))
    for i, p in enumerate(properties):
        print("{}: {}".format(i, p))

    return properties