import json


def get_relation_set(data):
    relations = set()
    for i in data:
        _, r, _ = i['instance']
        relations.add(r)
    return relations


def sub_list(a, b):
    '''
    a - a^b
    '''
    ince = set(a).intersection(set(b))
    # print(ince)
    a = list(a)
    for i in ince:
        a.remove(i)
    return a


def get_unseen_relation():
    train_path = "../data/DB100K-zero-shot/train_id.json"
    test_path = "../data/DB100K-zero-shot/test_id.json"
    with open(train_path) as f:
        train_data = json.load(f)
        train_set = get_relation_set(train_data)
    with open(test_path) as f:
        test_data = json.load(f)
        test_set = get_relation_set(test_data)
    return sub_list(test_set, train_set)


result_path = "../data/DB100K-zero-shot/TRIPLE_result/result.json"

with open(result_path, 'r') as f:
    result = json.load(f)
unseen_relation = get_unseen_relation()
print(len(unseen_relation))

# with open(unseen_relation_path, "r") as f:
#     unseen_relation = []
#     unseen = json.load(f)
#     for k, v in unseen.items():
#         unseen_relation.append(v)
# print(unseen_relation)

seen_mmr = []
un_mmr = []
seen_hit1 = []
un_hit1 = []
seen_hit3 = []
un_hit3 = []
seen_hit10 = []
un_hit10 = []
print(result[1])

for i in result:
    h, r, t = i["Triple"]
    if r in unseen_relation:
        un_mmr.append(i["MRR"])
        un_hit1.append(i["HITS@1"])
        un_hit3.append(i["HITS@3"])
        un_hit10.append(i["HITS@10"])
    else:
        seen_mmr.append(i["MRR"])
        seen_hit1.append(i["HITS@1"])
        seen_hit3.append(i["HITS@3"])
        seen_hit10.append(i["HITS@10"])

print("unseen:")
print("MMR:{}".format(sum(un_mmr) / len(un_mmr)))
print("HITS@1:{}".format(sum(un_hit1) / len(un_hit1)))
print("HITS@3:{}".format(sum(un_hit3) / len(un_hit3)))
print("HITS@10:{}".format(sum(un_hit10) / len(un_hit10)))

print("unseen:")
print("MMR:{}".format(sum(seen_mmr) / len(seen_mmr)))
print("HITS@1:{}".format(sum(seen_hit1) / len(seen_hit1)))
print("HITS@3:{}".format(sum(seen_hit3) / len(seen_hit3)))
print("HITS@10:{}".format(sum(seen_hit10) / len(seen_hit10)))

print("all")
print(sum(un_mmr + seen_mmr) / (len(un_mmr) + len(seen_mmr)))
print(sum(un_hit1 + seen_hit1) / (len(un_hit1) + len(seen_hit1)))
print(sum(un_hit3 + seen_hit3) / (len(un_hit3) + len(seen_hit3)))
print(sum(un_hit10 + seen_hit10) / (len(un_hit10) + len(seen_hit10)))
