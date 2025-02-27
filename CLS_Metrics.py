import re
import json
import os


def CLS_Metrics(file_path, dataset):
    right = 0
    wrong = 0
    error = 0
    ERR = []
    good = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        new_lines = lines
        
        imgs = []
        print(len(new_lines)-1)
        for line in new_lines[1:]:
            print(line.replace('\n', ''))
            img = line.split('@#@')[0]

            if dataset== 'MVTec':
                temp = line.split('@#@')
                label = temp[0].split('/')[-2]
                label = label.lower()
                answer = temp[-1]
                answer = answer.lower()
                answers = re.findall('[a-z]+', answer)

                if 'yes' in answers:
                    if label == 'good':
                        fp = fp + 1
                    elif label != 'good':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'good':
                        tn = tn + 1
                    elif label != 'good':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)


                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))


            elif dataset=='VisA':
                temp = line.split('@#@')
                label = temp[0].split('/')[-2]
                label = label.lower()
                answer = temp[-1]
                answer = answer.lower()
                answers = re.findall('[a-z]+', answer)

                if 'yes' in answers:
                    if label == 'normal':
                        fp = fp + 1
                    elif label != 'normal':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'normal':
                        tn = tn + 1
                    elif label != 'normal':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
                     
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))

            elif dataset=='SOC':
                answer = line.split('@#@')[-1]
                answer = answer.lower()
                answers = re.findall('[a-z]+', answer)

                img = os.path.basename(img)
                temp = 'D:\work\work_py\GPT-4V-AD-main\\bbox_image\SOC\\' + img.replace('.jpg', '.png')
                if os.path.exists(temp):
                    label = 'positive'
                else:
                    label = 'negative'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    # print(line)
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))
            
            elif dataset== 'CP-CHILD-B':
                answer = line.split('@#@')[-1]
                answer = answer.lower()
                answers = re.findall('[a-z]+', answer)

                if img.find('0 (') != -1:
                    label = 'negative'
                else:
                    label = 'positive'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))

            elif dataset=='COD10K':
                temp = line.split('@#@')
                label = temp[0].find('NonCAM')
                answer = temp[-1]
                answer = answer.lower()
                answers = re.findall('[a-z]+', answer)

                if 'yes' in answers:
                    if label != -1:
                        fp = fp + 1
                    elif label == -1:
                        tp = tp + 1
                elif 'no' in answers:
                    if label != -1:
                        tn = tn + 1
                    elif label == -1:
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
                     
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(str(label) + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))


            else:
                answer = line.split('@#@')[-1]
                answer = answer.lower()
                answers = re.findall('[a-z]+', answer)

                if 'yes' in answers:
                    right = right + 1
                elif 'no' in answers:
                    wrong = wrong + 1
                else:
                    error = error + 1
                    ERR.append(line)

                print('answer: {}, right: {}, wrong: {}, error: {} \n'.format(answer.replace('\n', ''), right, wrong, error))


        print('######################## {}'.format(len(ERR)))
        for e in ERR:
            print(e)

        finished = 1

def ClS_Metrics_json(file_path, dataset):
    right = 0
    wrong = 0
    error = 0
    ERR = []
    good = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()

        imgs = []
        for line in lines:
            record  = json.loads(line)
            
            img = record['img']

            answer = record['answer']
            answer = answer.lower()
            answers = re.findall('[a-z]+', answer)
            
                
            if dataset== 'COD10K':
                if img.find('NonCAM') != -1:
                    label = 'negative'
                else:
                    label = 'positive'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))

            elif dataset== 'SOC':
                img = os.path.basename(img)
                temp = 'D:\work\work_py\GPT-4V-AD-main\\bbox_image\SOC\\' + img.replace('.jpg', '.png')
                if os.path.exists(temp):
                    label = 'positive'
                else:
                    label = 'negative'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))
            
            elif dataset== 'CP-CHILD-B':
                if img.find('Non-Polyp') != -1:
                    label = 'negative'
                else:
                    label = 'positive'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))

            elif dataset == 'MVTec':
                temp = img.split('/')[-2]
                if temp == 'good':
                    label = 'negative'
                else:
                    label = 'positive'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))

            elif dataset == 'VisA':
                temp = img.split('/')[-2]
                if temp == 'Normal':
                    label = 'negative'
                else:
                    label = 'positive'

                if 'yes' in answers:
                    if label == 'negative':
                        fp = fp + 1
                    elif label != 'negative':
                        tp = tp + 1
                elif 'no' in answers:
                    if label == 'negative':
                        tn = tn + 1
                    elif label != 'negative':
                        fn = fn + 1
                else:
                    error = error + 1
                    ERR.append(line)
            
                print('answer: {}, tp: {}, tn: {}, fp: {}, fn: {}, err: {} \n'.format(label + '||' + answer.replace('\n', '') , tp, tn, fp, fn, error))

            else:
                if 'yes' in answers:
                    right = right + 1
                elif 'no' in answers:
                    wrong = wrong + 1
                
                print('answer: {}, right: {}, wrong: {}, error: {} \n'.format(answer.replace('\n', ''), right, wrong, error))

        print('######################## {}'.format(len(ERR)))
        for e in ERR:
            print(e)

        finished = 1




if __name__=='__main__':
    file_name = 'results/CLS/MiniGPT-v2/Trans10k.lst'
    dataset_name = file_name.split('/')[-1]
    dataset_name = dataset_name.split('.')[0]

    if file_name.find('json') == -1:
        CLS_Metrics(file_path=file_name, dataset=dataset_name)
    else:
        ClS_Metrics_json(file_path=file_name, dataset=dataset_name)