from __future__ import annotations

from typing import Tuple
import pandas as pd

__all__ = ['NoCIoUEvaluator']

class NoCIoUEvaluator:

    def __init__(
        self,
        result_file: str,
        labels: list[str],
        max_iou: list[float]):

        self.max_iou = max_iou
        self.labels = labels
        self.result_file = result_file

    def _eval_per_class(self, label: str, max_iou: float) -> Tuple[int, int, dict[str, float], dict[str, float]]:
        results_dict_KatIOU = {}
        num_objects = 0

        results_dict_per_click = {}
        results_dict_per_click_iou = {}

        with open(self.result_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                splits = line.rstrip().split(' ')
                object_id = splits[0]
                label_name = splits[1]
                num_clicks = splits[2]
                iou=splits[3]

                if label_name == label:

                    if float(iou)>=max_iou:
                        if (label_name+'_'+object_id) not in results_dict_KatIOU:
                            results_dict_KatIOU[label_name+'_'+object_id]=float(num_clicks)
                            num_objects+=1

                    elif int(num_clicks)>=20 and (float(iou)>=0):
                        if (label_name+'_'+object_id) not in results_dict_KatIOU:
                            results_dict_KatIOU[label_name+'_'+object_id] = float(num_clicks)
                            num_objects += 1

                    results_dict_per_click.setdefault(num_clicks, 0)
                    results_dict_per_click_iou.setdefault(num_clicks, 0)

                    results_dict_per_click[num_clicks]+=1
                    results_dict_per_click_iou[num_clicks]+=float(iou)

        if len(results_dict_KatIOU.values())==0:
            print('no objects to eval')
            return 0, 0, {}, {}

        click_at_IoU =sum(results_dict_KatIOU.values())/len(results_dict_KatIOU.values())
        print('click@', max_iou, click_at_IoU, num_objects, len(results_dict_KatIOU.values()))

        return sum(results_dict_KatIOU.values()), len(results_dict_KatIOU.values()), results_dict_per_click_iou, results_dict_per_click 


    def eval_results(self) -> str:
        print('--------- Evaluating -----------')
        NOC: dict[float, list[float]] = {}
        NOO: dict[float, list[float]] = {}
        if self.labels == {}:
            return ''

        for iou_max in self.max_iou:
            NOC[iou_max] = []
            NOO[iou_max] = []
            IOU_PER_CLICK_dict = None
            NOO_PER_CLICK_dict = None

            for label in self.labels:    
                print(f'eval per class for {label}')
                noc_perclass, noo_perclass, iou_per_click, noo_per_click = self._eval_per_class(label[0], iou_max)
                NOC[iou_max].append(noc_perclass)
                NOO[iou_max].append(noo_perclass)

                if IOU_PER_CLICK_dict is None:
                    IOU_PER_CLICK_dict = iou_per_click
                else:
                    for k in IOU_PER_CLICK_dict.keys():
                        IOU_PER_CLICK_dict[k] += iou_per_click[k]

                if NOO_PER_CLICK_dict is None:
                    NOO_PER_CLICK_dict = noo_per_click
                else:
                    for k in NOO_PER_CLICK_dict.keys():
                        NOO_PER_CLICK_dict[k] += noo_per_click[k]
        
        assert IOU_PER_CLICK_dict is not None
        assert NOO_PER_CLICK_dict is not None

        results_dict = {
            'NoC@80': sum(NOC[0.8])/sum(NOO[0.8]),
            'NoC@85': sum(NOC[0.85])/sum(NOO[0.85]),
            'NoC@90': sum(NOC[0.9])/sum(NOO[0.9]),
            'IoU@1': IOU_PER_CLICK_dict['1']/NOO_PER_CLICK_dict['1'],
            'IoU@2': IOU_PER_CLICK_dict['2']/NOO_PER_CLICK_dict['2'],
            'IoU@3': IOU_PER_CLICK_dict['3']/NOO_PER_CLICK_dict['3'],
            'IoU@5': IOU_PER_CLICK_dict['5']/NOO_PER_CLICK_dict['5'],
            'IoU@10': IOU_PER_CLICK_dict['10']/NOO_PER_CLICK_dict['10'],
            'IoU@15': IOU_PER_CLICK_dict['15']/NOO_PER_CLICK_dict['15']
        }
        print('****************************')
        print(results_dict)
        df = pd.DataFrame([results_dict])

        return df.to_string()
