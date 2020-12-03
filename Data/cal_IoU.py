

def IoU(box1, box2):
    inter = max( min(box1[2], box2[2]) - max(box1[0], box2[0]), 0) * max( min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    box1_ = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_ = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (box1_ + box2_ - inter)


def count_TP_FP(predict_boxes, real_boxes):
    TP = 0
    FP = 0

    # 모든 예측한 박스에 대해서
    for predict_box in predict_boxes:
        find = False

        # 모든 실제 박스들과 비교했을 때
        for real_box in real_boxes:

            # IoU가 0.5 이상이면 찾았다고 하고 loop 탈출
            if IoU(predict_box, real_box) > 0.5:
                find = True
                break

        # 예측한 박스와 겹치는 실제 박스를 찾았으면 TP 1 증가, 못찾았으면 엉뚱한 예측을 했으므로 FP 1 증가
        if find:
            TP += 1
        else:
            FP += 1

    return TP, FP

def cal_precision(pred_boxes, real_boxes):
    TP = 0.0
    FP = 0.0

    for i in range(len(pred_boxes)):
        TP += count_TP_FP(pred_boxes[i], real_boxes[i])[0]
        FP += count_TP_FP(pred_boxes[i], real_boxes[i])[1]

    print("TP/TP+FP=", round(TP/(TP+FP),3))
    return round(TP/(TP+FP),3)



print(cal_precision([[[389, 66, 436, 125], [138, 48, 196, 115], [455, 163, 504, 223], [786, 35, 835, 85], [646, 47, 695, 100], [226, 94, 276, 152]]],
              [[[370, 60, 400, 100], [120, 40, 180, 100], [450, 163, 504, 223], [780, 35, 835, 85], [646, 40, 695, 100], [200, 94, 276, 152]]]))